# inference.py - Optimized Logging with Dynamic HEF Configuration
# INFO: Essential operational status, startup, major state changes
# DEBUG: Detailed diagnostics, frame processing, inference metrics
# ERROR: Critical failures requiring attention, includes context for debugging

import cv2
import numpy as np
import time
import random
import subprocess
import re
from datetime import datetime
import hailo_platform as hpf
import socket
import paho.mqtt.client as mqtt
import json
import argparse
import logging
from multiprocessing import Queue as MPQueue
from queue import Empty
from scipy.special import expit as sigmoid
from config import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_MAX,CONFIDENCE_THRESHOLD, HEF_PATH, WEBCAM_INDEX, RTSP_URL,
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, SHOW_IN_GUI, INCLUDE_METADATA,
    MAX_CAPTURE_OPEN_RETRIES, CAPTURE_OPEN_RETRY_DELAY, MAX_READ_RETRIES,
    PLACEHOLDER_FRAME_DELAY, RTSP_RECONNECT_DELAY, RTSP_TRANSPORT,
    NMS_IOU_THRESHOLD, USE_GSTREAMER, LOG_LEVEL, MQTT_USER, MQTT_PASS)
from urllib.parse import urlparse, quote
import os
import threading


# Shared thermal state - written by inference loop, read by API/health
_hailo_thermal = {"ts0": None, "ts1": None, "status": "unknown"}

logging.basicConfig(
    level=LOG_LEVEL,
    format='[%(levelname)s] [%(asctime)s] [%(module)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/hailo_inference.log')
    ]
)
logger = logging.getLogger(__name__)

# Class definitions
class_names = ["blue helmet", "head", "red helmet", "white helmet", "yellow helmet"]

# YOLOv5 standard anchors by stride
YOLO_ANCHORS = {
    8: [[10, 13], [16, 30], [33, 23]],
    16: [[30, 61], [62, 45], [59, 119]],
    32: [[116, 90], [156, 198], [373, 326]]
}

# Layer name patterns for stride detection
STRIDE_PATTERNS = {
    'conv47': 8, 'conv54': 16, 'conv60': 32,  # YOLOv5s
    'conv65': 8, 'conv74': 16, 'conv82': 32,  # YOLOv5m
    'output1': 8, 'output2': 16, 'output3': 32  # Generic
}

def extract_base_model_name(hef_filename):
    """Extract base model name from HEF filename, removing timestamps."""
    basename = os.path.basename(hef_filename)
    name_without_ext = basename.replace('.hef', '')
    base_name = re.sub(r'_\d{8}_\d{6}$', '', name_without_ext)
    base_name = re.sub(r'_\d{8}$', '', base_name)
    return base_name

def detect_stride_from_layer_name(layer_name):
    """Detect stride from layer name using pattern matching."""
    for pattern, stride in STRIDE_PATTERNS.items():
        if pattern in layer_name:
            return stride
    return None

def detect_stride_from_shape(output_shape):
    """Detect stride from output shape by comparing to input size."""
    if len(output_shape) >= 2:
        grid_h = output_shape[0]
        stride = FRAME_HEIGHT // grid_h
        if stride in YOLO_ANCHORS:
            return stride
    return None

def build_output_configs_from_hef(hef):
    """Dynamically build output configurations from HEF metadata."""
    output_vstream_infos = hef.get_output_vstream_infos()
    configs = []

    logger.info(f"Detected {len(output_vstream_infos)} output layers from HEF:")

    for info in output_vstream_infos:
        layer_name = info.name
        output_shape = info.shape

        logger.info(f"  - {layer_name}: shape={output_shape}")

        stride = detect_stride_from_layer_name(layer_name)
        if stride is None:
            stride = detect_stride_from_shape(output_shape)
        if stride is None:
            stride = [8, 16, 32][min(len(configs), 2)]
            logger.warning(f"Using fallback stride {stride} for layer {layer_name}")

        anchors = YOLO_ANCHORS.get(stride, YOLO_ANCHORS[8])
        config = {'name': layer_name, 'stride': stride, 'anchors': anchors}
        configs.append(config)
        logger.info(f"    → Mapped to stride={stride}, anchors={anchors}")

    configs.sort(key=lambda x: x['stride'])
    return configs

base_model_name = extract_base_model_name(HEF_PATH)
logger.info(f"Extracted base model name: '{base_model_name}' from HEF: {os.path.basename(HEF_PATH)}")

def get_current_timestamp():
    """Returns the current timestamp in milliseconds."""
    return int(time.time() * 1000)  # ponytail: datetime.utcnow().timestamp() is wrong in non-UTC containers

def get_hailo_metadata() -> dict:
    """Extracts metadata from Hailo device using hailortcli command."""
    try:
        output = subprocess.check_output(["hailortcli", "fw-control", "identify"], text=True, timeout=5)
        metadata = {
            re.sub(r"\s+", "_", key.strip().lower()): value.replace("\x00", "").strip()
            for line in output.splitlines() if ":" in line
            for key, value in [line.split(":", 1)]
        }
        metadata["timestamp"] = get_current_timestamp()
        return metadata
    except Exception as e:
        logger.error(f"Failed to retrieve Hailo metadata: {e}", exc_info=True)
        return {"error": f"Failed to get Hailo metadata: {e}"}

def get_hailo_thermal() -> dict:
    """Returns latest Hailo chip temperature and status from the shared state."""
    return dict(_hailo_thermal)

def _thermal_status(ts0: float, ts1: float) -> str:
    # ponytail: thresholds match HailoRT health monitor zones (orange=95, red=110)
    t = max(ts0, ts1)
    if t >= 110:
        return "red"
    if t >= 95:
        return "yellow"
    return "green"

def decode_output(output, stride, anchors):
    """Decodes YOLOv5m output into bounding boxes, confidences, and class IDs."""
    if output.dtype == np.uint8:
        output = (output.astype(np.float32) - 128.0) / 16.0
    logger.debug(f"Output range: [{output.min():.4f}, {output.max():.4f}] dtype={output.dtype}")

    if len(output.shape) == 4:
        output = np.squeeze(output, axis=0)

    if len(output.shape) != 3:
        logger.error(f"Invalid output shape: {output.shape}")
        return np.array([])

    grid_h, grid_w, channels = output.shape
    num_anchors = len(anchors)
    entry_size = channels // num_anchors
    expected_entry_size = 5 + len(class_names)

    if entry_size != expected_entry_size:
        logger.error(f"Output entry size mismatch: {entry_size} vs {expected_entry_size}")
        return np.array([])

    output = output.reshape(grid_h, grid_w, num_anchors, entry_size)
    output_sig = sigmoid(output)

    tx, ty, tw, th, to = output_sig[..., 0], output_sig[..., 1], output_sig[..., 2], output_sig[..., 3], output_sig[..., 4]
    class_probs = output_sig[..., 5:]

    cx = np.arange(grid_w).reshape(1, grid_w, 1).repeat(grid_h, axis=0).repeat(num_anchors, axis=2)
    cy = np.arange(grid_h).reshape(grid_h, 1, 1).repeat(grid_w, axis=1).repeat(num_anchors, axis=2)

    x = (tx * 2 - 0.5 + cx) * stride
    y = (ty * 2 - 0.5 + cy) * stride
    w = (tw * 2) ** 2 * np.array(anchors)[:, 0].reshape(1, 1, num_anchors)  # ponytail: anchors already in pixels, stride is for x/y grid→pixel only
    h = (th * 2) ** 2 * np.array(anchors)[:, 1].reshape(1, 1, num_anchors)

    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2

    conf = to * np.max(class_probs, axis=-1)
    class_id = np.argmax(class_probs, axis=-1)

    detections = np.stack([x1, y1, x2, y2, conf, class_id], axis=-1).reshape(-1, 6)

    detections[:, 0] = np.clip(detections[:, 0], 0, FRAME_WIDTH)
    detections[:, 1] = np.clip(detections[:, 1], 0, FRAME_HEIGHT)
    detections[:, 2] = np.clip(detections[:, 2], detections[:, 0], FRAME_WIDTH)
    detections[:, 3] = np.clip(detections[:, 3], detections[:, 1], FRAME_HEIGHT)

    detections = detections[
        (detections[:, 2] - detections[:, 0] < FRAME_WIDTH * 0.9) &
        (detections[:, 3] - detections[:, 1] < FRAME_HEIGHT * 0.9)
    ]
    MAX_BOX_SIZE = min(FRAME_WIDTH, FRAME_HEIGHT) * FRAME_MAX  # 50% of smaller dimension
    box_widths = detections[:, 2] - detections[:, 0]
    box_heights = detections[:, 3] - detections[:, 1]
    detections = detections[
        (box_widths < MAX_BOX_SIZE) &
        (box_heights < MAX_BOX_SIZE) &
        (box_widths > 20) &
        (box_heights > 20)
    ]

    # ponytail: filter edge-clipped artifacts - letterbox padding activates helmet filters
    # at frame boundaries producing tiny high-confidence ghost boxes. Real objects have
    # substantial size even when touching the edge (real helmet clipped at top: height≈76px).
    if len(detections):
        x1, y1 = detections[:, 0], detections[:, 1]
        x2, y2 = detections[:, 2], detections[:, 3]
        w = x2 - x1; h = y2 - y1
        edge = 8  # px tolerance
        at_lr = (x1 < edge) | (x2 > FRAME_WIDTH - edge)
        at_tb = (y1 < edge) | (y2 > FRAME_HEIGHT - edge)
        ghost = (at_lr & (w < 60)) | (at_tb & (h < 60))
        detections = detections[~ghost]

    return detections

def publish_inference_data(client, data):
    """Publishes inference data to MQTT topic and flushes immediately for <20ms latency."""
    try:
        if not client.is_connected():
            client.reconnect()
        client.publish(MQTT_TOPIC, json.dumps(data))
        client.loop_write()  # ponytail: force socket flush instead of waiting for background loop (~100ms)
    except Exception as e:
        logger.error(f"MQTT publish failed: {type(e).__name__}")

def compute_iou(box, boxes):
    """Compute IoU for one box vs many."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / union

def initialize_webcam(webcam_index):
    """Initialize webcam with V4L2, fallback to available devices or test image."""
    logger.info(f"Initializing webcam at index {webcam_index}")

    available_devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            available_devices.append(i)
            cap.release()

    if available_devices:
        logger.info(f"Available video devices: {available_devices}")

    for attempt in range(MAX_CAPTURE_OPEN_RETRIES):
        cap = cv2.VideoCapture(webcam_index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"Webcam {webcam_index} initialized: {FRAME_WIDTH}x{FRAME_HEIGHT}@30fps")
            return cap, False
        time.sleep(CAPTURE_OPEN_RETRY_DELAY)

    logger.warning(f"Failed to open webcam {webcam_index}")

    test_image_path = "/local/workspace/share/test.jpg"
    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        if frame is not None:
            logger.info("Using static test image as fallback")
            return frame, True

    raise RuntimeError("No video source available")

def letterbox_resize(img, target_h=640, target_w=640):
    """YOLOv5 letterbox resize with padding value 114(=gray)"""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas with padding value 114
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas

def run_inference_main(use_webcam=False, frame_queue=None, rtsp_url=None):
    """Main inference loop with Hailo pipeline, MQTT publishing, and frame queuing."""
    logger.info("=" * 80)
    logger.info("Starting Hailo YOLOv5m Inference Pipeline")
    logger.info(f"Mode: {'Webcam' if use_webcam else 'RTSP'}")
    logger.info(f"HEF Model: {HEF_PATH}")
    logger.info(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    logger.info("=" * 80)

    stop_event = threading.Event()
    frame_times = []
    last_time = time.time()

    # Initialize MQTT
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        logger.info("MQTT client connected")
    except Exception as e:
        logger.error(f"MQTT connection failed: {type(e).__name__}")

    # Load HEF
    logger.info("Loading HEF model")
    hef = hpf.HEF(HEF_PATH)
    logger.info("HEF model loaded successfully")

    # Build dynamic output configuration
    output_configs = build_output_configs_from_hef(hef)
    logger.info(f"Output configuration built: {len(output_configs)} layers")

    with hpf.VDevice() as target:
        _phys_devices = target.get_physical_devices()
        configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()
        input_vstream_info = hef.get_input_vstream_infos()[0]

        if input_vstream_info.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise ValueError(f"Input shape mismatch: {input_vstream_info.shape}")

        logger.info(f"HEF input validated: {input_vstream_info.shape}")
        logger.info(f"Using {len(output_configs)} output layers")

        #input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            #network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group,
            quantized=True,
            format_type=hpf.FormatType.UINT8)
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=hpf.FormatType.UINT8)

        with network_group.activate(network_group_params):
            logger.info("Hailo network activated - Ready for inference")

            with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                cap = None
                raw_frame_queue = MPQueue(maxsize=10)

                def capture_thread(source, is_fallback_image=False):
                    nonlocal cap
                    while not stop_event.is_set():
                        if is_fallback_image:
                            try:
                                raw_frame_queue.put(source, timeout=0.1)
                            except:
                                pass
                            time.sleep(PLACEHOLDER_FRAME_DELAY)
                            continue

                        if cap is None or not cap.isOpened():
                            cap, is_fallback = initialize_webcam(WEBCAM_INDEX)
                            if is_fallback:
                                is_fallback_image = True
                                source = cap
                                continue

                        ret, frame = cap.read()
                        if ret:
                            try:
                                raw_frame_queue.put(frame, timeout=0.01)
                            except:
                                pass
                        else:
                            logger.warning("Frame read failed")
                            if cap:
                                cap.release()
                            cap = None
                            time.sleep(1)

                try:
                    if use_webcam:
                        cap, is_fallback = initialize_webcam(WEBCAM_INDEX)
                        threading.Thread(target=capture_thread, args=(cap if is_fallback else WEBCAM_INDEX, is_fallback), daemon=True).start()
                        logger.info("Webcam capture thread started")
                    else:
                        # ═══════════════════════════════════════════════════════════
                        # RTSP MODE - FFmpeg Only
                        # ═══════════════════════════════════════════════════════════
                        rtsp_url_to_use = rtsp_url if rtsp_url else RTSP_URL
                        
                        if not rtsp_url_to_use:
                            logger.error("RTSP mode selected but no RTSP_URL provided!")
                            raise ValueError("RTSP_URL is required for RTSP mode")
                        
                        logger.info(f"Configuring RTSP input: {rtsp_url_to_use}")
                        
                        # Parse and encode RTSP URL
                        parsed = urlparse(rtsp_url_to_use)
                        username = parsed.username or ""
                        password = parsed.password or ""
                        host = parsed.hostname
                        port = parsed.port or 554
                        path = parsed.path or ""
                        
                        # URL-encode username and password (handles special chars like !)
                        quoted_username = quote(username, safe='')
                        quoted_password = quote(password, safe='')
                        rtsp_url_encoded = f"rtsp://{quoted_username}:{quoted_password}@{host}:{port}{path}"
                        
                        logger.info(f"Parsed RTSP - Host: {host}:{port}, Path: {path}")
                        
                        cap_opened = False
                        
                        # Configure FFmpeg options for RTSP
                        logger.info("Attempting RTSP connection via FFmpeg")
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                            f"rtsp_transport;{RTSP_TRANSPORT}|"
                            f"reorder_queue_size;5|"
                            f"async;1|"
                            f"timeout;15000000"  # 15 second timeout in microseconds
                        )
                        
                        for attempt in range(MAX_CAPTURE_OPEN_RETRIES):
                            logger.debug(f"FFmpeg connection attempt {attempt+1}/{MAX_CAPTURE_OPEN_RETRIES}")
                            
                            try:
                                cap = cv2.VideoCapture(rtsp_url_encoded, cv2.CAP_FFMPEG)
                                
                                # Configure capture properties
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
                                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
                                
                                if cap.isOpened():
                                    # Test read to verify stream is working
                                    ret, test_frame = cap.read()
                                    
                                    if ret and test_frame is not None:
                                        actual_w = test_frame.shape[1]
                                        actual_h = test_frame.shape[0]
                                        logger.info("    RTSP opened successfully via FFmpeg")
                                        logger.info(f"   Resolution: {actual_w}x{actual_h}")
                                        logger.info(f"   Frame shape: {test_frame.shape}")
                                        
                                        cap_opened = True
                                        
                                        # Update capture_thread to handle RTSP
                                        def rtsp_capture_thread():
                                            """RTSP-specific capture thread"""
                                            nonlocal cap
                                            while not stop_event.is_set():
                                                if cap is None or not cap.isOpened():
                                                    logger.warning("RTSP connection lost - Reconnecting...")
                                                    time.sleep(RTSP_RECONNECT_DELAY)
                                                    
                                                    cap = cv2.VideoCapture(rtsp_url_encoded, cv2.CAP_FFMPEG)
                                                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
                                                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
                                                    
                                                    if not cap.isOpened():
                                                        logger.error("RTSP reconnection failed")
                                                        continue
                                                    
                                                    logger.info("RTSP reconnected successfully")
                                                
                                                ret, frame = cap.read()
                                                if ret and frame is not None:
                                                    try:
                                                        raw_frame_queue.put(frame, timeout=0.01)
                                                    except:
                                                        pass  # Queue full
                                                else:
                                                    logger.warning("RTSP frame read failed")
                                                    if cap:
                                                        cap.release()
                                                    cap = None
                                                    time.sleep(1)
                                        
                                        # Start RTSP capture thread
                                        threading.Thread(target=rtsp_capture_thread, daemon=True).start()
                                        logger.info("RTSP capture thread started")
                                        break
                                    else:
                                        logger.debug("FFmpeg opened but test read failed")
                                        cap.release()
                                        cap = None
                                else:
                                    logger.debug(f"FFmpeg failed to open stream (attempt {attempt+1})")
                            
                            except Exception as e:
                                logger.error(f"FFmpeg connection error: {type(e).__name__}: {e}")
                            
                            if attempt < MAX_CAPTURE_OPEN_RETRIES - 1:
                                logger.debug(f"Retrying in {CAPTURE_OPEN_RETRY_DELAY} seconds...")
                                time.sleep(CAPTURE_OPEN_RETRY_DELAY)
                        
                        # Fallback to placeholder if connection fails
                        if not cap_opened:
                            logger.error(f"  Failed to open RTSP stream after {MAX_CAPTURE_OPEN_RETRIES} retries")
                            logger.error("   Using placeholder frames - Check camera URL and network")
                            
                            # Start placeholder thread
                            def placeholder_thread():
                                while not stop_event.is_set():
                                    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
                                    cv2.putText(
                                        placeholder, 
                                        "No RTSP Connection", 
                                        (50, FRAME_HEIGHT // 2 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        1.0, 
                                        (255, 255, 255), 
                                        2
                                    )
                                    cv2.putText(
                                        placeholder, 
                                        f"Check: {rtsp_url_to_use[:30]}...", 
                                        (20, FRAME_HEIGHT // 2 + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, 
                                        (200, 200, 200), 
                                        1
                                    )
                                    if frame_queue:
                                        try:
                                            frame_queue.put(placeholder, timeout=0.01)
                                        except:
                                            pass
                                    time.sleep(PLACEHOLDER_FRAME_DELAY)
                            
                            threading.Thread(target=placeholder_thread, daemon=True).start()
                            logger.warning("Placeholder thread started - Inference will not run")
                    logger.info("Entering main inference loop")
                    frame_count = 0
                    fps_log_time = time.time()
                    fps_sum = 0.0
                    fps_count = 0
                    last_frame = None
                    detection_count_total = 0
                    detection_count_window = 0
                    latest_confidence = 0.0
                    max_confidence_window = 0.0
                    # ponytail: per-class EMA tracker {class_id: [[x1,y1,x2,y2], ...]}
                    SMOOTH_ALPHA = 0.35  # ponytail: tune lower for smoother, higher for more responsive
                    tracked: dict[int, list] = {}

                    placeholder_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
                    cv2.putText(placeholder_frame, "Waiting for frames...", (50, FRAME_HEIGHT // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    while not stop_event.is_set():
                        try:
                            frame = raw_frame_queue.get(timeout=0.5)
                            last_frame = frame
                        except Empty:
                            frame = last_frame if last_frame is not None else placeholder_frame

                        if use_webcam:
                            frame = cv2.flip(frame, 1)


                        frame = letterbox_resize(frame, FRAME_HEIGHT, FRAME_WIDTH)
                        # Convert BGR to RGB for inference (matches calibration)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        annotated = frame.copy()  # Still BGR - correct for cv2.rectangle!
                        input_data = {input_vstream_info.name: np.expand_dims(frame_rgb, axis=0)}

                        try:
                            results = infer_pipeline.infer(input_data)
                        except Exception as e:
                            logger.error(f"Inference failed: {type(e).__name__}")
                            if 'Timeout' in type(e).__name__ or 'TIMEOUT' in str(e):
                                # ponytail: break exits InferVStreams context (reinitializes pipeline buffers);
                                # continue retries with stuck pipeline → infinite 10s timeout loop
                                logger.warning("Hailo pipeline stuck - reinitializing")
                                break
                            continue

                        all_detections = []
                        for config in output_configs:
                            output = results[config['name']]
                            detections = decode_output(output, config['stride'], config['anchors'])
                            if len(detections) > 0:
                                all_detections.append(detections)

                        if all_detections:
                            all_detections = np.concatenate(all_detections, axis=0)
                        else:
                            all_detections = np.array([], dtype=np.float64).reshape(0, 6)

                        if len(all_detections) > 0:
                            filtered_detections = all_detections[all_detections[:, 4] > CONFIDENCE_THRESHOLD]
                        else:
                            filtered_detections = np.array([], dtype=np.float64).reshape(0, 6)

                        current_time = time.time()
                        frame_time = current_time - last_time
                        last_time = current_time
                        frame_times.append(frame_time)

                        if len(frame_times) > 30:
                            frame_times.pop(0)

                        avg_frame_time = sum(frame_times) / len(frame_times)
                        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                        fps_sum += fps
                        fps_count += 1
                        frame_count += 1

                        # ponytail: every 30 frames ~= 1-2s at 20fps; temp changes slowly
                        if frame_count % 30 == 0 and _phys_devices:
                            try:
                                t = _phys_devices[0].get_chip_temperature()
                                ts0, ts1 = t.ts0_temperature, t.ts1_temperature
                                _hailo_thermal["ts0"] = round(ts0, 1)
                                _hailo_thermal["ts1"] = round(ts1, 1)
                                _hailo_thermal["status"] = _thermal_status(ts0, ts1)
                            except Exception:
                                pass

                        if current_time - fps_log_time >= 10:
                            avg_fps = fps_sum / fps_count
                            timestamp_human = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            logger.info(
                                f"Performance Report [{timestamp_human}] | "
                                f"FPS: {avg_fps:.2f} | Frames: {frame_count} | "
                                f"Detections: {detection_count_total} (Window: {detection_count_window}) | "
                                f"Confidence Latest: {latest_confidence:.3f} Max: {max_confidence_window:.3f}"
                            )
                            fps_sum = 0.0
                            fps_count = 0
                            fps_log_time = current_time
                            detection_count_window = 0
                            max_confidence_window = 0.0

                        if filtered_detections.size > 0:
                            # per-class NMS then top-1: guarantees exactly one box per class
                            # ponytail: class-agnostic NMSBoxes lets same-class duplicates survive;
                            # running per-class then argmax is the standard YOLO post-processing fix.
                            per_class_best = []
                            for cid in np.unique(filtered_detections[:, 5].astype(int)):
                                cls_dets = filtered_detections[filtered_detections[:, 5].astype(int) == cid]
                                indices = cv2.dnn.NMSBoxes(cls_dets[:, :4].tolist(), cls_dets[:, 4].tolist(), CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD)
                                if len(indices):
                                    survivors = cls_dets[indices.flatten()]
                                    per_class_best.append(survivors[np.argmax(survivors[:, 4])])

                            if per_class_best:
                                final_detections = np.array(per_class_best)
                                # class-agnostic overlap pass: different classes on the same object → keep highest conf
                                # ponytail: IoU>0.3 = meaningful spatial overlap, not two people standing apart
                                if len(final_detections) > 1:
                                    final_detections = final_detections[np.argsort(final_detections[:, 4])[::-1]]
                                    keep = np.ones(len(final_detections), dtype=bool)
                                    for i in range(len(final_detections)):
                                        if not keep[i]:
                                            continue
                                        iou = compute_iou(final_detections[i, :4], final_detections[i+1:, :4])
                                        for j, ov in enumerate(iou):
                                            if ov > NMS_IOU_THRESHOLD:
                                                keep[i + 1 + j] = False
                                    final_detections = final_detections[keep]
                            else:
                                final_detections = np.zeros((0, 6))

                            num_dets = len(final_detections)
                            detection_count_total += num_dets
                            detection_count_window += num_dets

                            if num_dets > 0:
                                latest_confidence = float(final_detections[-1][4])
                                current_max = float(np.max(final_detections[:, 4]))
                                max_confidence_window = max(max_confidence_window, current_max)

                            new_tracked: dict[int, list] = {}
                            for det in final_detections:
                                raw_box, score, class_id = det[:4], det[4], int(det[5])
                                cid = class_id
                                cx = (raw_box[0] + raw_box[2]) / 2
                                cy = (raw_box[1] + raw_box[3]) / 2
                                prev_boxes = tracked.get(cid, [])
                                best_idx, best_dist = -1, float('inf')
                                for pi, pb in enumerate(prev_boxes):
                                    pcx = (pb[0] + pb[2]) / 2
                                    pcy = (pb[1] + pb[3]) / 2
                                    d = (cx - pcx) ** 2 + (cy - pcy) ** 2
                                    if d < best_dist:
                                        best_dist, best_idx = d, pi
                                if best_idx >= 0 and best_dist < (FRAME_WIDTH * 0.3) ** 2:
                                    prev = np.array(prev_boxes[best_idx])
                                    box = SMOOTH_ALPHA * np.array(raw_box) + (1 - SMOOTH_ALPHA) * prev
                                    prev_boxes.pop(best_idx)
                                else:
                                    box = np.array(raw_box)
                                new_tracked.setdefault(cid, []).append(box.tolist())
                                x1, y1, x2, y2 = [int(v) for v in box]
                                label = f"{class_names[cid]}: {(score * 100):.1f}%"
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated, label, (x1, max(y1 - 10, 20)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            tracked = new_tracked

                            publish_inference_data(client, {
                                "timestamp": get_current_timestamp(),
                                "detections": [
                                    {"class": class_names[int(d[5])], "confidence": round(float(d[4]), 2), "box": [round(float(x), 1) for x in d[:4]]}
                                    for d in final_detections
                                ],
                                "fps": round(fps, 2),
                                "thermal": dict(_hailo_thermal),
                            })
                        else:
                            tracked.clear()
                            publish_inference_data(client, {"timestamp": get_current_timestamp(), "detections": [], "fps": round(fps, 2), "thermal": dict(_hailo_thermal)})

                        if SHOW_IN_GUI:
                            cv2.imshow("Inference", annotated)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            if frame_queue:
                                try:
                                    frame_queue.put(annotated, timeout=0.01)
                                except:
                                    pass

                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt - Shutting down")
                finally:
                    stop_event.set()
                    if cap and not isinstance(cap, np.ndarray):
                        cap.release()
                    cv2.destroyAllWindows()
                    client.loop_stop()
                    client.disconnect()
                    logger.info("Inference pipeline terminated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hailo Inference")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    args = parser.parse_args()

    try:
        run_inference_main(use_webcam=args.webcam)
    except Exception as e:
        logger.critical(f"Fatal error: {type(e).__name__}: {e}", exc_info=True)
        exit(1)
