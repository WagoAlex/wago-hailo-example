from multiprocessing import Lock
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
import asyncio
import uvicorn
import subprocess
import re
from datetime import datetime
import json
import cv2
import numpy as np
import time
import signal
from multiprocessing import Queue
import queue
import os
import logging
import threading
from config import REST_API_PORT, SHOW_IN_GUI, FRAME_WIDTH, FRAME_HEIGHT, LOG_LEVEL
from inference import get_hailo_metadata, get_hailo_thermal
logging.basicConfig(
    level=LOG_LEVEL,
    format='[%(levelname)s] [%(asctime)s] [%(module)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/hailo_api.log')
    ]
)
logger = logging.getLogger(__name__)
last_frame = None
frame_lock = Lock()
shutdown = False
stop_event = threading.Event()
app = FastAPI(
    title="WAGO AI Inference API",
    description="API for streaming real-time HLS video with yolov5 detection annotations from a Hailo-8 pipeline, and retrieving device metadata. Inference data is logged via a separate MQTT process (mqtt_app.py).",
    version="1"
)
global_camera_queues = None
global_thermal_state = None
ffmpeg_processes = {}
feeders = {}
stderr_loggers = {}
# ponytail: latest_frames lets MJPEG read independently of HLS; both share the queue but
# MJPEG never competes for frames - it just reads whatever the HLS feeder last wrote.
latest_frames: dict[int, bytes] = {}
def get_current_timestamp():
    return int(datetime.utcnow().timestamp() * 1000)
@app.get("/metadata")
def read_metadata():
    logger.info("Request to /metadata")
    return {"metadata": get_hailo_metadata()}
@app.get("/inference")
def get_inference():
    logger.info("Request to /inference")
    return {"message": "No inference data available. Data is logged via MQTT in mqtt_app.py."}
def get_camera_queues() -> list[Queue]:
    logger.debug("Checking camera queues")
    if global_camera_queues is None or not global_camera_queues:
        logger.warning("Inference pipeline not ready - No camera queues initialized")
        raise HTTPException(status_code=503, detail="Inference pipeline not ready. Frame queues not initialized.")
    return global_camera_queues
def signal_handler(sig, frame):
    logger.warning("Received SIGTERM - Shutting down streams gracefully")
    global shutdown
    shutdown = True
    stop_event.set()
signal.signal(signal.SIGTERM, signal_handler)
def start_hls(camera_id: int, frame_queue: Queue):
    if camera_id in ffmpeg_processes:
        logger.info(f"HLS already started for camera {camera_id} - Skipping")
        return
    logger.info(f"Starting HLS for camera {camera_id}. Queue size on start: {frame_queue.qsize()}")
    hls_dir = f"/tmp/hls_{camera_id}"
    logger.debug(f"Creating HLS directory: {hls_dir}")
    os.makedirs(hls_dir, exist_ok=True)
    # HLS Configs
    hls_time = os.environ.get('HLS_TIME', '0.1')
    hls_list_size = os.environ.get('HLS_LIST_SIZE', '60')
    hls_flags = os.environ.get('HLS_FLAGS', 'independent_segments+append_list+delete_segments')
    hls_segment_filename = os.environ.get('HLS_SEGMENT_FILENAME', f'{hls_dir}/playlist%d.ts')
    playlist_path = os.environ.get('HLS_PLAYLIST_PATH', f'{hls_dir}/playlist.m3u8')

    # General FFmpeg Configs
    ffmpeg_bin = os.environ.get('FFMPEG_BIN', 'ffmpeg')
    loglevel = os.environ.get('FFMPEG_LOGLEVEL', 'debug')
    overwrite_output = os.environ.get('FFMPEG_OVERWRITE_OUTPUT', '-y')

    input_format = os.environ.get('FFMPEG_INPUT_FORMAT', 'rawvideo')
    input_codec = os.environ.get('FFMPEG_INPUT_CODEC', 'rawvideo')
    input_pix_fmt = os.environ.get('FFMPEG_INPUT_PIX_FMT', 'bgr24')
    frame_width = os.environ.get('FRAME_WIDTH', '480')
    frame_height = os.environ.get('FRAME_HEIGHT', '640')
    framerate = os.environ.get('FRAME_RATE', '30')

    output_codec = os.environ.get('FFMPEG_OUTPUT_CODEC', 'libx264')
    preset = os.environ.get('FFMPEG_PRESET', 'veryfast')
    tune = os.environ.get('FFMPEG_TUNE', 'zerolatency')
    output_pix_fmt = os.environ.get('FFMPEG_OUTPUT_PIX_FMT', 'yuv420p')
    output_format = os.environ.get('FFMPEG_OUTPUT_FORMAT', 'hls')
    ffmpeg_cmd = [
    ffmpeg_bin,
    '-loglevel', loglevel,
    overwrite_output,
    '-f', input_format,
    '-vcodec', input_codec,
    '-pix_fmt', input_pix_fmt,
    '-s', f'{frame_width}x{frame_height}',
    '-r', framerate,
    '-i', '-',
    '-c:v', output_codec,
    '-preset', preset,
    '-tune', tune,
    '-pix_fmt', output_pix_fmt,
    '-f', output_format,
    '-hls_time', hls_time,
    '-hls_list_size', hls_list_size,
    '-hls_flags', hls_flags,
    '-hls_segment_filename', hls_segment_filename,
    playlist_path
    ]
    
    logger.debug(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    ffmpeg_processes[camera_id] = ffmpeg_process
    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    cv2.putText(placeholder, f"Initializing stream (Camera {camera_id})...", (50, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i in range(100):
        logger.debug(f"Camera {camera_id}: Writing initial placeholder {i+1}/100 to FFmpeg")
        try:
            ffmpeg_process.stdin.write(placeholder.tobytes())
            ffmpeg_process.stdin.flush()
            time.sleep(0.01)
        except Exception as e:
            logger.error(f"Camera {camera_id}: Failed to write placeholder {i+1}/100: {str(e)}", exc_info=True)
            break
    segment_files = [f for f in os.listdir(hls_dir) if f.endswith('.ts')]
    logger.debug(f"Camera {camera_id}: Segment files after placeholders: {segment_files}")
    def feeder():
        empty_attempts = 0
        max_retries = 2
        placeholder_logged = False
        local_last_frame = None
        last_log_time = time.time()
        frame_counter = 0 
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.01)
                empty_attempts = 0
                placeholder_logged = False
                with frame_lock:
                    last_frame = frame.copy()
                local_last_frame = last_frame
                _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                latest_frames[camera_id] = jpg.tobytes()
                frame_counter += 1
                current_time = time.time()
                if frame_counter % 100 == 0 or (current_time - last_log_time) > 10:
                    #logger.warning(f"Camera {camera_id}: Processed {frame_counter} frames, queue size: {frame_queue.qsize()}")
                    last_log_time = current_time
                ffmpeg_process.stdin.write(frame.tobytes())
                ffmpeg_process.stdin.flush()
                logger.debug(f"Camera {camera_id}: Frame written successfully")
            except queue.Empty:
                empty_attempts += 1
                if empty_attempts == max_retries:
                    if not placeholder_logged:
                        #logger.error(f"Camera {camera_id}: No frames after {max_retries} attempts; using last/placeholder")
                        placeholder_logged = True
                    if local_last_frame is not None:
                        #logger.info(f"Camera {camera_id}: Writing last frame to FFmpeg, size: {local_last_frame.size} bytes")
                        ffmpeg_process.stdin.write(local_last_frame.tobytes())
                    else:
                        #logger.info(f"Camera {camera_id}: Writing placeholder frame to FFmpeg, size: {placeholder.size} bytes")
                        ffmpeg_process.stdin.write(placeholder.tobytes())
                    ffmpeg_process.stdin.flush()
            except OSError as e:
                if e.errno == 32:  # Broken pipe
                    #logger.debug("Broken pipe on FFmpeg write - likely shutdown")
                    break
                else:
                    raise
            except Exception as e:
                logger.error(f"Camera {camera_id}: Feeder error - {str(e)}", exc_info=True)
                break
    t = threading.Thread(target=feeder, daemon=True)
    t.start()
    feeders[camera_id] = t
    def log_stderr():
        for line in iter(ffmpeg_process.stderr.readline, b''):
            logger.debug(f"FFmpeg camera {camera_id}: {line.decode().strip()}")
    stderr_t = threading.Thread(target=log_stderr, daemon=True)
    stderr_t.start()
    stderr_loggers[camera_id] = stderr_t
    start_time = time.time()
    while not os.path.exists(playlist_path) and time.time() - start_time < 30:
        logger.debug(f"Camera {camera_id}: Waiting for playlist (elapsed: {time.time() - start_time:.2f}s)")
        time.sleep(0.2)
    if not os.path.exists(playlist_path):
        logger.error(f"Camera {camera_id}: HLS playlist not generated in 30s. Creating placeholder .m3u8")
        with open(playlist_path, 'w') as f:
            f.write('#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n' +
                    '\n'.join([f'#EXTINF:2.0,\nplaylist{i}.ts' for i in range(60)]) + '\n#EXT-X-ENDLIST')
        for i in range(60):
            placeholder_ts = f"{hls_dir}/playlist{i}.ts"
            with open(placeholder_ts, 'wb') as f:
                f.write(b'Dummy TS data')
            logger.info(f"Camera {camera_id}: Placeholder .ts created at {placeholder_ts}")
    else:
        logger.info(f"Camera {camera_id}: HLS playlist generated at {playlist_path}")
        segment_files = [f for f in os.listdir(hls_dir) if f.endswith('.ts')]
        logger.info(f"Camera {camera_id}: Segment files after playlist: {segment_files}")
        with open(playlist_path, 'r') as f:
            m3u8_content = f.read()
        logger.debug(f"Camera {camera_id}: Generated .m3u8 content:\n{m3u8_content}")
@app.get(
    "/video/stream",
    response_class=Response,
    responses={
        200: {
            "content": {"application/vnd.apple.mpegurl": {}},
            "description": "HLS playlist (.m3u8) for video stream."
        },
        403: {
            "content": {"application/json": {"example": {"detail": "Streaming disabled when SHOW_IN_GUI=1"}}},
            "description": "Returned when streaming is disabled due to GUI mode."
        },
        503: {
            "content": {"application/json": {"example": {"detail": "No frames available in the inference pipeline"}}},
            "description": "Returned when the frame queue is empty or an error occurs in the inference pipeline."
        }
    },
    summary="Stream HLS video feed",
    description="Streams real-time HLS video with YOLOv5m-helmet detection annotations from the Hailo-8 pipeline. Requires SHOW_IN_GUI=0 in the environment configuration. Use ?camera_id=N for multi-camera selection."
)
async def video_feed(camera_id: int = 0, camera_queues: list[Queue] = Depends(get_camera_queues)):
    logger.info(f"Received request for /video/stream - camera_id: {camera_id}")
    try:
        if SHOW_IN_GUI:
            logger.warning("Streaming disabled - SHOW_IN_GUI=1")
            raise HTTPException(status_code=403, detail="Streaming disabled when SHOW_IN_GUI=1")
        if camera_id < 0 or camera_id >= len(camera_queues):
            logger.warning(f"Invalid camera_id: {camera_id} (available: 0-{len(camera_queues)-1})")
            raise HTTPException(status_code=400, detail=f"Invalid camera_id: {camera_id} (available: 0-{len(camera_queues)-1})")
        playlist_path = f"/tmp/hls_{camera_id}/playlist.m3u8"
        if not os.path.exists(playlist_path):
            logger.error(f"Playlist not found for camera {camera_id} at {playlist_path}")
            raise HTTPException(status_code=503, detail="HLS playlist not available. Check logs.")
        with open(playlist_path, 'r') as f:
            m3u8_content = f.read()
        logger.debug(f"Camera {camera_id}: Serving .m3u8 content:\n{m3u8_content}")
        logger.info(f"Serving playlist for camera {camera_id} from {playlist_path}")
        return FileResponse(playlist_path, media_type="application/vnd.apple.mpegurl")
    except HTTPException as he:
        logger.warning(f"HTTPException in video_feed: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error in video_feed for camera {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error in video stream: {str(e)}. Check container logs for traceback.")
@app.get("/stream/mjpeg/{camera_id}")
async def mjpeg_stream(camera_id: int):
    if not global_camera_queues or camera_id >= len(global_camera_queues):
        raise HTTPException(status_code=404, detail="Camera not found")
    async def generate():
        # ponytail: read latest_frames written by HLS feeder - no queue competition
        last_sent = None
        while True:
            jpg = latest_frames.get(camera_id)
            if jpg and jpg is not last_sent:
                last_sent = jpg
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            else:
                await asyncio.sleep(0.033)
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video/{segment_name:path}")
async def get_segment(segment_name: str, camera_id: int = Query(0)):
    logger.info(f"Received request for /video/{segment_name} - camera_id: {camera_id}")
    hls_dir = f"/tmp/hls_{camera_id}"
    # Normalize segment name to handle .ts or .ts.ts
    normalized_segment = re.sub(r'\.ts(\.ts)*$', '.ts', segment_name)
    segment_path = f"{hls_dir}/{normalized_segment}"
    logger.debug(f"Checking segment path: {segment_path}")
    if os.path.exists(segment_path):
        logger.info(f"Serving segment for camera {camera_id} from {segment_path}")
        return FileResponse(segment_path, media_type="video/MP2T")
    dir_contents = os.listdir(hls_dir) if os.path.exists(hls_dir) else []
    logger.error(f"Segment not found for camera {camera_id} at {segment_path}. Directory contents: {dir_contents}")
    raise HTTPException(status_code=404, detail=f"Segment not found: {normalized_segment}")
@app.get(
    "/health",
    response_model=dict,
    summary="Check service health",
    description="Returns the health status of the video streaming service, including queue status and Hailo device metadata."
)
async def health_check(camera_queues: list[Queue] = Depends(get_camera_queues)):
    logger.info("Received request for /health")
    statuses = []
    for idx, q in enumerate(camera_queues):
        queue_empty = q.empty()
        queue_size = q.qsize() if hasattr(q, 'qsize') else 'N/A'
        statuses.append({
            "camera_id": idx,
            "queue_empty": queue_empty,
            "queue_size": queue_size
        })
        logger.debug(f"Health check for camera {idx}: empty={queue_empty}, size={queue_size}")
    hailo_metadata = get_hailo_metadata()
    if "error" in hailo_metadata:
        logger.error(f"Hailo device error in health: {hailo_metadata['error']}")
        raise HTTPException(status_code=503, detail=f"Hailo device error: {hailo_metadata['error']}")
    status_msg = "healthy" if all(not s["queue_empty"] for s in statuses) else "warning: some queues empty"
    logger.info(f"Health status: {status_msg}")
    return {
        "status": status_msg,
        "cameras": statuses,
        "metadata": hailo_metadata,
        "thermal": dict(global_thermal_state) if global_thermal_state is not None else get_hailo_thermal(),
        "show_in_gui": SHOW_IN_GUI,
        "timestamp": get_current_timestamp()
    }
@app.on_event("shutdown")
def shutdown_event():
    logger.warning("Shutdown event triggered - Terminating FFmpeg processes and threads")
    stop_event.set()
    for camera_id, p in ffmpeg_processes.items():
        logger.info(f"Terminating FFmpeg for camera {camera_id} (PID: {p.pid})")
        p.terminate()
        p.wait()
    for t in feeders.values():
        t.join()
    for t in stderr_loggers.values():
        t.join()
    logger.info("HLS streams shut down successfully")
def run_api(camera_queues: list[Queue], thermal_state=None):
    global global_camera_queues, global_thermal_state
    global_camera_queues = camera_queues
    global_thermal_state = thermal_state
    logger.info(f"Starting API with {len(camera_queues)} camera queues on port {REST_API_PORT}")
    for camera_id in range(len(camera_queues)):
        start_hls(camera_id, camera_queues[camera_id])
    uvicorn.run(app, host="0.0.0.0", port=REST_API_PORT)
if __name__ == "__main__":
    run_api([Queue()])