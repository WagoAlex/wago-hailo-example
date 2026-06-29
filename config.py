import os
SHOW_IN_GUI = int(os.environ.get("SHOW_IN_GUI", "0"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 640))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 640))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.42))
MQTT_BROKER = os.environ.get("MQTT_BROKER", "192.168.2.181")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "inference/yolov5m-results")
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")
PUBLISH_INTERVAL  = int(os.environ.get("PUBLISH_INTERVAL", 1))
HEF_PATH = os.environ.get("HEF_PATH", "yolov5m-helmet.hef")
WEBCAM_INDEX = int(os.environ.get("WEBCAM_INDEX", 0))
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://admin:Master1!@192.168.2.189:554/h264Preview_01_main") # Added: Fallback single URL
RTSP_URLS = os.environ.get("RTSP_URLS",
                           "rtsp://admin:Master1!@192.168.2.189:554/h264Preview_01_main," # Reolink default
                           "rtsp://192.168.2.190:554/live/0," # SICK template 1 (replace <ip> in env)
                           "rtsp://192.168.2.191:554/live/1") # SICK template 2
REST_API_PORT = int(os.environ.get("REST_API_PORT", 8042))
INCLUDE_METADATA = int(os.environ.get("INCLUDE_METADATA", "0")) # Default false ; set to 1 to inlude metadata from MQTT payload
MAX_CAPTURE_OPEN_RETRIES = int(os.environ.get("MAX_CAPTURE_OPEN_RETRIES", 2))
CAPTURE_OPEN_RETRY_DELAY = float(os.environ.get("CAPTURE_OPEN_RETRY_DELAY", 3.0))
MAX_READ_RETRIES = int(os.environ.get("MAX_READ_RETRIES", 5))
PLACEHOLDER_FRAME_DELAY = float(os.environ.get("PLACEHOLDER_FRAME_DELAY", 0.5))
RTSP_RECONNECT_DELAY = float(os.environ.get("RTSP_RECONNECT_DELAY", 5.0)) # New: Env for auto-reconnect backoff
RTSP_TRANSPORT = os.environ.get("RTSP_TRANSPORT", "tcp") # Env toggle for tcp/udp (default tcp for stability)
QUEUE_WARN_THRESHOLD = int(os.environ.get("QUEUE_WARN_THRESHOLD", 20))
QUEUE_DROP_THRESHOLD = int(os.environ.get("QUEUE_DROP_THRESHOLD", 1800))
QUEUE_MONITOR_INTERVAL = float(os.environ.get("QUEUE_MONITOR_INTERVAL", 30.0))
WARN_TOPIC = os.environ.get("WARN_TOPIC", "inference/warnings")
USE_GSTREAMER = int(os.environ.get("USE_GSTREAMER", 1)) # Toggle GStreamer (1=on, 0=FFmpeg only)
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.45)) # For post-NMS filter if needed
NMS_IOU_THRESHOLD = float(os.environ.get("NMS_IOU_THRESHOLD", 0.4)) # For cv2.dnn.NMSBoxes
FRAME_MAX = float(os.environ.get("FRAME_MAX", 0.5)) # For capping max detection frame size 
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()


