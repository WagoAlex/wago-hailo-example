# wago-hailo-example

Real-time helmet detection on a WAGO device using a Hailo-8 AI accelerator, YOLOv5m, and a multi-process Python stack. Inference results are streamed over MQTT and viewable as HLS video via a REST API.

> **Execution note:** This repo cannot be run on the development host. Deploy and test on `192.168.2.124` where the Hailo runtime and connected cameras are available.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│  Spawns one inference process per camera,           │
│  plus MQTT and API processes. Monitors queues       │
│  and restarts crashed processes with backoff.       │
└──────┬───────────────┬──────────────────┬───────────┘
       │               │                  │
       ▼               ▼                  ▼
┌────────────┐  ┌────────────┐  ┌──────────────────────┐
│inference.py│  │ mqtt_app.py│  │   api_app.py          │
│            │  │            │  │                       │
│ Hailo-8    │  │ Subscribes │  │ FastAPI server        │
│ pipeline   │  │ to MQTT    │  │ /health               │
│ YOLOv5m    │  │ inference  │  │ /metadata             │
│ decode     │  │ topic      │  │ /video/stream  (HLS)  │
│ IoU filter │  │            │  │ /video/segment (HLS)  │
│ MQTT pub   │  │            │  │ /stream/mjpeg  (live) │
└──────┬─────┘  └────────────┘  └──────────────────────┘
       │  annotated frames via per-camera Queue (maxsize=2)
       ├──────────────────────────────▶ HLS segments (FFmpeg)
       └──────────────────────────────▶ MJPEG stream (direct)
```

**Data flow per frame:**

1. Camera frame captured (webcam or RTSP)
2. Letterbox resized to 640x640, fed to Hailo-8 via `hailort`
3. Raw output decoded: sigmoid, anchor grid, bbox decode
4. IoU post-filter removes overlapping boxes
5. Result published to MQTT (`inference/yolov5m-results`)
6. Annotated frame (bounding boxes drawn) pushed to per-camera `Queue` (maxsize=2)
7. **MJPEG:** client reads directly from queue via `/stream/mjpeg/{camera_id}` - ~1-3s latency
8. **HLS:** feeder thread pipes frames to FFmpeg, client polls `/video/stream/{camera_id}` for `.m3u8` - ~10-15s latency

---

## Model

- **Model:** YOLOv5m, trained on helmet/no-helmet detection
- **Format:** Hailo HEF (`.hef`) - compiled for Hailo-8
- **Classes:** defined in `yolov5m-helmet.txt`
- **Input:** 640x640 RGB, letterboxed with gray padding (value 114)
- **Output:** stride-based anchor grids (8/16/32), dynamically configured from HEF metadata

---

## Prerequisites

- Hailo-8 AI accelerator connected and recognized by `hailortcli`
- Docker with access to `hailo_docker_hailort_ub2204:4.20.0` base image
- RTSP camera(s) or USB webcam
- MQTT broker reachable at the configured address

---

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `HEF_PATH` | `yolov5m-helmet.hef` | Path to compiled Hailo model |
| `RTSP_URL` | _(Reolink default)_ | Single RTSP stream URL |
| `RTSP_URLS` | _(3 camera defaults)_ | Comma-separated multi-camera URLs |
| `WEBCAM_INDEX` | `0` | USB webcam index |
| `MQTT_BROKER` | `192.168.2.181` | MQTT broker host |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `inference/yolov5m-results` | Topic for inference results |
| `MQTT_USER` / `MQTT_PASS` | _(empty)_ | MQTT credentials |
| `REST_API_PORT` | `8042` | FastAPI listen port |
| `CONFIDENCE_THRESHOLD` | `0.42` | Minimum detection confidence |
| `IOU_THRESHOLD` | `0.45` | Post-NMS IoU filter threshold |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640` | Inference input size |
| `HLS_TIME` | `0.1` | HLS segment duration (seconds) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `INCLUDE_METADATA` | `0` | Include Hailo device metadata in MQTT payload |

---

## Running

### Webcam mode

```bash
docker build -t wago-hailo-example .
docker run --rm \
  --device /dev/hailo0 \
  --device /dev/video0 \
  -e MQTT_BROKER=192.168.2.181 \
  -p 8042:8042 \
  wago-hailo-example webcam
```

### RTSP mode (single camera)

```bash
docker run --rm \
  --device /dev/hailo0 \
  -e RTSP_URL=rtsp://user:pass@192.168.2.189:554/stream \
  -e MQTT_BROKER=192.168.2.181 \
  -p 8042:8042 \
  wago-hailo-example rtsp
```

### RTSP mode (multiple cameras)

```bash
docker run --rm \
  --device /dev/hailo0 \
  -e RTSP_URLS="rtsp://192.168.2.189:554/stream1,rtsp://192.168.2.190:554/stream2" \
  -e MQTT_BROKER=192.168.2.181 \
  -p 8042:8042 \
  wago-hailo-example rtsp
```

> **Test host:** `192.168.2.124` - Hailo-8 and cameras are wired here.

---

## API

Base URL: `http://192.168.2.124:8042`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check + Hailo device metadata |
| `/metadata` | GET | Full Hailo firmware metadata |
| `/stream/mjpeg/{camera_id}` | GET | MJPEG live stream (~1-3s latency) |
| `/video/stream/{camera_id}` | GET | HLS playlist (`.m3u8`) for camera N |
| `/video/segment/{camera_id}/{segment}` | GET | Individual HLS `.ts` segment |
| `/inference` | GET | Inference status (data delivered via MQTT) |

### Stream modes

**MJPEG** (recommended for live monitoring):
```bash
# Browser or VLC - annotated frames, ~1-3s latency
curl http://192.168.2.124:8042/stream/mjpeg/0 --output stream.mjpeg
vlc http://192.168.2.124:8042/stream/mjpeg/0
```

**HLS** (buffered, compatible with all players):
```bash
# Play stream in VLC or any HLS-capable player
curl http://192.168.2.124:8042/health
vlc http://192.168.2.124:8042/video/stream/0
```

Both modes serve frames with bounding boxes already drawn by the inference pipeline. The MJPEG endpoint drains any queued backlog on connect so the first frame shown is always live.

---

## MQTT payload

Published to `inference/yolov5m-results` after each processed frame:

```json
{
  "timestamp": 1719600000000,
  "camera_id": 0,
  "detections": [
    {
      "class": "helmet",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## Project structure

```
src/
├── main.py          # Process orchestrator, queue monitor, restart logic
├── inference.py     # Hailo pipeline, YOLOv5 decode, IoU filter, MQTT publish
├── api_app.py       # FastAPI server, HLS streaming via FFmpeg
├── mqtt_app.py      # MQTT subscriber for inference results
├── config.py        # All env-var configuration with defaults
├── requirements.txt # Python dependencies
├── Dockerfile       # Container build (extends hailo_docker_hailort_ub2204)
├── entrypoint.sh    # Container entrypoint, selects webcam/rtsp mode
└── yolov5m-helmet.txt  # Class label list for the detection model
```

---

## Logs

| Process | Log file |
|---|---|
| main / orchestrator | `/var/log/hailo_main.log` |
| API server | `/var/log/hailo_api.log` |
| Hailo runtime | `/var/log/hailo.log` |

```bash
docker exec <container> tail -f /var/log/hailo_main.log
```
