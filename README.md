[![Docker Hub](https://img.shields.io/badge/docker-wagoalex%2Fwago--hailo--example-6EC800)](https://hub.docker.com/r/wagoalex/wago-hailo-example)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-6EC800.svg)](LICENSE)
[![HailoRT](https://img.shields.io/badge/HailoRT-4.20.0-1F2837.svg)](#prerequisites)
[![Model](https://img.shields.io/badge/model-YOLOv5m%20helmet-1F2837.svg)](#model)

# wago-hailo-example

> Real-time helmet detection on a WAGO Edge Controller using a Hailo-8 AI accelerator. Inference results stream over MQTT and as MJPEG or HLS video - ready to connect to [wago-ai-suite](https://github.com/WagoAlex/wago-ai-suite).

---

## Choose your path

| I am a... | I want to... | Start here |
|-----------|-------------|------------|
| **OT / safety engineer** | Run helmet detection on a camera and get MQTT alerts | [Quick Start](#quick-start) |
| **ML engineer** | Understand the model, swap it out, or retrain | [Model](#model) |
| **Frontend / integration developer** | Connect this to a dashboard or the wago-ai-suite UI | [MQTT payload](#mqtt-payload) - [API](#api) |

---

## What it does

A multi-process Python stack that runs a YOLOv5m model on the Hailo-8 neural accelerator and streams results two ways:

| Output | How | Latency |
|--------|-----|---------|
| **MQTT detections** | JSON published to `inference/yolov5m-results` after every frame | < 20ms (forced socket flush) |
| **MJPEG stream** | Annotated frames served directly from the inference queue | ~1-3s |
| **HLS stream** | Annotated frames piped through FFmpeg, served as `.m3u8` | ~10-15s |

The [wago-ai-suite](https://github.com/WagoAlex/wago-ai-suite) Visual Inference view subscribes to the MQTT topic and overlays bounding boxes on the stream in the browser.

---

## Quick Start

### Prerequisites

- Hailo-8 PCIe module installed and recognized (`hailortcli fw-control identify`)
- HailoRT driver `4.20.0` - download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l) (free registration required)
- Docker base image `hailo_docker_hailort_ub2204:4.20.0` - included in the HailoRT Docker package from the same download page
- USB webcam (`/dev/video0`) or RTSP camera
- MQTT broker reachable from the container

### 1. Build

```bash
docker build -t wago-hailo-example .
```

### 2. Run

**Webcam:**

```bash
docker run --rm \
  --device /dev/hailo0 \
  --device /dev/video0 \
  -e MQTT_BROKER=192.168.2.181 \
  -p 8042:8042 \
  wago-hailo-example webcam
```

**Single RTSP camera:**

```bash
docker run --rm \
  --device /dev/hailo0 \
  -e RTSP_URL=rtsp://user:pass@192.168.2.189:554/stream \
  -e MQTT_BROKER=192.168.2.181 \
  -p 8042:8042 \
  wago-hailo-example rtsp
```

**Multiple RTSP cameras:**

```bash
docker run --rm \
  --device /dev/hailo0 \
  -e RTSP_URLS="rtsp://192.168.2.189:554/stream1,rtsp://192.168.2.190:554/stream2" \
  -e MQTT_BROKER=192.168.2.181 \
  -p 8042:8042 \
  wago-hailo-example rtsp
```

### 3. Verify

```bash
curl http://localhost:8042/health
```

Once running, open `http://localhost:8042/stream/mjpeg/0` in a browser to see the live annotated stream.

> [!NOTE]
> This container requires a physical Hailo-8 device. It cannot run on a development host without the accelerator and HailoRT driver.

---

## Architecture

```
main.py  (process orchestrator)
  │  spawns one inference process per camera
  │  plus MQTT subscriber and API server
  │  monitors queues, restarts crashed processes with backoff
  │
  ├── inference.py  (per camera)
  │     Hailo-8 pipeline via hailort
  │     YOLOv5m decode: sigmoid, anchor grid, bbox
  │     IoU post-filter, NMS
  │     Publishes JSON to MQTT every frame
  │     Pushes annotated frames to per-camera Queue (maxsize=2)
  │
  ├── api_app.py  (FastAPI, port 8042)
  │     /health, /metadata
  │     /stream/mjpeg/{camera_id}  - drains queue, streams JPEG
  │     /video/stream/{camera_id}  - HLS playlist via FFmpeg
  │     /video/segment/{camera_id}/{segment}
  │
  └── mqtt_app.py  (subscriber, for logging/debugging)
        Subscribes to inference/yolov5m-results
        Logs received payloads - not connected to the UI

MQTT broker (external, e.g. Mosquitto)
  topic: inference/yolov5m-results
  consumed by: wago-ai-suite Visual-Inference.js
```

---

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLOv5m |
| Task | Helmet / no-helmet detection |
| Format | Hailo HEF (`.hef`) - compiled for Hailo-8 |
| Input | 640x640 RGB, letterboxed with gray padding (value 114) |
| Output strides | 8 / 16 / 32 - anchor grids configured dynamically from HEF metadata |
| Classes | Defined in `yolov5m-helmet.txt` |
| Confidence gate | `CONFIDENCE_THRESHOLD=0.70` (cuts noise at 65%, valid detections at 88-93%) |

To use a different `.hef` model:

```bash
-e HEF_PATH=/models/my-model.hef \
-v /host/models:/models
```

The decode pipeline (`build_output_configs_from_hef`) reads stride and anchor config directly from the HEF metadata, so no code changes are needed when swapping models with the same output shape.

---

## MQTT payload

Published to `inference/yolov5m-results` after every processed frame:

```json
{
  "timestamp": 1719600000000,
  "fps": 28.4,
  "detections": [
    {
      "class": "helmet",
      "confidence": 0.91,
      "box": [120.0, 45.0, 310.0, 280.0]
    }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `timestamp` | int | Milliseconds since epoch |
| `fps` | float | Inference frame rate |
| `detections[].class` | string | Label from `yolov5m-helmet.txt` |
| `detections[].confidence` | float | 0.0-1.0, already filtered by `CONFIDENCE_THRESHOLD` |
| `detections[].box` | float[4] | `[x1, y1, x2, y2]` in 640x640 pixel space |

> [!IMPORTANT]
> The field name is `box`, not `bbox`. The wago-ai-suite frontend validates this field name explicitly - a mismatch silently drops all detections.

Empty `detections: []` is published every frame when nothing is detected, so the subscriber always receives a heartbeat.

---

## API

Base URL: `http://<host>:8042`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check + Hailo firmware identity |
| `/metadata` | GET | Full Hailo device metadata from `hailortcli` |
| `/stream/mjpeg/{camera_id}` | GET | MJPEG live stream, annotated frames, ~1-3s latency |
| `/video/stream/{camera_id}` | GET | HLS playlist (`.m3u8`) |
| `/video/segment/{camera_id}/{segment}` | GET | Individual HLS `.ts` segment |
| `/inference` | GET | Inference status (detections come via MQTT, not here) |

**MJPEG** - recommended for live monitoring:

```bash
# Browser or VLC
vlc http://localhost:8042/stream/mjpeg/0
```

**HLS** - buffered, compatible with all players:

```bash
vlc http://localhost:8042/video/stream/0
```

Both modes serve frames with bounding boxes already drawn by the inference pipeline. The MJPEG endpoint drains any queued backlog on connect so the first frame is always live.

The wago-ai-suite backend proxies both endpoints - the frontend never calls them directly.

---

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HEF_PATH` | `yolov5m-helmet.hef` | Path to compiled Hailo model file |
| `RTSP_URL` | _(Reolink default)_ | Single RTSP stream URL |
| `RTSP_URLS` | _(3 camera defaults)_ | Comma-separated multi-camera RTSP URLs |
| `RTSP_TRANSPORT` | `tcp` | RTSP transport: `tcp` or `udp` |
| `RTSP_RECONNECT_DELAY` | `5.0` | Seconds before reconnecting a dropped RTSP stream |
| `WEBCAM_INDEX` | `0` | USB webcam device index |
| `MQTT_BROKER` | `192.168.2.181` | MQTT broker hostname or IP |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `inference/yolov5m-results` | Topic where detection JSON is published |
| `MQTT_USER` / `MQTT_PASS` | _(empty)_ | MQTT credentials |
| `REST_API_PORT` | `8042` | FastAPI listen port |
| `CONFIDENCE_THRESHOLD` | `0.70` | Minimum detection score published to MQTT |
| `IOU_THRESHOLD` | `0.15` | Post-NMS IoU filter - suppresses duplicate boxes per object |
| `NMS_IOU_THRESHOLD` | `0.15` | IoU threshold for `cv2.dnn.NMSBoxes` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640` | Inference input dimensions |
| `FRAME_MAX` | `0.5` | Maximum detection box size as fraction of frame |
| `USE_GSTREAMER` | `1` | `1` = GStreamer pipeline, `0` = FFmpeg only |
| `INCLUDE_METADATA` | `0` | `1` = include Hailo device metadata in MQTT payload |
| `SHOW_IN_GUI` | `0` | `1` = show annotated frames in local GUI window (requires display) |
| `QUEUE_WARN_THRESHOLD` | `20` | Log warning when frame queue depth exceeds this |
| `QUEUE_DROP_THRESHOLD` | `1800` | Drop frames when queue depth exceeds this |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

> [!TIP]
> `CONFIDENCE_THRESHOLD=0.70` is calibrated for this model: valid detections score 88-93%, noise peaks at 60-65%. Lower it only if you're using a different model.

---

## Project structure

```
src/
├── main.py             # Process orchestrator, queue monitor, restart backoff
├── inference.py        # Hailo-8 pipeline, YOLOv5m decode, IoU filter, MQTT publish
├── api_app.py          # FastAPI server, MJPEG and HLS streaming
├── mqtt_app.py         # MQTT subscriber (logging/debug - not connected to UI)
├── config.py           # All env-var configuration with defaults
├── requirements.txt    # Python dependencies
├── Dockerfile          # Extends hailo_docker_hailort_ub2204:4.20.0 (from Hailo Developer Zone)
├── entrypoint.sh       # Selects webcam or rtsp mode from CMD arg
└── yolov5m-helmet.txt  # Class label list
```

---

## Logs

| Process | Log file |
|---------|----------|
| Main orchestrator | `/var/log/hailo_main.log` |
| API server | `/var/log/hailo_api.log` |
| Hailo runtime | `/var/log/hailo.log` |

```bash
docker exec <container> tail -f /var/log/hailo_main.log
```

---

## Troubleshooting

**`hailortcli` not found or device not recognized**

```bash
hailortcli fw-control identify
```

If this fails outside Docker, the PCIe driver is not loaded. Install `hailort-pcie-driver_4.20.0_all.deb` and reboot.

**No detections despite objects in frame**

- Check `CONFIDENCE_THRESHOLD` - default `0.70` is tuned for this model. Lower it temporarily to `0.30` to confirm the pipeline is working.
- Confirm the HEF file matches the model architecture. The decode pipeline reads anchor config from HEF metadata - a mismatched file produces garbage output, not an error.

**MQTT messages not arriving in wago-ai-suite**

- Topic must be exactly `inference/yolov5m-results` on both sides.
- Broker must accept WebSocket connections on port `9001` (the frontend connects via WS).
- Confirm with: `mosquitto_sub -h <broker> -t 'inference/#' -v`

**MJPEG stream freezes**

The queue `maxsize=2` drops frames under load to stay live. If the consumer (browser or proxy) is slow, frames back up and are dropped. This is by design - the stream always shows the latest frame, not a replay.

---

## Requirements

- Hailo-8 PCIe module + HailoRT driver `4.20.0`
- Docker base image `hailo_docker_hailort_ub2204:4.20.0` - download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l) (free registration required)
- RTSP camera(s) or USB webcam at `/dev/video0`
- MQTT broker (Mosquitto recommended)

---

## License

[Mozilla Public License 2.0](LICENSE)
