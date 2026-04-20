# CANNECT.AI — CV Analytics Edge Module

On-premise computer vision analytics for CANNECT.AI vending stations. Processes video from **5 cameras** in real-time, extracts audience metrics (people count, attention, demographics), and serves them via HTTP API for backend consumption.

## Architecture

```
┌─────────────── Mini PC inside vending station ───────────────┐
│                                                               │
│  [5 Cameras RTSP/USB]                                        │
│       ↓                                                       │
│  [YOLOv8n Person Detection] → [ByteTrack] → Track ID        │
│       ↓                                                       │
│  [SCRFD Face Detection] → [Face Alignment 112x112]           │
│       ↓                                                       │
│  [MediaPipe Head Pose] → yaw/pitch/roll → Attention Timer    │
│  [InsightFace GenderAge] → male/female + age group           │
│  [MobileFaceNet Embedding] → Cross-camera unique viewer      │
│       ↓                                                       │
│  [CounterStore] ← per-campaign attribution via AdTracker     │
│       ↓                                                       │
│  [FastAPI :8080]          [Push to backend every 60s]        │
│   GET /metrics             POST /api/analytics/events        │
│   GET /metrics/by-campaign                                    │
│   GET /health                                                 │
│   POST /current-ad                                            │
└───────────────────────────────────────────────────────────────┘
```

**Pull-based**: Backend polls `GET /metrics` when needed.  
**Push fallback**: Also pushes to backend every 60 seconds.

## What It Measures

| Metric | Description |
|---|---|
| **People Count** | Total people detected across all cameras (deduplicated via face embeddings) |
| **Direction** | Movement: `toScreen`, `fromScreen`, `left`, `right` |
| **Attention** | Dwell time — how long each person looks at the screen |
| **Attention >5s** | Key KPI: people who looked >5 seconds |
| **Gender** | male / female / unknown (InsightFace, confidence threshold) |
| **Age Group** | child (0-12), teen (13-17), young (18-30), adult (31-50), senior (51+) |
| **Unique Viewers** | Cross-camera deduplication via face embeddings (RAM only, TTL 30 min) |
| **Per-Campaign** | All metrics segregated by currently playing ad campaign |

## Quick Start

### 1. Install

```bash
# Requires Python 3.11+
pip install uv
uv venv --python 3.11 .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

uv pip install opencv-python numpy pydantic pydantic-settings \
    onnxruntime mediapipe httpx fastapi uvicorn scipy filterpy lap psutil
```

### 2. Download Models

```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640)
import shutil; shutil.move('yolov8n.onnx', 'models/yolov8n.onnx')
"

# InsightFace models (SCRFD + GenderAge + FaceNet)
python -c "
import urllib.request, zipfile, shutil, os
for pack in ['buffalo_sc', 'buffalo_l']:
    url = f'https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip'
    urllib.request.urlretrieve(url, f'{pack}.zip')
    with zipfile.ZipFile(f'{pack}.zip') as z: z.extractall('models')
    sub = f'models/{pack}'
    if os.path.isdir(sub):
        for f in os.listdir(sub): shutil.move(f'{sub}/{f}', f'models/{f}')
        os.rmdir(sub)
    os.remove(f'{pack}.zip')
shutil.copy2('models/det_500m.onnx', 'models/scrfd_500m.onnx')
shutil.copy2('models/w600k_mbf.onnx', 'models/mobilefacenet.onnx')
"

# MediaPipe face landmarker downloads automatically on first run
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your station ID, camera URLs, backend address
```

### 4. Run

```bash
# Multi-camera with preview window (development/testing)
python -m src.run_multicam

# Production mode (no GUI, systemd service)
python -m src.main
```

## API Endpoints (Edge Server :8080)

### `GET /metrics`
Global aggregated metrics. Optional `?reset=true` to clear after read.

```json
{
  "stationId": "69d4e54739079402c7d5608e",
  "timestamp": "2026-04-20T14:30:00Z",
  "currentAd": { "campaignId": "...", "videoId": "..." },
  "global": {
    "traffic": { "people_total": 47, "toScreen": 18, "fromScreen": 12 },
    "attention": { "total_looking": 23, "attention_over_5s": 11, "avg_dwell_time": 6.8, "unique_viewers": 19 },
    "demographics": {
      "gender": { "male": 12, "female": 9, "unknown": 2 },
      "age_groups": { "child": 1, "teen": 2, "young": 8, "adult": 9, "senior": 3 }
    },
    "by_camera": { "camera_1": { "people": 15, "looking": 8 } }
  }
}
```

### `GET /metrics/by-campaign`
Per-campaign breakdown for B2B analytics.

### `GET /health`
System health: CPU, RAM, disk, camera status.

### `POST /current-ad`
Video player notifies which ad is playing:
```json
{ "event": "playback_started", "campaignId": "...", "videoId": "...", "startedAt": "...", "expectedDuration": 15 }
```

## Push to Backend

Every 60 seconds, pushes metrics to the CANNECT backend:

```
POST http://192.168.0.106:3000/api/analytics/events
```

Payload format matches the backend API spec with `stationId`, `cameraId`, `campaignId`, and event types: `traffic`, `attention`, `demographic`.

## Test with Backend

```bash
# Send synthetic test data to local backend
python -m src.test_send --backend http://192.168.0.106:3000 --interval 60

# Check dashboard
open http://192.168.0.106:3000/dashboard/analytics
```

## Project Structure

```
├── src/
│   ├── capture/           # Camera capture (RTSP/USB, thread-per-camera)
│   │   ├── grabber.py
│   │   └── camera_manager.py
│   ├── detection/         # ML detection models
│   │   ├── person_detector.py   # YOLOv8n ONNX
│   │   └── face_detector.py     # SCRFD-500M ONNX
│   ├── tracking/
│   │   └── bytetrack.py         # ByteTrack + Kalman filter
│   ├── analysis/          # Face analysis pipeline
│   │   ├── head_pose.py         # MediaPipe Face Landmarker + solvePnP
│   │   ├── attention.py         # Dwell time tracker (>5s KPI)
│   │   ├── gender_age.py        # InsightFace genderage
│   │   ├── face_embedding.py    # MobileFaceNet for unique viewers
│   │   └── emotion.py           # HSEmotion (optional)
│   ├── aggregation/       # Metrics storage
│   │   ├── counters.py          # Per-campaign in-memory counters
│   │   ├── persistence.py       # SQLite write-through cache
│   │   └── ad_tracker.py        # Current ad campaign tracking
│   ├── server/            # Edge HTTP API
│   │   ├── api.py               # FastAPI endpoints
│   │   └── auth.py              # X-Station-Token auth
│   ├── config/
│   │   └── settings.py          # Environment-based configuration
│   ├── main.py            # Production pipeline (headless)
│   ├── run_multicam.py    # Multi-camera with preview window
│   ├── demo_viewer.py     # Single-camera demo
│   └── test_send.py       # Synthetic data sender for testing
├── models/                # ONNX models (not in git, download separately)
├── data/                  # SQLite persistence
├── systemd/               # Linux service files
├── pyproject.toml
└── .env.example
```

## Privacy

- Face images are **never saved** to disk or sent to backend
- Face embeddings exist **only in RAM** with 30-minute TTL
- Only **aggregated numbers** are transmitted (counts, averages)
- No personal identification — no matching against external databases
- GDPR Art. 25 (Privacy by Design) compliant

## Hardware Requirements

**Minimum** (for testing):
- Any x86 PC with USB cameras
- 8GB RAM, 4+ CPU cores

**Production** (5 cameras, full pipeline):
- Mini PC: AMD Ryzen 7 / Intel i7 (8+ cores)
- RAM: 32GB DDR5
- SSD: 256GB+ NVMe
- USB 3.0 hub for 5 cameras
- Optional: NVIDIA GPU for TensorRT acceleration

## Configuration

See `.env.example` for all available settings:
- `STATION_ID` — MongoDB ObjectId of the vending station
- `CAMERA_COUNT` / `CAMERA_N_URL` — Camera RTSP/USB sources
- `API_URL` — Backend endpoint for push mode
- `STATION_TOKEN` — Shared secret for pull API auth
- `ATTENTION_DWELL_THRESHOLD_SEC` — Attention threshold (default: 5.0s)
- `GENDER_CONFIDENCE` — Min confidence for gender classification
- `FACE_MATCH_THRESHOLD` — Cosine similarity for face dedup (default: 0.6)
