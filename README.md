# CANNECT.AI Computer Vision Analytics

Real-time computer vision analytics system for vending machine advertising displays. Processes camera feeds from Raspberry Pi devices to track viewer engagement, demographics, and traffic patterns.

## Features

- **Traffic Counting**: Detects and counts people and vehicles using YOLOv8
- **Attention Tracking**: Measures viewer dwell time, engagement, and generates heatmaps
- **Demographics Analysis**: Anonymized age/gender estimation (no biometric storage)
- **Multi-Camera Support**: Coordinates 4 cameras across 2 vending machines
- **360° Tracking**: Cross-camera tracking for complete coverage
- **Real-time API Integration**: Sends analytics to CANNECT.AI platform

## System Architecture

```
Raspberry Pi 5 (Vending 1)     Raspberry Pi 5 (Vending 2)
    ├─ Camera 1 (Front)            ├─ Camera 1 (Front)
    └─ Camera 2 (Side)             └─ Camera 2 (Side)
           │                              │
           └──────────RTSP─────────────────┘
                      │
                 GCP VM (GPU)
            ┌─────────┴────────┐
            │  CV Analytics     │
            │  - YOLOv8         │
            │  - Tracking       │
            │  - Demographics   │
            └─────────┬─────────┘
                      │
             CANNECT.AI API
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Batrbekk/cannect-camera.git
cd cannect-camera
```

### 2. Install Dependencies

```bash
# On VM (with Python 3.11+)
python3.11 -m pip install -r requirements.txt
```

### 3. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano config/config.yaml
```

Update camera stream URLs in `config/config.yaml`:

```yaml
cameras:
  vending_1:
    camera_1:
      stream_url: 'rtsp://192.168.1.101:8554/cam1'  # Your Raspberry Pi IP
```

### 4. Set API Token

```bash
# Add your CANNECT.AI API token to .env
echo "CANNECT_API_TOKEN=your_token_here" >> .env
```

## Configuration

### Camera Setup

Each vending machine has 2 cameras connected to a Raspberry Pi 5. Configure RTSP streams in `config/config.yaml`.

**On Raspberry Pi**, stream camera using:

```bash
# Install
sudo apt install v4l-utils

# Stream camera
ffmpeg -f v4l2 -i /dev/video0 -vcodec h264 -f rtsp rtsp://0.0.0.0:8554/cam1
```

Or use `mediamtx` for better performance:

```bash
# Download mediamtx
wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.0/mediamtx_v1.5.0_linux_arm64v8.tar.gz
tar -xzf mediamtx_*.tar.gz
./mediamtx
```

### Analytics Configuration

Enable/disable analytics modules in `config/config.yaml`:

```yaml
analytics:
  traffic_counting:
    enabled: true
  attention_tracking:
    enabled: true
    dwell_time_threshold: 2.0  # seconds
  demographics:
    enabled: true
    anonymize_data: true  # Always true for privacy
```

### Model Configuration

Choose YOLO model based on hardware:

```yaml
model:
  name: 'yolov8n.pt'  # Nano - fastest (CPU)
  # name: 'yolov8s.pt'  # Small (GPU recommended)
  device: 'cpu'  # or 'cuda' with GPU
```

## Usage

### Run Analytics

```bash
cd src
python3.11 main.py
```

### Test Camera Connections

```bash
python3.11 tests/test_cameras.py
```

### Monitor Logs

```bash
tail -f logs/cv_analytics.log
```

## Data Privacy

**IMPORTANT**: This system is designed with privacy in mind:

- ✅ **NO facial recognition**
- ✅ **NO biometric data storage**
- ✅ **NO personal identification**
- ✅ **Anonymized demographics only**
- ✅ **Aggregated statistics**

All demographics analysis is done in real-time with immediate anonymization. Only statistical aggregates (e.g., "35% male, 25-35 age group") are stored.

## API Integration

Analytics data is sent to CANNECT.AI API every 10 seconds:

```json
{
  "timestamp": "2025-12-19T11:00:00Z",
  "cameras": {
    "vending_001_cam_1": {
      "traffic": {
        "people_count": 5,
        "vehicle_count": 2
      },
      "attention": {
        "engaged_viewers": 3,
        "avg_dwell_time": 4.5
      },
      "demographics": {
        "age_distribution": {"19-30": 2, "31-45": 1},
        "gender_distribution": {"M": 2, "F": 1}
      }
    }
  }
}
```

## Performance

### CPU Mode (no GPU)
- n1-standard-8: ~15-20 FPS per camera
- 4 cameras: ~3-5 FPS per camera (frame skipping recommended)

### GPU Mode (NVIDIA T4)
- 60-100 FPS per camera
- 4 cameras: 15+ FPS per camera

Adjust `frame_skip` in config to optimize:

```yaml
processing:
  frame_skip: 2  # Process every 2nd frame
```

## Deployment

### On Google Cloud VM

```bash
# SSH to VM
gcloud compute ssh cannect-cv-vm --zone=europe-west4-a

# Clone and setup
git clone https://github.com/Batrbekk/cannect-camera.git
cd cannect-camera
python3.11 -m pip install -r requirements.txt

# Configure
nano config/config.yaml
nano .env

# Run
cd src
python3.11 main.py
```

### Run as Service

```bash
# Create systemd service
sudo nano /etc/systemd/system/cannect-cv.service
```

```ini
[Unit]
Description=CANNECT.AI CV Analytics
After=network.target

[Service]
Type=simple
User=batyrbekkuandyk
WorkingDirectory=/home/batyrbekkuandyk/cannect-camera/src
ExecStart=/usr/bin/python3.11 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable cannect-cv
sudo systemctl start cannect-cv

# Check status
sudo systemctl status cannect-cv
```

## Troubleshooting

### No camera connection

```bash
# Test RTSP stream
ffplay rtsp://192.168.1.101:8554/cam1

# Check network
ping 192.168.1.101
```

### Low FPS

- Increase `frame_skip` in config
- Use smaller YOLO model (`yolov8n.pt`)
- Request GPU quota from Google Cloud

### API errors

- Check `CANNECT_API_TOKEN` in `.env`
- Verify API endpoint in `config/config.yaml`
- Check logs: `logs/cv_analytics.log`

## Development

### Project Structure

```
cannect-camera/
├── config/
│   └── config.yaml          # Configuration
├── src/
│   ├── analytics/
│   │   ├── traffic_counter.py      # YOLO detection
│   │   ├── attention_tracker.py    # Engagement tracking
│   │   └── demographics.py         # Age/gender (anonymized)
│   ├── utils/
│   │   ├── camera_handler.py       # RTSP stream handling
│   │   └── api_client.py           # API integration
│   └── main.py              # Main entry point
├── logs/                    # Log files
├── data/                    # Temporary data
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables
```

## License

Proprietary - CANNECT.AI

## Support

For issues, contact: batrbekk@gmail.com
