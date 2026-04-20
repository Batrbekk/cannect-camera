"""Application settings loaded from environment variables.

Updated for Mini PC x86 Linux, pull-based architecture, per-campaign attribution.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class CameraSettings(BaseSettings):
    """Per-camera RTSP/USB configuration."""

    model_config = {"env_prefix": ""}

    camera_count: int = Field(default=5, alias="CAMERA_COUNT")
    camera_1_url: str = Field(default="rtsp://admin:pass@192.168.1.101:554/stream1", alias="CAMERA_1_URL")
    camera_2_url: str = Field(default="rtsp://admin:pass@192.168.1.102:554/stream1", alias="CAMERA_2_URL")
    camera_3_url: str = Field(default="rtsp://admin:pass@192.168.1.103:554/stream1", alias="CAMERA_3_URL")
    camera_4_url: str = Field(default="rtsp://admin:pass@192.168.1.104:554/stream1", alias="CAMERA_4_URL")
    camera_5_url: str = Field(default="rtsp://admin:pass@192.168.1.105:554/stream1", alias="CAMERA_5_URL")

    def get_url(self, camera_index: int) -> str:
        return getattr(self, f"camera_{camera_index}_url")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}

    # Identity
    station_id: str = Field(default="69d4e54739079402c7d5608e", alias="STATION_ID")

    # Security
    station_token: str = Field(default="", alias="STATION_TOKEN")
    allowed_backend_ips: str = Field(default="", alias="ALLOWED_BACKEND_IPS")

    # Backend (fallback push mode)
    api_url: str = Field(default="http://192.168.0.106:3000/api/analytics/events", alias="API_URL")
    api_key: str = Field(default="", alias="API_KEY")

    # Cameras
    camera: CameraSettings = Field(default_factory=CameraSettings)

    # Hardware acceleration
    inference_backend: str = Field(default="onnxruntime", alias="INFERENCE_BACKEND")
    use_hw_decode: bool = Field(default=False, alias="USE_HW_DECODE")
    gpu_device: int = Field(default=0, alias="GPU_DEVICE")

    # Server (edge HTTP)
    server_host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(default=8080, alias="SERVER_PORT")

    # Processing
    capture_fps: int = Field(default=30, description="Camera capture FPS")
    processing_fps: int = Field(default=10, description="Downsampled FPS for CV pipeline")

    # Detection thresholds
    min_person_bbox_height: int = Field(default=60, alias="MIN_PERSON_BBOX_HEIGHT")
    min_track_frames: int = Field(default=3, alias="MIN_TRACK_FRAMES")

    # Face analysis
    gender_confidence: float = Field(default=0.55, alias="GENDER_CONFIDENCE")
    age_confidence: float = Field(default=0.60, alias="AGE_CONFIDENCE")
    face_match_threshold: float = Field(default=0.60, alias="FACE_MATCH_THRESHOLD")
    unique_face_ttl_sec: int = Field(default=1800, alias="UNIQUE_FACE_TTL_SEC")

    # Head pose thresholds
    direct_yaw_max: float = 15.0
    direct_pitch_max: float = 10.0
    partial_yaw_max: float = 30.0
    attention_start_frames: int = 3
    attention_stop_frames: int = 5

    # Attention
    attention_dwell_threshold_sec: float = Field(default=5.0, alias="ATTENTION_DWELL_THRESHOLD_SEC")

    # Aggregation / persistence
    counter_reset_on_day_change: bool = Field(default=True, alias="COUNTER_RESET_ON_DAY_CHANGE")
    counter_persistence_path: str = Field(default="data/counters.db", alias="COUNTER_PERSISTENCE_PATH")

    # Video player integration
    player_webhook_enabled: bool = Field(default=True, alias="PLAYER_WEBHOOK_ENABLED")

    # Privacy (hard-coded safety)
    save_frames: bool = Field(default=False, alias="SAVE_FRAMES")
    save_embeddings: bool = Field(default=False, alias="SAVE_EMBEDDINGS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Models directory
    models_dir: str = Field(default="models", alias="MODELS_DIR")


settings = Settings()
