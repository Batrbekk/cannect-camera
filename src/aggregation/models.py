"""Pydantic models for analytics — pull-based edge API + push fallback.

Per-campaign attribution: all metrics are segregated by campaignId.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# --- Enums ---

class Direction(str, Enum):
    TO_SCREEN = "toScreen"
    FROM_SCREEN = "fromScreen"
    LEFT = "left"
    RIGHT = "right"


class GazeStatus(str, Enum):
    DIRECT = "direct"
    PARTIAL = "partial"
    GLANCE = "glance"


class AgeGroup(str, Enum):
    CHILD = "child"
    TEEN = "teen"
    YOUNG = "young"
    ADULT = "adult"
    SENIOR = "senior"

    @classmethod
    def from_age(cls, age: int) -> "AgeGroup":
        if age <= 12:
            return cls.CHILD
        elif age <= 17:
            return cls.TEEN
        elif age <= 30:
            return cls.YOUNG
        elif age <= 50:
            return cls.ADULT
        else:
            return cls.SENIOR


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class EmotionType(str, Enum):
    JOY = "joy"
    INTEREST = "interest"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# --- Edge pull-API response models ---

class TrafficMetrics(BaseModel):
    people_total: int = 0
    toScreen: int = 0
    fromScreen: int = 0
    left: int = 0
    right: int = 0


class AttentionMetrics(BaseModel):
    total_looking: int = 0
    attention_over_5s: int = 0
    avg_dwell_time: float = 0.0
    unique_viewers: int = 0


class DemographicMetrics(BaseModel):
    analyzed_count: int = 0
    gender: dict[str, int] = Field(default_factory=lambda: {"male": 0, "female": 0, "unknown": 0})
    age_groups: dict[str, int] = Field(default_factory=lambda: {
        "child": 0, "teen": 0, "young": 0, "adult": 0, "senior": 0
    })


class CameraMetrics(BaseModel):
    people: int = 0
    looking: int = 0
    attention_over_5s: int = 0


class CurrentAdInfo(BaseModel):
    campaignId: str | None = None
    videoId: str | None = None
    playbackStartedAt: str | None = None


class GlobalMetrics(BaseModel):
    traffic: TrafficMetrics = Field(default_factory=TrafficMetrics)
    attention: AttentionMetrics = Field(default_factory=AttentionMetrics)
    demographics: DemographicMetrics = Field(default_factory=DemographicMetrics)
    by_camera: dict[str, CameraMetrics] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Response for GET /metrics."""
    stationId: str
    timestamp: str
    sinceTimestamp: str | None = None
    currentAd: CurrentAdInfo = Field(default_factory=CurrentAdInfo)
    global_metrics: GlobalMetrics = Field(default_factory=GlobalMetrics, alias="global")

    model_config = {"populate_by_name": True}


class CampaignMetrics(BaseModel):
    impressions: int = 0
    deep_views_over_5s: int = 0
    total_looking: int = 0
    avg_dwell_time: float = 0.0
    unique_viewers: int = 0
    gender: dict[str, int] = Field(default_factory=lambda: {"male": 0, "female": 0, "unknown": 0})
    age_groups: dict[str, int] = Field(default_factory=lambda: {
        "child": 0, "teen": 0, "young": 0, "adult": 0, "senior": 0
    })


class CampaignEntry(BaseModel):
    campaignId: str | None = None
    videoId: str | None = None
    totalPlaybacks: int = 0
    totalPlaybackSeconds: float = 0.0
    metrics: CampaignMetrics = Field(default_factory=CampaignMetrics)


class CampaignMetricsResponse(BaseModel):
    """Response for GET /metrics/by-campaign."""
    stationId: str
    timestamp: str
    campaigns: list[CampaignEntry] = Field(default_factory=list)


class CameraHealthInfo(BaseModel):
    status: str = "offline"
    fps: float = 0.0
    last_frame_at: str | None = None
    last_error: str | None = None


class HealthResponse(BaseModel):
    """Response for GET /health."""
    stationId: str
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    ram_usage: int = 0
    disk_free_mb: int = 0
    cameras: dict[str, CameraHealthInfo] = Field(default_factory=dict)


class CurrentAdPayload(BaseModel):
    """Payload for POST /current-ad from local video player."""
    event: str  # playback_started / playback_ended / playback_changed
    campaignId: str | None = None
    videoId: str | None = None
    startedAt: str | None = None
    expectedDuration: int | None = None


# --- Push fallback (POST /api/analytics/events) ---

class DirectionCount(BaseModel):
    toScreen: int = 0
    fromScreen: int = 0
    left: int = 0
    right: int = 0


class TimeIntervalCount(BaseModel):
    hour: int = 0
    day: int = 0
    week: int = 0


class PeopleData(BaseModel):
    total: int = 0
    direction: DirectionCount = Field(default_factory=DirectionCount)
    byTimeInterval: TimeIntervalCount = Field(default_factory=TimeIntervalCount)


class TrafficEventData(BaseModel):
    people: PeopleData = Field(default_factory=PeopleData)


class HeadAngleBreakdown(BaseModel):
    direct: int = 0
    partial: int = 0
    glance: int = 0


class PeopleAttentionData(BaseModel):
    totalLooking: int = 0
    attentionOver5s: int = 0
    averageDwellTime: float = 0.0
    attentionScore: int = 0
    headAngle: HeadAngleBreakdown = Field(default_factory=HeadAngleBreakdown)
    uniqueViewers: int = 0


class AttentionEventData(BaseModel):
    peopleAttention: PeopleAttentionData = Field(default_factory=PeopleAttentionData)


class GenderCount(BaseModel):
    male: int = 0
    female: int = 0
    unknown: int = 0


class AgeGroupCount(BaseModel):
    child: int = 0
    teen: int = 0
    young: int = 0
    adult: int = 0
    senior: int = 0


class GroupCount(BaseModel):
    individuals: int = 0
    couples: int = 0
    families: int = 0
    largeGroups: int = 0


class DemographicEventData(BaseModel):
    gender: GenderCount = Field(default_factory=GenderCount)
    ageGroups: AgeGroupCount = Field(default_factory=AgeGroupCount)
    groups: GroupCount = Field(default_factory=GroupCount)
    withChildren: int = 0


class DwellTimeStats(BaseModel):
    average: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0


class EngagementEventData(BaseModel):
    dwellTime: DwellTimeStats = Field(default_factory=DwellTimeStats)
    viewingRate: float = 0.0
    repeatViewers: int = 0


class AnalyticsEvent(BaseModel):
    type: str
    data: TrafficEventData | AttentionEventData | DemographicEventData | EngagementEventData


class AnalyticsPayload(BaseModel):
    stationId: str
    cameraId: str
    campaignId: str | None = None
    timestamp: datetime | None = None
    events: list[AnalyticsEvent]


# --- Internal tracking models ---

class TrackedPerson(BaseModel):
    track_id: int
    camera_id: str
    bbox: tuple[int, int, int, int]
    direction: Direction | None = None
    frame_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    face_embedding: list[float] | None = None
    gender: Gender | None = None
    gender_confidence: float = 0.0
    age: int | None = None
    age_confidence: float = 0.0
    age_group: AgeGroup | None = None
    gaze_status: GazeStatus | None = None
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    dwell_time: float = 0.0
    is_looking: bool = False
    looking_start_time: float | None = None
    emotion: EmotionType | None = None
    emotion_confidence: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class DetectionResult(BaseModel):
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0

    model_config = {"arbitrary_types_allowed": True}


class FaceDetectionResult(BaseModel):
    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: list[tuple[float, float]] | None = None

    model_config = {"arbitrary_types_allowed": True}
