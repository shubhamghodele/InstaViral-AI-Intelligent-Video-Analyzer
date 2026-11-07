"""
Pydantic schemas for all FastAPI request / response models.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request Models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    video_id:   str
    niche:      str = Field(..., description="Dance | Comedy | Tutorial | Motivation | Other")
    viral_urls: list[str] = Field(
        default=[],
        description="Instagram reel URLs user wants to compare against"
    )


class BenchmarkSeedRequest(BaseModel):
    category:  str
    video_url: str


# ── Feature Vector ────────────────────────────────────────────────

class FeatureVector(BaseModel):
    # Audio features
    transcript_sentiment: float = 0.0   # -1 to 1
    hook_strength:        float = 0.0   # 0-10
    cta_present:          float = 0.0   # 0 or 1

    # Visual features (from Groq Vision JSON)
    visual_energy:        float = 0.0   # 0-10
    lighting_quality:     float = 0.0   # 0-10
    composition:          float = 0.0   # 0-10
    color_vibrancy:       float = 0.0   # 0-10
    text_overlay:         float = 0.0   # 0 or 1

    # Emotion features (EmotiEffLib — 0 if no face)
    avg_positive_emotion: float = 0.0   # 0-1
    emotion_variance:     float = 0.0   # 0-1
    smile_ratio:          float = 0.0   # 0-1

    # Structure features
    cuts_per_minute:      float = 0.0   # raw count
    motion_intensity:     float = 0.0   # 0-10
    music_present:        float = 0.0   # 0 or 1

    def to_list(self) -> list[float]:
        return list(self.model_dump().values())


# ── Emotion entry ─────────────────────────────────────────────────

class EmotionEntry(BaseModel):
    time_sec: float
    emotion:  str
    score:    float


# ── Scene entry ───────────────────────────────────────────────────

class SceneEntry(BaseModel):
    frame_number: int
    timestamp:    float
    description:  str
    visual_score: float


# ── Analysis Result ───────────────────────────────────────────────

class AnalysisResult(BaseModel):
    video_id:         str
    virality_score:   float                  # 0-100
    feature_vector:   list[float]
    transcript:       str
    emotion_timeline: list[EmotionEntry]
    scene_analysis:   list[SceneEntry]
    suggestions:      str
    caption:          str
    hashtags:         list[str]
    posting_time:     str
    has_face:         bool
    duration_sec:     float
    niche:            str
    created_at:       Optional[datetime] = None


# ── Job Status ────────────────────────────────────────────────────

class JobStatus(BaseModel):
    video_id:   str
    status:     str
    created_at: Optional[datetime] = None
    error:      Optional[str]      = None


# ── Upload Response ───────────────────────────────────────────────

class UploadResponse(BaseModel):
    video_id: str
    message:  str
    filename: str
