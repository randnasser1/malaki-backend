from pydantic import BaseModel
from typing import List, Optional, Dict

class SentimentRequest(BaseModel):
    text: str
    context: Optional[str] = None

class SentimentResponse(BaseModel):
    sentiment_score: float
    primary_emotion: str
    emotion_scores: Dict[str, float]
    confidence: float

class GroomingRequest(BaseModel):
    messages: List[str]
    sender_labels: Optional[List[str]] = None

class GroomingResponse(BaseModel):
    grooming_probability: float
    risk_level: str
    detected_stage: Optional[str] = None
    explanation: str

class ProfilingRequest(BaseModel):
    text: str

class ProfilingResponse(BaseModel):
    predicted_age_group: str
    confidence: float

class EmbeddingRequest(BaseModel):
    child_id: str
    embedding: List[float]  # 384-dim vector from device
    timestamp: Optional[int] = None

class EmbeddingResponse(BaseModel):
    grooming_probability: float
    risk_level: str
    sentiment_score: Optional[float] = None

class DailyWellbeingRequest(BaseModel):
    child_id: str
    date: str
    daily_mood: str
    daily_mood_score: float
    journal_text: Optional[str] = None
    timestamp: Optional[int] = None

class DailyWellbeingResponse(BaseModel):
    emotional_wellbeing_score: float
    journal_sentiment: Optional[float]
    message: str

class WellbeingSummary(BaseModel):
    date: str
    emotional_score: float
    daily_mood: str
    journal_sentiment: Optional[float]

class DashboardDataResponse(BaseModel):
    child_name: str
    sentiment_trends: List[Dict]
    wellbeing_indicators: List[Dict]
    music_insights: Dict
    app_usage: Dict
    risk_alerts: List[Dict]

# ── Event Analysis (ERD pipeline) ────────────────────────────────────────────

class EventPayload(BaseModel):
    event_id: str
    device_id: str
    event_type: str               # MESSAGE | URL | APP_USAGE | JOURNAL
    source_app: Optional[str] = None
    sender_role: Optional[str] = None   # SELF | OTHER | UNKNOWN
    timestamp_utc: int
    text: Optional[str] = None

class EventBatchRequest(BaseModel):
    events: List[EventPayload]

class EventResult(BaseModel):
    event_id: str
    grooming_prob: float
    stage_label: Optional[str]
    sentiment_score: float
    emotion_vector: Dict[str, float]
    anomaly_score: float
    final_risk_score: float
    risk_level: str               # LOW | MEDIUM | HIGH
    threshold_used: float
    model_version: str
    is_alert_triggered: bool
    top_tokens: List[Dict]
    human_reason: Optional[str]

class EventBatchResponse(BaseModel):
    results: List[EventResult]