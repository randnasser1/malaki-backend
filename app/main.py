import re
import asyncio
import logging
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from typing import Dict, List
from datetime import datetime, timedelta
from .models import (
    SentimentRequest, SentimentResponse,
    GroomingRequest, GroomingResponse,
    DailyWellbeingRequest, DailyWellbeingResponse,
    EventBatchRequest, EventBatchResponse, EventResult,
)
from .inference import inference_engine
from .firestore_service import FirestoreService
from .music_service import init_music_service, get_music_service
from . import tbats_service
import google.auth.transport.requests
from google.oauth2 import service_account
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="google")
# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("malaki")

# Initialize Firestore service
firestore = FirestoreService()

# ── LIFESPAN EVENT HANDLER (MUST BE BEFORE app = FastAPI()) ───────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Starting Malaki API — loading models…")
    inference_engine.load_all_models()
    init_music_service(inference_engine)
    log.info("✅ All models loaded. models_loaded=%s", inference_engine.models_loaded)
    yield
    log.info("🛑 Shutting down Malaki API")

# Create app with lifespan
app = FastAPI(
    title="Malaki AI Guardian API",
    description="Backend API for AI Child Guardian",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"message": "Malaki API is running successfully"}

@app.get("/health")
async def health_check():
    log.info("GET /health — models_loaded=%s device=%s",
             inference_engine.models_loaded,
             getattr(inference_engine, "device", "?"))
    return {"status": "ok", "models_loaded": inference_engine.models_loaded}

@app.post("/predict/sentiment", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    if not inference_engine.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    result = inference_engine.analyze_sentiment(request.text)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/predict/grooming", response_model=GroomingResponse)
async def predict_grooming(request: GroomingRequest):
    if not inference_engine.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    result = inference_engine.detect_grooming(request.messages)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/predict/music-mood")
async def predict_music_mood(features: Dict[str, float]):
    if not inference_engine.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    result = inference_engine.classify_music_mood(features)
    return result

# ========== DASHBOARD ENDPOINTS ==========

@app.post("/wellbeing/daily")
async def submit_daily_wellbeing(request: DailyWellbeingRequest):
    log.info("POST /wellbeing/daily — child=%s date=%s mood=%s mood_score=%.2f journal_len=%s",
             request.child_id, request.date, request.daily_mood, request.daily_mood_score,
             len(request.journal_text) if request.journal_text else 0)

    if not inference_engine.models_loaded:
        log.error("Models not loaded — rejecting /wellbeing/daily")
        raise HTTPException(status_code=503, detail="Models not loaded")

    journal_sentiment = None
    if request.journal_text:
        log.debug("  Running DistilBERT on journal text (%d chars)…", len(request.journal_text))
        sentiment_result = inference_engine.analyze_sentiment(request.journal_text)
        if "error" not in sentiment_result:
            journal_sentiment = sentiment_result["sentiment_score"]
            log.info("  DistilBERT journal sentiment=%.3f primary_emotion=%s",
                     journal_sentiment, sentiment_result.get("primary_emotion", "?"))
        else:
            log.warning("  DistilBERT error: %s", sentiment_result["error"])
    else:
        log.debug("  No journal text — skipping DistilBERT")

    if journal_sentiment is not None:
        emotional_score = (request.daily_mood_score * 0.4) + (journal_sentiment * 0.6)
        log.info("  Emotional score = %.2f*0.4 + %.2f*0.6 = %.3f",
                 request.daily_mood_score, journal_sentiment, emotional_score)
    else:
        emotional_score = request.daily_mood_score
        log.info("  Emotional score = mood_score only = %.3f", emotional_score)

    wellbeing_data = {
        "childId": request.child_id,
        "date": request.date,
        "dailyMood": request.daily_mood,
        "dailyMoodScore": request.daily_mood_score,
        "journalText": request.journal_text,
        "journalSentiment": journal_sentiment,
        "emotionalWellbeingScore": emotional_score,
        "timestamp": request.timestamp or int(datetime.now().timestamp() * 1000)
    }

    await firestore.save_daily_wellbeing(wellbeing_data)
    log.info("  ✅ Saved wellbeing_daily_summary for child=%s date=%s", request.child_id, request.date)

    return DailyWellbeingResponse(
        emotional_wellbeing_score=emotional_score,
        journal_sentiment=journal_sentiment,
        message="Wellbeing data saved successfully"
    )

@app.get("/wellbeing/history/{child_id}")
async def get_wellbeing_history(child_id: str, days: int = 7):
    wellbeing_history = await firestore.get_wellbeing_history(child_id, days)
    
    if len(wellbeing_history) >= 2:
        first_score = wellbeing_history[0].get("emotionalWellbeingScore", 0.5)
        last_score = wellbeing_history[-1].get("emotionalWellbeingScore", 0.5)
        trend = "improving" if last_score > first_score else "declining" if last_score < first_score else "stable"
    else:
        trend = "stable"
    
    return {
        "history": wellbeing_history,
        "trend": trend,
        "average_score": sum(w.get("emotionalWellbeingScore", 0.5) for w in wellbeing_history) / len(wellbeing_history) if wellbeing_history else 0.5
    }

@app.get("/dashboard/{parent_id}")
async def get_dashboard_data(parent_id: str):
    child_id = await firestore.get_linked_child_id(parent_id)
    if not child_id:
        raise HTTPException(status_code=404, detail="No child linked to this parent")
    
    child_info = await firestore.get_child_info(child_id)
    wellbeing_history = await firestore.get_wellbeing_history(child_id, 7)
    music_insights = await firestore.get_music_insights(child_id)
    app_usage = await firestore.get_app_usage_insights(child_id)
    risk_alerts = await firestore.get_recent_risk_alerts(child_id, 24)
    
    sentiment_trends = []
    if wellbeing_history:
        for day in get_last_7_days():
            day_data = next((w for w in wellbeing_history if w.get("date") == day), None)
            if day_data:
                score = int(day_data.get("emotionalWellbeingScore", 0.5) * 100)
                sentiment_trends.append({"day": day[:3], "score": score, "has_data": True})
            else:
                sentiment_trends.append({"day": day[:3], "score": None, "has_data": False})
    else:
        sentiment_trends = None
    
    wellbeing_indicators = []
    if wellbeing_history:
        emotional_score = calculate_emotional_score(wellbeing_history)
        wellbeing_indicators.append({"category": "Emotional", "score": emotional_score, "has_data": True})
    else:
        wellbeing_indicators.append({"category": "Emotional", "score": None, "has_data": False, "message": "No mood or journal entries yet"})
    
    social_score = await firestore.get_social_score(child_id)
    if social_score is not None:
        wellbeing_indicators.append({"category": "Social", "score": social_score, "has_data": True})
    else:
        wellbeing_indicators.append({"category": "Social", "score": None, "has_data": False, "message": "Insufficient social interaction data"})
    
    if app_usage and app_usage.get("activity_score") is not None:
        wellbeing_indicators.append({"category": "Activity", "score": app_usage.get("activity_score"), "has_data": True})
    else:
        wellbeing_indicators.append({"category": "Activity", "score": None, "has_data": False, "message": "Collecting app usage patterns"})
    
    if app_usage and app_usage.get("sleep_score") is not None:
        wellbeing_indicators.append({"category": "Sleep", "score": app_usage.get("sleep_score"), "has_data": True})
    else:
        wellbeing_indicators.append({"category": "Sleep", "score": None, "has_data": False, "message": "Analyzing late-night usage patterns"})
    
    return {
        "child_name": child_info.get("name", "Your Child"),
        "sentiment_trends": sentiment_trends,
        "wellbeing_indicators": wellbeing_indicators,
        "music_insights": music_insights if music_insights else {"has_data": False, "message": "No music listening data collected yet"},
        "app_usage": app_usage if app_usage else {"has_data": False, "message": "No app usage data collected yet"},
        "risk_alerts": risk_alerts if risk_alerts else []
    }

MODEL_VERSION = "1.0.0"
RISK_THRESHOLD = 0.5

_TIMESTAMP_PREFIX = re.compile(
    r'^\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*'   # [2026-05-07 20:33:30]
    r'(\[[\w.]+\]\s*)?'                                        # optional [com.package]
)

def _clean_text(text: str) -> str:
    """Strip accessibility-service timestamp/package prefixes captured in old builds."""
    return _TIMESTAMP_PREFIX.sub("", text).strip()


def send_push_notification_to_parent(child_id: str, risk_level: str, message: str):
    """Send FCM HTTP v1 notification to parent's device"""
    try:
        # 1. Use your existing service account
        credentials = service_account.Credentials.from_service_account_file(
            'service-account.json',
            scopes=['https://www.googleapis.com/auth/firebase.messaging']
        )
        
        # 2. Get access token
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)
        access_token = credentials.token
        
        # 3. Get parent's FCM token from Firestore
        parent_tokens = firestore.db.collection("parent_tokens").limit(1).stream()
        
        # 4. Your project ID
        PROJECT_ID = "malak-f3515"
        
        for doc in parent_tokens:
            fcm_token = doc.to_dict().get("fcmToken")
            if not fcm_token:
                continue
            
            # 5. FCM HTTP v1 endpoint
            url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "message": {
                    "token": fcm_token,
                    "data": {
                        "riskLevel": risk_level,
                        "message": message[:200],
                        "childId": child_id,
                        "timestamp": str(int(datetime.now().timestamp() * 1000))
                    },
                    "android": {
                        "priority": "high"
                    }
                }
            }
            
            response = requests.post(url, json=payload, headers=headers)
            print(f"✅ FCM v1 notification sent: {response.status_code}")
            
    except Exception as e:
        print(f"❌ FCM v1 failed: {e}")
# Call this in /events/analyze when HIGH risk

@app.post("/events/analyze", response_model=EventBatchResponse)
async def analyze_events(request: EventBatchRequest):
    import re as _re
    total = len(request.events)
    message_events = [e for e in request.events if e.event_type == "MESSAGE"]
    skipped_types = [e.event_type for e in request.events if e.event_type != "MESSAGE"]

    log.info("POST /events/analyze — %d total events, %d MESSAGE, %d skipped (%s)",
             total, len(message_events), len(skipped_types), set(skipped_types) or "none")

    if not message_events:
        log.debug("  No MESSAGE events in batch — returning empty")
        return EventBatchResponse(results=[])

    if not inference_engine.models_loaded:
        log.error("Models not loaded — rejecting /events/analyze")
        raise HTTPException(status_code=503, detail="Models not loaded")

    results: list[EventResult] = []
    to_analyze = []

    for event in message_events:
        existing = firestore.db.collection("event_analysis").document(event.event_id).get()
        if existing.exists:
            d = existing.to_dict()
            log.debug("  ⏭ CACHED  event=%s… risk=%s grooming=%.3f",
                      event.event_id[:12], d.get("riskLevel"), d.get("groomingProbability", 0))
            results.append(EventResult(
                event_id=event.event_id,
                grooming_prob=float(d.get("groomingProbability", 0.0)),
                stage_label=None,
                sentiment_score=float(d.get("sentimentScore", 0.5)),
                emotion_vector={k: float(v) for k, v in d.get("emotionVector", {}).items()},
                anomaly_score=0.0,
                final_risk_score=float(d.get("riskScore", 0.0)),
                risk_level=d.get("riskLevel", "LOW"),
                threshold_used=RISK_THRESHOLD,
                model_version=MODEL_VERSION,
                is_alert_triggered=d.get("riskLevel", "LOW") in ("HIGH", "MEDIUM"),
                top_tokens=[],
                human_reason=d.get("explanation"),
            ))
        else:
            to_analyze.append(event)

    log.info("  %d cached, %d need analysis", len(results), len(to_analyze))

    for i, event in enumerate(to_analyze, 1):
        raw_text = event.text or ""
        text = _clean_text(raw_text)
        log.info("  [%d/%d] Analyzing event=%s… text_len=%d (raw=%d) source=%s",
                 i, len(to_analyze), event.event_id[:12],
                 len(text), len(raw_text), event.source_app or "?")

        if not text:
            log.warning("    Empty text after cleaning — skipping model inference")

        sentiment_score = 0.5
        emotion_vector = {}
        grooming_prob = 0.0
        stage_label = None
        human_reason = None
        author_label = "Unknown"
        author_confidence = 0.0
        primary_emotion = "neutral"

        if text:
            multi = inference_engine.analyze_message_all_models(text)
            if "error" not in multi:
                sentiment_score   = float(multi.get("sentiment_score", 0.5))
                emotion_vector    = multi.get("emotion_vector", {})
                grooming_prob     = float(multi.get("grooming_prob", 0.0))
                stage_label       = multi.get("stage_label")
                human_reason      = multi.get("human_reason")
                author_label      = multi.get("bert_author_label", "Unknown")
                author_confidence = multi.get("bert_adult_prob", 0.5)
                primary_emotion   = multi.get("primary_emotion", "neutral")

                top_emotions = sorted(emotion_vector.items(), key=lambda x: x[1], reverse=True)[:3]
                log.info("    grooming=%.3f  sentiment=%.3f  emotion=%s  author=%s(%.2f)",
                         grooming_prob, sentiment_score,
                         " ".join(f"{e}:{s:.2f}" for e, s in top_emotions),
                         author_label, author_confidence)
                if human_reason:
                    log.info("    reason: %s", human_reason[:120])
            else:
                log.error("    Model error: %s", multi["error"])

        if grooming_prob >= 0.65:
            risk_level = "HIGH"
        elif grooming_prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        final_risk_score = grooming_prob
        log.info("    → risk=%s (grooming_threshold=0.65/0.40)", risk_level)

        await firestore.save_event_analysis(
            child_id=event.device_id,
            event_id=event.event_id,
            event_type="MESSAGE",
            sentiment_score=sentiment_score,
            emotion_vector=emotion_vector,
            grooming_prob=grooming_prob,
            risk_level=risk_level,
            risk_score=final_risk_score,
            anomaly_score=0.0,
            explanation=human_reason,
            timestamp_utc=event.timestamp_utc,
            message_text=text,
            author_label=author_label,
            author_confidence=author_confidence
        )
        log.debug("    Saved to event_analysis")

        if risk_level == "HIGH":
            log.warning("    🚨 HIGH risk — sending push notification to parent")
            send_push_notification_to_parent(event.device_id, risk_level, text[:100])

        if risk_level in ("HIGH", "MEDIUM"):
            await firestore.save_risk_alert(
                child_id=event.device_id,
                event_id=event.event_id,
                risk_level=risk_level,
                risk_score=final_risk_score,
                reason=human_reason,
                source_text=text,
                author_label=author_label,
                grooming_prob=grooming_prob,
                author_confidence=author_confidence,
            )
            log.info("    Saved risk_assessment (%s)", risk_level)

        results.append(EventResult(
            event_id=event.event_id,
            grooming_prob=grooming_prob,
            stage_label=stage_label,
            sentiment_score=sentiment_score,
            emotion_vector=emotion_vector,
            anomaly_score=0.0,
            final_risk_score=final_risk_score,
            risk_level=risk_level,
            threshold_used=RISK_THRESHOLD,
            model_version=MODEL_VERSION,
            is_alert_triggered=risk_level in ("HIGH", "MEDIUM"),
            top_tokens=[],
            human_reason=human_reason,
        ))

    log.info("POST /events/analyze done — %d results returned", len(results))
    return EventBatchResponse(results=results)
# ── Music emotion classification ──────────────────────────────────────────────

@app.post("/music/process/{child_id}")
async def process_child_music(child_id: str):
    log.info("POST /music/process/%s", child_id)
    svc = get_music_service()
    if svc is None:
        log.error("  Music service not ready")
        raise HTTPException(status_code=503, detail="Music service not ready")

    docs = await firestore.get_unprocessed_music_docs(child_id)
    log.info("  Found %d unprocessed music_tracking docs", len(docs))

    if not docs:
        log.info("  Nothing to process — returning early")
        return {"message": "No unprocessed music docs found", "processed": 0}

    total_tracks = 0
    for i, doc in enumerate(docs, 1):
        entries = doc.get("entries", [])
        log.info("  Doc %d/%d — %d track entries", i, len(docs), len(entries))
        enriched = await svc.process_music_doc(doc)
        results = enriched["emotion_results"]
        emotion_counts = {}
        for r in results:
            emotion_counts[r.get("emotion", "?")] = emotion_counts.get(r.get("emotion", "?"), 0) + 1
        log.info("    Classified %d tracks: %s", len(results),
                 " ".join(f"{e}={c}" for e, c in sorted(emotion_counts.items())))
        await firestore.save_music_emotion_results(doc["_doc_id"], results)
        total_tracks += len(results)

    log.info("  ✅ music/process done — %d docs, %d tracks total", len(docs), total_tracks)
    return {"message": f"Processed {len(docs)} docs, {total_tracks} tracks",
            "docs_processed": len(docs), "tracks_classified": total_tracks}

# ── TBATS anomaly detection ───────────────────────────────────────────────────

_TBATS_CACHE_TTL_MS = 2 * 60 * 60 * 1000  # 2 hours


async def _compute_tbats(child_id: str, days: int) -> Dict:
    """Run the full TBATS pipeline and return the response dict."""
    music_series      = await firestore.get_music_emotion_series(child_id, days)
    usage_series      = await firestore.get_app_usage_series(child_id, days)
    usage_full_series = await firestore.get_app_usage_full_series(child_id, days)
    music_summary     = await firestore.get_music_emotion_summary(child_id)

    log.info("  Music series: %d entries  |  Usage series: %d days  |  Music summary has_data=%s",
             len(music_series), len(usage_series), music_summary.get("has_data", False))
    if music_series:
        dates = sorted({e["date"] for e in music_series})
        log.debug("  Music date range: %s → %s", dates[0], dates[-1])
    if usage_series:
        log.debug("  Usage date range: %s → %s  totalTimeMin range: %s–%s min",
                  usage_series[0]["date"], usage_series[-1]["date"],
                  min(e["totalTimeMin"] for e in usage_series),
                  max(e["totalTimeMin"] for e in usage_series))

    loop = asyncio.get_event_loop()
    music_result, usage_result, app_category_result, music_emotion_result = await asyncio.gather(
        loop.run_in_executor(None, tbats_service.analyze_music_emotions, music_series),
        loop.run_in_executor(None, tbats_service.analyze_app_usage, usage_series),
        loop.run_in_executor(None, tbats_service.analyze_app_usage_by_category, usage_full_series),
        loop.run_in_executor(None, tbats_service.analyze_music_emotions_by_emotion, music_series),
    )

    log.info("  Music TBATS → concern=%s  days_analyzed=%s  note=%s",
             music_result.get("concern_level"), music_result.get("total_days_analyzed"),
             music_result.get("data_note") or music_result.get("error", ""))
    log.info("  Usage TBATS → concern=%s  data_points=%s  volatility=%.3f  note=%s",
             usage_result.get("concern_level"), usage_result.get("data_points"),
             usage_result.get("current_volatility", 0),
             usage_result.get("data_note") or usage_result.get("error", ""))

    if music_result.get("anomaly_dates"):
        log.warning("  Music anomaly dates: %s", music_result["anomaly_dates"])
    if usage_result.get("anomaly_dates"):
        log.warning("  Usage anomaly dates: %s", usage_result["anomaly_dates"])

    music_result["current_emotion"]  = music_summary.get("dominant_emotion", "")
    music_result["mood_description"] = music_summary.get("mood_description", "")
    music_result["total_tracks"]     = music_summary.get("total_tracks", 0)
    music_result["has_music_data"]   = music_summary.get("has_data", False)

    combined = tbats_service.combined_concern_level(music_result, usage_result)
    log.info("  Combined concern level: %s", combined)

    if combined == "HIGH":
        reasons = []
        if music_result.get("concern_level") == "HIGH":
            reasons.append("Music emotion anomaly detected")
        if usage_result.get("concern_level") in ("HIGH", "MEDIUM"):
            reasons.append("Abnormal screen-time pattern")
        log.warning("  🚨 HIGH combined concern — saving risk alert: %s", reasons)
        await firestore.save_risk_alert(
            child_id=child_id,
            event_id=f"tbats_{int(datetime.now().timestamp())}",
            risk_level="HIGH",
            risk_score=1.0,
            reason="; ".join(reasons),
            source_text=None,
        )

    log.info("  app_categories=%s music_emotions=%s",
             list(app_category_result.get("categories", {}).keys()) if app_category_result.get("has_enough_data") else f"not_enough({app_category_result.get('days_collected', 0)})",
             list(music_emotion_result.get("emotions", {}).keys()) if music_emotion_result.get("has_enough_data") else f"not_enough({music_emotion_result.get('days_collected', 0)})")

    return {
        "child_id":               child_id,
        "concern_level":          combined,
        "music_analysis":         music_result,
        "usage_analysis":         usage_result,
        "app_category_analysis":  app_category_result,
        "music_emotion_analysis": music_emotion_result,
        "days_analyzed":          days,
    }


async def _refresh_tbats_cache(child_id: str, days: int):
    """Background task: recompute TBATS and overwrite the cache."""
    log.info("TBATS background refresh started for %s", child_id)
    try:
        result = await _compute_tbats(child_id, days)
        await firestore.save_tbats_cache(child_id, result)
        log.info("TBATS background refresh complete for %s", child_id)
    except Exception as e:
        log.error("TBATS background refresh failed for %s: %s", child_id, e)


@app.get("/analyze/tbats/{child_id}")
async def run_tbats_analysis(
    child_id: str,
    background_tasks: BackgroundTasks,
    days: int = 30,
    force_refresh: bool = False,
):
    log.info("GET /analyze/tbats/%s?days=%d force_refresh=%s", child_id, days, force_refresh)

    if not force_refresh:
        cached = await firestore.get_tbats_cache(child_id)
        if cached:
            age_ms  = int(datetime.now().timestamp() * 1000) - cached.get("cached_at", 0)
            age_min = age_ms / 60000
            if age_ms < _TBATS_CACHE_TTL_MS:
                log.info("GET /analyze/tbats/%s → CACHE HIT (age=%.1f min)", child_id, age_min)
                return {k: v for k, v in cached.items() if k != "cached_at"}
            else:
                log.info("GET /analyze/tbats/%s → STALE CACHE (age=%.1f min) — returning stale, refreshing in background", child_id, age_min)
                background_tasks.add_task(_refresh_tbats_cache, child_id, days)
                return {k: v for k, v in cached.items() if k != "cached_at"}

    # Cache miss: return a lightweight placeholder immediately so Android doesn't
    # hang, then compute and save in the background. Next open hits the cache.
    log.info("GET /analyze/tbats/%s → CACHE MISS — returning placeholder, computing in background", child_id)
    background_tasks.add_task(_refresh_tbats_cache, child_id, days)
    return {
        "child_id":      child_id,
        "concern_level": "LOW",
        "status":        "computing",
        "music_analysis": {
            "concern_level":       "LOW",
            "has_music_data":      False,
            "data_note":           "Analyzing patterns, check back shortly",
            "total_days_analyzed": 0,
        },
        "usage_analysis": {
            "concern_level": "LOW",
            "data_points":   0,
            "data_note":     "Analyzing patterns, check back shortly",
        },
        "app_category_analysis":  {"has_enough_data": False, "days_collected": 0, "needs_days": 2},
        "music_emotion_analysis": {"has_enough_data": False, "days_collected": 0, "needs_days": 2},
        "days_analyzed": days,
    }

def get_last_7_days():
    dates = []
    for i in range(6, -1, -1):
        date = datetime.now() - timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
    return dates

def calculate_emotional_score(wellbeing_history):
    if not wellbeing_history:
        return 75
    scores = [w.get("emotionalWellbeingScore", 0.5) for w in wellbeing_history]
    return int((sum(scores) / len(scores)) * 100)