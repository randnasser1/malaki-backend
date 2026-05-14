from google.cloud import firestore
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class FirestoreService:
    def __init__(self):
        self.db = firestore.Client()

    async def save_daily_wellbeing(self, data: Dict):
        try:
            doc_id = f"{data['childId']}_{data['date']}"
            doc_ref = self.db.collection("wellbeing_daily_summary").document(doc_id)
            doc_ref.set(data)
            return doc_ref.id
        except Exception as e:
            print(f"❌ save_daily_wellbeing failed: {e}")
            return None

    async def get_wellbeing_history(self, child_id: str, days: int) -> Optional[List[Dict]]:
        start_date = datetime.now() - timedelta(days=days)
        cutoff = int(start_date.timestamp() * 1000)
        docs = self.db.collection("wellbeing_daily_summary")\
            .where("childId", "==", child_id)\
            .limit(days * 2)\
            .stream()
        results = [
            doc.to_dict() for doc in docs
            if doc.to_dict().get("timestamp", 0) >= cutoff
        ]
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return results[:days] if results else None

    async def get_linked_child_id(self, parent_id: str) -> Optional[str]:
        doc = self.db.collection("users").document(parent_id).get()
        if doc.exists:
            return doc.to_dict().get("childId")
        return None

    async def get_child_info(self, child_id: str) -> Dict:
        doc = self.db.collection("users").document(child_id).get()
        if doc.exists:
            return doc.to_dict()
        return {"name": "Your Child"}

    async def get_music_insights(self, child_id: str) -> Optional[Dict]:
        docs = self.db.collection("music_tracking")\
            .where("childId", "==", child_id)\
            .limit(20)\
            .stream()
        rows = sorted(
            [doc.to_dict() for doc in docs],
            key=lambda x: x.get("timestamp", 0), reverse=True
        )
        return rows[0] if rows else None

    async def get_app_usage_insights(self, child_id: str) -> Optional[Dict]:
        docs = self.db.collection("app_usage")\
            .where("childId", "==", child_id)\
            .limit(20)\
            .stream()
        rows = sorted(
            [doc.to_dict() for doc in docs],
            key=lambda x: x.get("date", ""), reverse=True
        )
        return rows[0] if rows else None

    async def get_recent_risk_alerts(self, child_id: str, hours: int) -> List[Dict]:
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
        docs = self.db.collection("risk_assessment")\
            .where("childId", "==", child_id)\
            .limit(100)\
            .stream()
        rows = [
            doc.to_dict() for doc in docs
            if doc.to_dict().get("timestamp", 0) >= start_time
        ]
        rows.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return rows[:20]

    def _build_threat_type(self, author_label: Optional[str], grooming_prob: Optional[float]) -> str:
        gp = grooming_prob or 0.0
        if author_label == "Adult" and gp >= 0.65:
            return "confirmed_predator"
        elif author_label == "Minor" and gp >= 0.65:
            return "peer_predatory"
        elif author_label == "Adult":
            return "suspicious_adult"
        elif author_label == "Minor":
            return "risky_peer"
        elif gp >= 0.65:
            return "predatory_unknown_author"
        else:
            return "suspicious_unknown"

    def _build_explainability(self, author_label: Optional[str], threat_type: str,
                              grooming_prob: Optional[float], reason: Optional[str]) -> str:
        gp_pct = f"{(grooming_prob or 0) * 100:.0f}%"
        base = reason or ""
        if threat_type == "confirmed_predator":
            return (f"HIGH RISK: An adult contact is showing predatory behavior "
                    f"(grooming score: {gp_pct}). This requires immediate action. {base}").strip()
        elif threat_type == "peer_predatory":
            return (f"NEEDS INVESTIGATION: A peer (possibly another child) is showing predatory patterns "
                    f"(grooming score: {gp_pct}). This may be teen grooming — review the messages carefully. {base}").strip()
        elif threat_type == "suspicious_adult":
            return f"Suspicious adult contact. An adult is communicating in ways that raise concern. {base}".strip()
        elif threat_type == "risky_peer":
            return f"A peer interaction was flagged as potentially risky. Review the conversation context. {base}".strip()
        elif threat_type == "predatory_unknown_author":
            return f"Predatory patterns detected from an unidentified contact (grooming score: {gp_pct}). {base}".strip()
        else:
            return base or "Suspicious behavior detected. Monitor this contact."

    async def save_risk_alert(self, child_id: str, event_id: str, risk_level: str,
                          risk_score: float, reason: Optional[str], source_text: Optional[str],
                          author_label: Optional[str] = None, grooming_prob: Optional[float] = None,
                          author_confidence: Optional[float] = None):
        try:
            threat_type = self._build_threat_type(author_label, grooming_prob)
            explainability = self._build_explainability(author_label, threat_type, grooming_prob, reason)
            alert_data = {
                "childId":            child_id,
                "eventId":            event_id,
                "riskLevel":          risk_level,
                "confidenceScore":    risk_score,
                "blockReasons":       [reason] if reason else [],
                "messageText":        source_text[:500] if source_text else "",
                "url":                source_text[:500] if source_text and source_text.startswith("http") else "",
                "timestamp":          int(datetime.now().timestamp() * 1000),
                "authorLabel":        author_label or "Unknown",
                "authorConfidence":   author_confidence if author_confidence is not None else 0.0,
                "threatType":         threat_type,
                "groomingProbability": grooming_prob or 0.0,
                "explainabilityText": explainability,
            }
            self.db.collection("risk_assessment").add(alert_data)
            print(f"🚨 {risk_level} risk alert saved for {event_id} ({threat_type})")
        except Exception as e:
            print(f"❌ save_risk_alert failed for {event_id}: {e}")
    # ── Music emotion pipeline ────────────────────────────────────────────────

    async def get_unprocessed_music_docs(self, child_id: str) -> List[Dict]:
        """Return music_tracking docs that haven't been emotion-classified yet."""
        print(f"🔍 Looking for unprocessed music docs for child {child_id}")
        all_docs = self.db.collection("music_tracking")\
            .where("childId", "==", child_id)\
            .limit(50)\
            .stream()

        result = []
        for doc in all_docs:
            d = doc.to_dict()
            ep = d.get("emotion_processed")
            if ep is False or ep is None or "emotion_processed" not in d:
                d["_doc_id"] = doc.id
                result.append(d)
                print(f"   Found unprocessed doc: {doc.id} with {len(d.get('entries', []))} entries")

        result.sort(key=lambda x: x.get("timestamp", 0))
        print(f"✅ Total unprocessed docs: {len(result)}")
        return result

    async def save_music_emotion_results(self, doc_id: str, emotion_results: List[Dict]):
        """Persist emotion classifications back onto the music_tracking document."""
        self.db.collection("music_tracking").document(doc_id).update({
            "emotion_results":   emotion_results,
            "emotion_processed": True,
            "processed_at":      int(datetime.now().timestamp() * 1000),
        })

    async def get_music_emotion_series(self, child_id: str, days: int = 30) -> List[Dict]:
        """Return [{date, emotion}] for TBATS analysis."""
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        print(f"🔍 Fetching music emotion series for child {child_id}")
        docs = self.db.collection("music_tracking")\
            .where("childId", "==", child_id)\
            .limit(100)\
            .stream()

        series = []
        for doc in docs:
            d = doc.to_dict()
            if not d.get("emotion_processed", False):
                continue
            if d.get("timestamp", 0) < cutoff:
                continue
            
            emotion_results = d.get("emotion_results", [])
            for entry in emotion_results:
                emotion = entry.get("emotion")
                date_str = entry.get("date")
                if not date_str:
                    ts = entry.get("timestamp")
                    if ts:
                        date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                if date_str and emotion:
                    series.append({"date": date_str, "emotion": emotion})
        
        print(f"📊 Total entries for TBATS: {len(series)}")
        return series

    
    async def get_music_emotion_summary(self, child_id: str) -> Dict:
        """Aggregate dominant RF emotion from the last 10 processed music_tracking docs."""
        MOOD_LABEL = {
            "happy":     "Positive / Upbeat",
            "party":     "Social / Energetic",
            "energetic": "Energetic",
            "romantic":  "Calm / Romantic",
            "chill":     "Relaxed",
            "calm":      "Calm / Peaceful",
            "focus":     "Focused / Neutral",
            "sad":       "Melancholic",
        }
        docs = self.db.collection("music_tracking") \
            .where("childId", "==", child_id) \
            .limit(50) \
            .stream()

        all_rows = sorted(
            [d for doc in docs for d in [doc.to_dict()] if d.get("emotion_processed")],
            key=lambda x: x.get("timestamp", 0), reverse=True
        )[:10]

        emotion_counts: Dict[str, int] = {}
        total_tracks = 0
        for row in all_rows:
            for entry in row.get("emotion_results", []):
                emotion = entry.get("emotion")
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    total_tracks += 1

        if not emotion_counts:
            return {"has_data": False, "total_tracks": 0}

        dominant = max(emotion_counts, key=emotion_counts.get)
        return {
            "has_data":            True,
            "dominant_emotion":    dominant,
            "mood_description":    MOOD_LABEL.get(dominant, dominant.capitalize()),
            "emotion_distribution": emotion_counts,
            "total_tracks":        total_tracks,
        }

    async def get_app_usage_full_series(self, child_id: str, days: int = 30) -> List[Dict]:
        """Return [{date, totalTimeMin, apps}] for per-category TBATS analysis."""
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        docs = self.db.collection("app_usage")\
            .where("childId", "==", child_id)\
            .limit(60)\
            .stream()
        rows = []
        for d in docs:
            data = d.to_dict()
            if data.get("date") and data.get("timestamp", 0) >= cutoff:
                rows.append({
                    "date":         data["date"],
                    "totalTimeMin": data.get("totalTimeMin", 0),
                    "apps":         data.get("apps", []),
                })
        rows.sort(key=lambda x: x["date"])
        return rows

    async def get_app_usage_series(self, child_id: str, days: int = 30) -> List[Dict]:
        """Return [{date, totalTimeMin}] for TBATS analysis."""
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        docs = self.db.collection("app_usage")\
            .where("childId", "==", child_id)\
            .limit(60)\
            .stream()
        rows = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("date") and data.get("timestamp", 0) >= cutoff:
                rows.append({
                    "date": data["date"],
                    "totalTimeMin": data.get("totalTimeMin", 0)
                })
        rows.sort(key=lambda x: x["date"])
        return rows


    async def save_tbats_cache(self, child_id: str, result: Dict):
        """Save the full /analyze/tbats response to the top-level tbats_cache collection."""
        self.db.collection("tbats_cache").document(child_id)\
            .set({**result, "cached_at": int(datetime.now().timestamp() * 1000)})
        
    async def get_tbats_cache(self, child_id: str) -> Optional[Dict]:
        """Return the cached full TBATS response, or None if it doesn't exist yet."""
        doc = self.db.collection("tbats_cache").document(child_id).get()
        return doc.to_dict() if doc.exists else None
    async def get_social_score(self, child_id: str) -> Optional[int]:
        docs = self.db.collection("app_usage")\
            .where("childId", "==", child_id)\
            .limit(20)\
            .stream()
        all_rows = sorted(
            [doc.to_dict() for doc in docs],
            key=lambda x: x.get("date", ""), reverse=True
        )
        usage_days = all_rows[:7]
        if len(usage_days) < 3:
            return None
        social_apps = ["whatsapp", "messenger", "instagram", "telegram", "snapchat", "imessage"]
        total_social_time = 0
        for day in usage_days:
            apps = day.get("apps", [])
            for app in apps:
                if any(s in app.get("package", "").lower() for s in social_apps):
                    total_social_time += app.get("time_min", 0)
        avg_social_time = total_social_time / len(usage_days)
        return min(100, int((avg_social_time / 120) * 100))

    async def save_event_analysis(self, child_id: str, event_id: str, event_type: str,
                               sentiment_score: float, emotion_vector: Dict[str, float],
                               grooming_prob: float, risk_level: str, risk_score: float,
                               anomaly_score: float, explanation: Optional[str],
                               timestamp_utc: int, message_text: Optional[str] = None,
                               author_label: Optional[str] = None,
                               author_confidence: Optional[float] = None):
        """Save ALL analyzed events to Firestore for dashboard trends"""
        try:
            analysis_data = {
                "childId":            child_id,
                "eventId":            event_id,
                "eventType":          event_type,
                "sentimentScore":     sentiment_score,
                "emotionVector":      emotion_vector,
                "groomingProbability": grooming_prob,
                "riskLevel":          risk_level,
                "riskScore":          risk_score,
                "anomalyScore":       anomaly_score,
                "explanation":        explanation,
                "timestamp":          timestamp_utc,
                "analyzedAt":         int(datetime.now().timestamp() * 1000),
             "authorLabel": author_label,  # 🆕 "Minor" or "Adult"
        "authorConfidence": author_confidence,  # 🆕 0.0 to 1.0
    }
            
            if message_text:
                analysis_data["messageText"] = message_text[:500]
                analysis_data["messageLength"] = len(message_text)
            
            # Use event_id as document ID — idempotent, retries overwrite not duplicate
            self.db.collection("event_analysis").document(event_id).set(analysis_data)
            print(f"📊 Saved analysis for {event_id}")
            
        except Exception as e:
            print(f"❌ save_event_analysis failed: {e}")