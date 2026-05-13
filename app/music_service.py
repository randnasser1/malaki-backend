import os
import re
import csv
import asyncio
import httpx
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List

try:
    from rapidfuzz import fuzz, process as fuzz_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️ rapidfuzz not installed — fuzzy CSV lookup disabled")

RECCOBEATS_HOST = "reccobeats1.p.rapidapi.com"
RECCOBEATS_BASE = f"https://{RECCOBEATS_HOST}"

# Columns the RF model expects (must match inference.py feature_array order)
RF_FEATURE_ORDER = [
    "valence", "energy", "danceability", "acousticness",
    "instrumentalness", "tempo", "loudness", "speechiness",
    "liveness", "key", "mode",
]

RF_DEFAULTS = {
    "valence": 0.5, "energy": 0.5, "danceability": 0.5,
    "acousticness": 0.3, "instrumentalness": 0.0, "tempo": 120.0,
    "loudness": -10.0, "speechiness": 0.05, "liveness": 0.1,
    "key": 5.0, "mode": 1.0,
}


def _norm(s: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class MusicService:
    def __init__(self, inference_engine, csv_path: str = "audio_features_clean.csv"):
        self.engine = inference_engine
        self.df: Optional[pd.DataFrame] = None
        self._search_keys: List[str] = []
        self._api_key: str = os.getenv("RAPIDAPI_KEY", "")
        self._track_cache: Dict[str, Tuple[str, str]] = {}  # (artist|track) → (emotion, source)
        self._load_csv(csv_path)

    def _load_csv(self, path: str):
        try:
            self.df = pd.read_csv(path)
            self.df["_key"] = (
                self.df["artist_name"].fillna("").apply(_norm)
                + " - "
                + self.df["track_name"].fillna("").apply(_norm)
            )
            self._search_keys = self.df["_key"].tolist()
            print(f"✅ audio_features_clean.csv loaded: {len(self.df):,} tracks")
        except Exception as e:
            print(f"⚠️ Could not load audio_features_clean.csv: {e}")

    # ── 1. Fuzzy CSV lookup ───────────────────────────────────────────────────

    def fuzzy_lookup(self, artist: str, track: str, threshold: int = 82) -> Optional[str]:
        """Return emotion from CSV if a confident fuzzy match is found."""
        if not RAPIDFUZZ_AVAILABLE or self.df is None:
            return None
        query = _norm(f"{artist} - {track}")
        match = fuzz_process.extractOne(
            query, self._search_keys,
            scorer=fuzz.token_sort_ratio
        )
        if match and match[1] >= threshold:
            row = self.df[self.df["_key"] == match[0]]
            if not row.empty:
                return str(row.iloc[0]["emotion"])
        return None

    # ── 2. Reccobeats API ─────────────────────────────────────────────────────

    async def fetch_reccobeats_features(
        self, artist: str, track: str
    ) -> Optional[Dict[str, float]]:
        """Fetch audio features from Reccobeats via RapidAPI."""
        if not self._api_key:
            return None
        query = f"{artist} {track}"
        headers = {
            "x-rapidapi-key": self._api_key,
            "x-rapidapi-host": RECCOBEATS_HOST,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{RECCOBEATS_BASE}/track",
                    params={"q": query, "limit": "1"},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            # Reccobeats wraps results under "content" key
            items = data.get("content") or data.get("tracks") or data.get("items") or []
            if not items:
                return None

            af = items[0].get("audioFeatures") or items[0].get("audio_features") or {}
            if not af:
                return None

            return {
                "valence":          float(af.get("valence",          RF_DEFAULTS["valence"])),
                "energy":           float(af.get("energy",           RF_DEFAULTS["energy"])),
                "danceability":     float(af.get("danceability",     RF_DEFAULTS["danceability"])),
                "acousticness":     float(af.get("acousticness",     RF_DEFAULTS["acousticness"])),
                "instrumentalness": float(af.get("instrumentalness", RF_DEFAULTS["instrumentalness"])),
                "tempo":            float(af.get("tempo",            RF_DEFAULTS["tempo"])),
                "loudness":         float(af.get("loudness",         RF_DEFAULTS["loudness"])),
                "speechiness":      float(af.get("speechiness",      RF_DEFAULTS["speechiness"])),
                "liveness":         float(af.get("liveness",         RF_DEFAULTS["liveness"])),
                "key":              float(af.get("key",              RF_DEFAULTS["key"])),
                "mode":             float(af.get("mode",             RF_DEFAULTS["mode"])),
            }
        except Exception as e:
            print(f"⚠️ Reccobeats failed for '{artist} - {track}': {e}")
            return None

    # ── 3. RF classification ─────────────────────────────────────────────────

    def classify_from_features(self, features: Dict[str, float]) -> Optional[str]:
        result = self.engine.classify_music_mood(features)
        if "error" in result:
            return None
        return result.get("mood")

    # ── 4. Append a classified track back to the CSV ─────────────────────────

    def _append_to_csv(
        self, artist: str, track: str, emotion: str,
        features: Dict[str, float], csv_path: str = "audio_features_clean.csv"
    ):
        """Persist a newly classified track so future fuzzy lookups hit the CSV."""
        row = {
            "artist_name": artist,
            "track_name":  track,
            "emotion":     emotion,
        }
        row.update({k: features.get(k, RF_DEFAULTS[k]) for k in RF_FEATURE_ORDER})

        file_exists = os.path.isfile(csv_path)
        fieldnames = ["artist_name", "track_name", "emotion"] + RF_FEATURE_ORDER
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            # Reload index so the new track is fuzzy-matchable next time
            self._load_csv(csv_path)
        except Exception as e:
            print(f"⚠️ Could not append to {csv_path}: {e}")

    # ── 5. Full pipeline ─────────────────────────────────────────────────────

    async def get_emotion(
        self, artist: str, track: str
    ) -> Optional[Tuple[str, str]]:
        """
        Returns (emotion, source) or None if no data is available.
        source is one of: 'csv', 'rf_reccobeats'.
        Returns None when both CSV and ReccoBeats fail — caller must skip the track.
        """
        cache_key = f"{_norm(artist)}|||{_norm(track)}"
        if cache_key in self._track_cache:
            return self._track_cache[cache_key]  # may be None (previous miss)

        # Step 1 — fuzzy CSV lookup
        emotion = self.fuzzy_lookup(artist, track)
        if emotion:
            result: Optional[Tuple[str, str]] = (emotion, "csv")
            self._track_cache[cache_key] = result
            return result

        # Step 2 — ReccoBeats audio features → RF classification
        features = await self.fetch_reccobeats_features(artist, track)
        if features:
            emotion = self.classify_from_features(features)
            if emotion:
                result = (emotion, "rf_reccobeats")
                self._track_cache[cache_key] = result
                self._append_to_csv(artist, track, emotion, features)
                return result

        # No data found — cache the miss and skip this track
        self._track_cache[cache_key] = None
        return None

    # ── 5. Batch processing ──────────────────────────────────────────────────

    async def process_music_doc(self, doc: Dict) -> Dict:
        """
        Takes one Firestore music_tracking document.
        Returns an enriched version with emotion_results and emotion_processed=True.
        """
        entries = doc.get("entries", [])
        emotion_results = []
        reccobeats_calls = 0

        for entry in entries:
            track_info = entry if isinstance(entry, dict) else {}
            if "track_info" in track_info:
                track_info = track_info["track_info"]

            artist = str(track_info.get("artist", "")).strip()
            track = str(track_info.get("track", "")).strip()
            ts = entry.get("timestamp") if isinstance(entry, dict) else None

            if not artist or not track:
                continue

            result = await self.get_emotion(artist, track)
            if result is None:
                # ReccoBeats returned nothing — skip, don't pollute averages
                continue

            emotion, source = result

            if source == "rf_reccobeats":
                reccobeats_calls += 1
                if reccobeats_calls % 5 == 0:
                    await asyncio.sleep(1)

            # Convert timestamp to date string
            date_str = None
            if ts:
                from datetime import datetime
                date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")

            emotion_results.append({
                "artist":    artist,
                "track":     track,
                "emotion":   emotion,
                "source":    source,
                "timestamp": ts,
                "date":      date_str,
            })

        return {
            "emotion_results":    emotion_results,
            "emotion_processed":  True,
        }
    # ── Singleton ─────────────────────────────────────────────────────────────────

    _music_service: Optional["MusicService"] = None


def init_music_service(inference_engine) -> "MusicService":
    global _music_service
    _music_service = MusicService(inference_engine)
    return _music_service


def get_music_service() -> Optional["MusicService"]:
    return _music_service
