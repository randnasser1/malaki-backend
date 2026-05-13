"""
TBATS-based anomaly detection for music emotion and app usage time series.

Logic mirrors the tbats_music.ipynb notebook exactly:
  - Map emotions to [-1, +1] scores
  - Daily aggregation + rolling smooth (window=5)
  - Drift = diff of smoothed signal
  - Volatility = rolling std (window=7)
  - Positive / negative drift events at 1.5σ threshold
  - Anomalies via modified Z-score (MAD) at 97th percentile
  - Optional TBATS 7-day forecast
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

try:
    from tbats import TBATS as TBATSEstimator
    _TBATS_OK = True
except ImportError:
    _TBATS_OK = False
    print("⚠️ tbats not installed — forecasts will be empty until installed")

# Emotion → numeric score  (same map as notebook)
EMOTION_SCORE_MAP: Dict[str, float] = {
    "happy":     1.0,
    "party":     0.8,
    "energetic": 0.6,
    "romantic":  0.4,
    "chill":     0.2,
    "calm":      0.0,
    "focus":    -0.2,
    "sad":      -1.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_daily_series(dated_scores: List[Tuple[str, float]]) -> pd.Series:
    """
    dated_scores: [("YYYY-MM-DD", score), ...]
    Returns daily-resampled, interpolated Series indexed by date.
    """
    df = pd.DataFrame(dated_scores, columns=["date", "score"])
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["score"].mean()
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).interpolate(method="time")
    return daily


def _tbats_analysis(series: pd.Series) -> Dict:
    """
    Core analysis matching notebook cells 1–4.
    Returns a summary dict suitable for Firestore storage and dashboard display.
    Works with any number of days:
      < 2  → minimal placeholder
      2–6  → z-score + linear slope + consecutive-decline heuristics (no TBATS)
      7+   → full TBATS analysis
    """
    n = len(series)

    if n < 2:
        print(f"  [_tbats_analysis] n={n} → PLACEHOLDER (not enough data)")
        score = round(float(series.iloc[-1]), 3) if n > 0 else 0.0
        return {
            "concern_level":      "LOW",
            "current_score":      score,
            "avg_score":          score,
            "current_volatility": 0.0,
            "anomaly_dates":      [],
            "negative_drift_dates": [],
            "positive_drift_dates": [],
            "forecast_7d":        [],
            "total_days_analyzed": n,
            "data_note":          "Collecting initial data (need 2+ days)",
            "analyzed_at":        datetime.utcnow().isoformat(),
        }

    if n < 7:
        vals = series.values.astype(float)
        current_score = float(vals[-1])
        avg_score     = float(np.mean(vals))
        volatility    = float(np.std(vals)) if n > 1 else 0.0

        # Z-score: how many std-devs is today from the short-window mean?
        z_score = (current_score - avg_score) / max(volatility, 1e-6)

        # Linear trend slope (units/day) via least-squares
        if n >= 3:
            slope = float(np.polyfit(np.arange(n, dtype=float), vals, 1)[0])
        elif n == 2:
            slope = float(vals[-1] - vals[0])
        else:
            slope = 0.0

        # Count consecutive daily declines from the most recent point
        consec_decline = 0
        for i in range(n - 1, 0, -1):
            if vals[i] < vals[i - 1]:
                consec_decline += 1
            else:
                break

        # Signals (negative direction = concerning for emotion scores)
        sharp_drop   = z_score < -1.5
        moderate_drop = z_score < -1.0
        trending_down = slope < -0.05 and n >= 3
        persistent_fall = consec_decline >= 2
        high_volatility = volatility > 0.35

        if sharp_drop and trending_down:
            concern_level = "HIGH"
        elif (moderate_drop and persistent_fall) or (high_volatility and moderate_drop):
            concern_level = "HIGH"
        elif moderate_drop or (persistent_fall and trending_down) or high_volatility:
            concern_level = "MEDIUM"
        else:
            concern_level = "LOW"

        print(f"  [_tbats_analysis] n={n} → HEURISTIC (z-score/slope)  "
              f"z={z_score:.3f}  slope={slope:.4f}  consec_decline={consec_decline}  "
              f"volatility={volatility:.3f}  → concern={concern_level}")

        return {
            "concern_level":        concern_level,
            "current_score":        round(current_score, 3),
            "avg_score":            round(avg_score, 3),
            "current_volatility":   round(volatility, 3),
            "deviation_z":          round(z_score, 3),
            "trend_slope":          round(slope, 4),
            "consec_decline_days":  consec_decline,
            "anomaly_dates":        [],
            "negative_drift_dates": [],
            "positive_drift_dates": [],
            "forecast_7d":          [],
            "total_days_analyzed":  n,
            "data_note":            f"Collecting data ({n}/7 days for full pattern analysis)",
            "analyzed_at":          datetime.utcnow().isoformat(),
        }

    print(f"  [_tbats_analysis] n={n} → FULL TBATS path (MAD anomaly + drift + forecast)")
    # ── Smooth ────────────────────────────────────────────────────────────────
    smooth = series.rolling(5, center=True, min_periods=1).mean()

    # ── Drift & volatility ────────────────────────────────────────────────────
    drift = smooth.diff()
    volatility = series.rolling(7, min_periods=1).std().fillna(0.0)

    drift_std = float(drift.std()) if float(drift.std()) > 1e-9 else 1e-6
    threshold = drift_std * 1.5

    pos_drift_mask = drift > threshold
    neg_drift_mask = drift < -threshold

    # ── Anomaly detection (MAD / modified Z-score) ────────────────────────────
    vals = series.values.astype(float)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median))) + 1e-6
    z_scores = 0.6745 * (vals - median) / mad
    anomaly_threshold = float(np.percentile(np.abs(z_scores), 97))
    anomaly_mask = np.abs(z_scores) > anomaly_threshold

    # ── TBATS forecast ────────────────────────────────────────────────────────
    # Require 14+ points (2 full weekly seasons) before imposing seasonal_periods=[7].
    # Fewer points cause the optimizer to diverge and spend minutes without converging.
    forecast_7d: List[float] = []
    if _TBATS_OK:
        try:
            clean = smooth.dropna()
            if len(clean) >= 7:
                seasonal = [7] if len(clean) >= 14 else None
                model = TBATSEstimator(seasonal_periods=seasonal, use_arma_errors=False)
                fitted = model.fit(clean.values)
                forecast_7d = [round(float(v), 3) for v in fitted.forecast(steps=7)]
        except Exception as e:
            print(f"TBATS fit error: {e}")

    # ── Summarise ─────────────────────────────────────────────────────────────
    date_strs = series.index.strftime("%Y-%m-%d").tolist()
    anomaly_dates    = [date_strs[i] for i, v in enumerate(anomaly_mask) if v]
    neg_drift_dates  = [d for d, v in zip(date_strs, neg_drift_mask) if v]
    pos_drift_dates  = [d for d, v in zip(date_strs, pos_drift_mask) if v]

    last_3_dates = date_strs[-3:]
    recent_neg     = any(d in neg_drift_dates  for d in last_3_dates)
    recent_anomaly = any(d in anomaly_dates     for d in last_3_dates)
    current_vol    = float(volatility.iloc[-1])

    if recent_neg and recent_anomaly:
        concern_level = "HIGH"
    elif recent_neg or (recent_anomaly and current_vol > 0.3):
        concern_level = "MEDIUM"
    else:
        concern_level = "LOW"

    print(f"  [_tbats_analysis] FULL result: concern={concern_level}  "
          f"anomaly_dates={anomaly_dates[-3:]}  neg_drift={neg_drift_dates[-3:]}  "
          f"forecast_7d={forecast_7d[:3]}{'...' if len(forecast_7d) > 3 else ''}")

    return {
        "concern_level":       concern_level,
        "current_score":       round(float(series.iloc[-1]), 3),
        "avg_score":           round(float(series.mean()), 3),
        "current_volatility":  round(current_vol, 3),
        "anomaly_dates":       anomaly_dates[-10:],
        "negative_drift_dates": neg_drift_dates[-10:],
        "positive_drift_dates": pos_drift_dates[-10:],
        "forecast_7d":         forecast_7d,
        "total_days_analyzed": len(series),
        "analyzed_at":         datetime.utcnow().isoformat(),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_music_emotions(emotion_entries: List[Dict]) -> Dict:
    """
    emotion_entries: [{"date": "YYYY-MM-DD", "emotion": "happy"}, ...]
    Runs TBATS on the emotion score time series.
    """
    scored: List[Tuple[str, float]] = []
    for e in emotion_entries:
        date = e.get("date") or e.get("timestamp")
        emotion = e.get("emotion", "")
        if date and emotion in EMOTION_SCORE_MAP:
            # Accept epoch ms as date
            if isinstance(date, (int, float)):
                date = datetime.utcfromtimestamp(date / 1000).strftime("%Y-%m-%d")
            scored.append((str(date)[:10], EMOTION_SCORE_MAP[emotion]))

    if not scored:
        return {"error": "No classified emotion data yet", "concern_level": "LOW", "type": "music_emotion"}

    series = _build_daily_series(scored)
    result = _tbats_analysis(series)
    result["type"] = "music_emotion"
    result["data_points"] = len(scored)
    return result


def analyze_app_usage(usage_entries: List[Dict]) -> Dict:
    """
    usage_entries: [{"date": "YYYY-MM-DD", "totalTimeMin": 180}, ...]
    Normalises screen time to [0, 1] (cap = 480 min / 8 h) then runs TBATS.
    Sudden spikes or prolonged high usage are flagged as concerning.
    """
    scored: List[Tuple[str, float]] = []
    for e in usage_entries:
        date = e.get("date")
        total = float(e.get("totalTimeMin") or e.get("total_time_min") or 0.0)
        if date:
            scored.append((str(date)[:10], np.log1p(total) / np.log1p(1440)))

    if not scored:
        return {"error": "No app usage data yet", "concern_level": "LOW", "type": "app_usage"}

    series = _build_daily_series(scored)
    result = _tbats_analysis(series)
    result["type"] = "app_usage"

    n = result.get("total_days_analyzed", 7)
    if n < 7:
        # For screen time, HIGH usage is the concern (positive z = bad), so override
        # the emotion-oriented concern level from _tbats_analysis.
        z     = result.get("deviation_z", 0.0)
        slope = result.get("trend_slope", 0.0)
        vol   = result.get("current_volatility", 0.0)
        consec = result.get("consec_decline_days", 0)

        # Consecutive *increases* from tail
        vals = series.values.astype(float)
        consec_rise = 0
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] > vals[i - 1]:
                consec_rise += 1
            else:
                break

        sharp_spike    = z > 1.5
        moderate_spike = z > 1.0
        trending_up    = slope > 0.05 and n >= 3
        persistent_rise = consec_rise >= 2
        high_vol       = vol > 0.35

        if sharp_spike and trending_up:
            concern_level = "HIGH"
        elif (moderate_spike and persistent_rise) or (high_vol and moderate_spike):
            concern_level = "HIGH"
        elif moderate_spike or (persistent_rise and trending_up) or high_vol:
            concern_level = "MEDIUM"
        else:
            concern_level = "LOW"

        result["concern_level"] = concern_level
    else:
        # Full analysis: high volatility alone is concerning for screen time
        if result.get("current_volatility", 0) > 0.4 and result["concern_level"] == "LOW":
            result["concern_level"] = "MEDIUM"

    result["data_points"] = len(scored)
    return result


def combined_concern_level(music_result: Dict, usage_result: Dict) -> str:
    """
    If both signals are concerning simultaneously, escalate.
    """
    levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    m = levels.get(music_result.get("concern_level", "LOW"), 0)
    u = levels.get(usage_result.get("concern_level", "LOW"), 0)
    combined = max(m, u) + (1 if m > 0 and u > 0 else 0)
    combined = min(combined, 2)
    return ["LOW", "MEDIUM", "HIGH"][combined]


# ── App category mapping ──────────────────────────────────────────────────────

APP_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "social":        ["whatsapp", "instagram", "telegram", "snapchat", "messenger",
                      "facebook", "twitter", "tiktok", "discord", "wechat", "viber",
                      "line", "kik", "skype"],
    "entertainment": ["youtube", "netflix", "spotify", "disney", "twitch", "vlc",
                      "prime", "hulu", "hbo", "peacock", "apple.tv", "soundcloud",
                      "deezer", "pandora", "crunchyroll"],
    "gaming":        ["game", "pubg", "roblox", "minecraft", "clash", "candy",
                      "temple", "subway", "brawl", "fortnite", "among", "pokemon",
                      "mobile.legend", "freefire", "garena"],
    "productivity":  ["chrome", "gmail", "docs", "sheets", "office", "word",
                      "excel", "outlook", "drive", "notion", "keep", "calendar"],
    "education":     ["duolingo", "khan", "coursera", "udemy", "classroom", "zoom",
                      "teams", "school", "quizlet", "brilliant", "photomath"],
    "browsing":      ["browser", "firefox", "edge", "opera", "safari", "brave",
                      "duck", "uc.browser"],
}


def categorize_app(package_or_name: str) -> str:
    name = (package_or_name or "").lower()
    for category, keywords in APP_CATEGORY_KEYWORDS.items():
        if any(k in name for k in keywords):
            return category
    return "other"


def _tbats_residual_analysis(series: pd.Series) -> Dict:
    """
    Trains TBATS on series[:-1], forecasts the last point, computes residual.
    Flags today as anomaly if |residual| > 2 * in-sample residual std.
    Requires at least 3 real data points — with n<3 the std fallback (0.1) makes
    any non-zero change a guaranteed anomaly, producing false positives.
    """
    n = len(series)
    today_str    = series.index[-1].strftime("%Y-%m-%d")
    actual_today = float(series.iloc[-1])

    if n < 3:
        print(f"    [residual] n={n} → SKIPPED (need 3+ points to avoid std=0.1 false positives)")
        return {
            "enough_data":       False,
            "days_collected":    n,
            "today":             today_str,
            "actual":            round(actual_today, 3),
        }

    train_vals = series.iloc[:-1].values.astype(float)

    # Baseline: linear trend residuals as std estimate.
    # This is always used — TBATS below may override the forecast value but
    # the tbats library doesn't expose in-sample y_hat, so we compute std
    # from the linear fit residuals rather than from the flat mean.
    xs = np.arange(len(train_vals), dtype=float)
    if len(train_vals) >= 2:
        lin_slope, lin_intercept = np.polyfit(xs, train_vals, 1)
        lin_fitted    = lin_intercept + lin_slope * xs
        lin_resid     = train_vals - lin_fitted
        residual_std  = max(float(np.std(lin_resid)), 1e-6)
        forecast_today = float(lin_intercept + lin_slope * len(train_vals))
    else:
        residual_std   = 1e-6
        forecast_today = float(train_vals[0]) if len(train_vals) > 0 else actual_today
    fit_method = "linear"

    if _TBATS_OK and n >= 5:
        try:
            seasonal = [7] if n >= 14 else None  # need 2 full seasons to fit weekly seasonality
            model    = TBATSEstimator(seasonal_periods=seasonal, use_arma_errors=False)
            fitted   = model.fit(train_vals)
            # Override only the forecast — std stays linear-based because
            # the tbats library does not expose in-sample fitted values (y_hat).
            forecast_today = float(fitted.forecast(steps=1)[0])
            fit_method = "tbats+linear_std"
        except Exception as e:
            print(f"TBATS residual fit error: {e}")

    residual   = actual_today - forecast_today
    is_anomaly = abs(residual) > 2.0 * residual_std

    if residual > 0.5 * residual_std:
        direction = "high"
    elif residual < -0.5 * residual_std:
        direction = "low"
    else:
        direction = "normal"

    print(f"    [residual] fit={fit_method}  n={n}  actual={actual_today:.3f}  "
          f"forecast={forecast_today:.3f}  residual={residual:.3f}  "
          f"std={residual_std:.3f}  anomaly={is_anomaly}  dir={direction}")

    return {
        "enough_data":       True,
        "days_collected":    n,
        "today":             today_str,
        "actual":            round(actual_today,   3),
        "forecast":          round(forecast_today, 3),
        "residual":          round(residual,       3),
        "residual_std":      round(residual_std,   3),
        "is_anomaly_today":  is_anomaly,
        "anomaly_direction": direction,
        "fit_method":        fit_method,
        "analyzed_at":       datetime.utcnow().isoformat(),
    }


def analyze_app_usage_by_category(usage_entries: List[Dict]) -> Dict:
    """
    usage_entries: [{"date": "YYYY-MM-DD", "apps": [{"app_name": str, "time_min": float}]}, ...]
    Runs residual-based TBATS per app category on daily normalised screen time [0,1].
    Returns {"has_enough_data": bool, "days_collected": int, "categories": {cat: result}}.
    Each category is evaluated independently — categories with 2+ days get anomaly detection
    (simple mean-based for 2-4 days, TBATS for 5+), fewer days show {"enough_data": False}.
    """
    from collections import defaultdict

    cat_day: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_dates: set = set()

    for entry in usage_entries:
        date = str(entry.get("date", ""))[:10]
        if not date:
            continue
        all_dates.add(date)
        for app in entry.get("apps", []):
            name     = str(app.get("app_name", "") or app.get("package", "") or "")
            time_min = float(app.get("time_min", 0) or 0)
            cat      = categorize_app(name)
            cat_day[cat][date] += time_min

    distinct_days = len(all_dates)
    if distinct_days == 0:
        return {
            "has_enough_data": False,
            "days_collected":  0,
            "needs_days":      2,
        }

    print(f"  [app_by_category] distinct_days={distinct_days}  categories={list(cat_day.keys())}")
    results: Dict[str, Dict] = {}
    for cat, day_totals in cat_day.items():
        n_days = len(day_totals)
        if n_days < 2:
            print(f"    [{cat}] n={n_days} → SKIPPED (need 2+ days)")
            results[cat] = {"enough_data": False, "days_collected": n_days}
            continue
        print(f"    [{cat}] n={n_days}  avg_min={round(sum(day_totals.values())/len(day_totals),1)}")
        scored   = [(d, np.log1p(t) / np.log1p(1440)) for d, t in sorted(day_totals.items())]
        series   = _build_daily_series(scored)
        analysis = _tbats_residual_analysis(series)
        analysis["avg_daily_min"] = round(sum(day_totals.values()) / len(day_totals), 1)
        results[cat] = analysis

    return {
        "has_enough_data": True,
        "days_collected":  distinct_days,
        "categories":      results,
    }


def analyze_music_emotions_by_emotion(emotion_entries: List[Dict]) -> Dict:
    """
    emotion_entries: [{"date": "YYYY-MM-DD", "emotion": "happy"}, ...]
    Runs residual-based TBATS per emotion label on daily play-count time series.
    Returns {"has_enough_data": bool, "days_collected": int, "emotions": {emotion: result}}.
    Each emotion is evaluated independently — emotions with 2+ days get anomaly detection
    (simple mean-based for 2-4 days, TBATS for 5+), fewer days show {"enough_data": False}.
    """
    from collections import defaultdict

    emotion_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_dates: set = set()

    for entry in emotion_entries:
        date    = str(entry.get("date", ""))[:10]
        emotion = entry.get("emotion", "")
        if isinstance(date, (int, float)):
            date = datetime.utcfromtimestamp(date / 1000).strftime("%Y-%m-%d")
        if not date or not emotion:
            continue
        all_dates.add(date)
        emotion_day[emotion][date] += 1

    distinct_days = len(all_dates)
    if distinct_days == 0:
        return {
            "has_enough_data": False,
            "days_collected":  0,
            "needs_days":      2,
        }

    print(f"  [music_by_emotion] distinct_days={distinct_days}  emotions={list(emotion_day.keys())}")
    results: Dict[str, Dict] = {}
    for emotion, day_counts in emotion_day.items():
        n_days = len(day_counts)
        if n_days < 2:
            print(f"    [{emotion}] n={n_days} → SKIPPED (need 2+ days)")
            results[emotion] = {"enough_data": False, "days_collected": n_days}
            continue
        print(f"    [{emotion}] n={n_days}  counts={dict(sorted(day_counts.items()))}")
        scored   = [(d, float(c)) for d, c in sorted(day_counts.items())]
        series   = _build_daily_series(scored)
        analysis = _tbats_residual_analysis(series)
        results[emotion] = analysis

    return {
        "has_enough_data": True,
        "days_collected":  distinct_days,
        "emotions":        results,
    }
