"""
Feature Extractor + Comparison Engine
───────────────────────────────────────
Combines all analysis outputs into a 14-dimensional feature vector,
normalises it, and computes cosine similarity against viral benchmarks.

Feature vector dimensions (in order):
  0  transcript_sentiment   (-1→1 → normalised 0-1)
  1  hook_strength          (0-10 → normalised 0-1)
  2  cta_present            (0 or 1)
  3  visual_energy          (0-10 → normalised 0-1)
  4  lighting_quality       (0-10 → normalised 0-1)
  5  composition            (0-10 → normalised 0-1)
  6  color_vibrancy         (0-10 → normalised 0-1)
  7  text_overlay           (0 or 1)
  8  avg_positive_emotion   (already 0-1)
  9  emotion_variance       (already 0-1)
  10 smile_ratio            (already 0-1)
  11 cuts_per_minute        (normalised: 0→0, ≥20→1)
  12 motion_intensity       (0-10 → normalised 0-1)
  13 music_present          (0 or 1)
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "transcript_sentiment",
    "hook_strength",
    "cta_present",
    "visual_energy",
    "lighting_quality",
    "composition",
    "color_vibrancy",
    "text_overlay",
    "avg_positive_emotion",
    "emotion_variance",
    "smile_ratio",
    "cuts_per_minute",
    "motion_intensity",
    "music_present",
]

# Human-readable labels for the dashboard
FEATURE_LABELS = {
    "transcript_sentiment":  "Audio Sentiment",
    "hook_strength":         "Hook Strength (First 3s)",
    "cta_present":           "Call-to-Action Present",
    "visual_energy":         "Visual Energy",
    "lighting_quality":      "Lighting Quality",
    "composition":           "Frame Composition",
    "color_vibrancy":        "Color Vibrancy",
    "text_overlay":          "On-Screen Text/Captions",
    "avg_positive_emotion":  "Positive Emotion (Face)",
    "emotion_variance":      "Emotion Dynamism",
    "smile_ratio":           "Smile Ratio",
    "cuts_per_minute":       "Editing Pace (Cuts/min)",
    "motion_intensity":      "Motion Intensity",
    "music_present":         "Background Music",
}


# ═══════════════════════════════════════════════════════════════════
#  BUILD FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════════

def build_feature_vector(
    audio_result:   dict,
    visual_result:  dict,
    emotion_result: dict,
    structure:      dict,
) -> dict:
    """
    Combines all analysis results into a normalised 14-dim feature vector.

    Args:
        audio_result:   output of audio_analyzer.analyze_audio()
        visual_result:  output of vision_analyzer.analyze_frames()
        emotion_result: output of emotion_analyzer.analyze_emotions()
        structure: {
            "cuts_per_minute": float,
            "motion_intensity": float,
            "music_present": bool,
            "duration_sec": float,
        }

    Returns:
        {
          "vector":  list[float],   # 14 values, all 0-1
          "raw":     dict,          # raw values before normalisation
          "labeled": dict           # { label: normalised_value }
        }
    """
    # ── raw values ────────────────────────────────────────────────
    sentiment_raw = float(audio_result.get("transcript_sentiment", 0.0))
    raw = {
        "transcript_sentiment":  _normalise_sentiment(sentiment_raw),
        "hook_strength":         _norm10(audio_result.get("hook_strength",        5.0)),
        "cta_present":           float(audio_result.get("cta_present",            0.0)),
        "visual_energy":         _norm10(visual_result.get("visual_energy",       5.0)),
        "lighting_quality":      _norm10(visual_result.get("lighting_quality",    5.0)),
        "composition":           _norm10(visual_result.get("composition",         5.0)),
        "color_vibrancy":        _norm10(visual_result.get("color_vibrancy",      5.0)),
        "text_overlay":          float(visual_result.get("text_overlay",          0.0)),
        "avg_positive_emotion":  float(emotion_result.get("avg_positive_emotion", 0.0)),
        "emotion_variance":      float(emotion_result.get("emotion_variance",     0.0)),
        "smile_ratio":           float(emotion_result.get("smile_ratio",          0.0)),
        "cuts_per_minute":       _norm_cuts(structure.get("cuts_per_minute",      0.0)),
        "motion_intensity":      _norm10(structure.get("motion_intensity",        5.0)),
        "music_present":         float(structure.get("music_present",             False)),
    }

    # Ensure every value is clamped 0-1
    vector = [round(min(1.0, max(0.0, raw[k])), 4) for k in FEATURE_NAMES]

    labeled = {
        FEATURE_LABELS[k]: round(min(1.0, max(0.0, raw[k])), 4)
        for k in FEATURE_NAMES
    }

    return {
        "vector":  vector,
        "raw":     {k: round(min(1.0, max(0.0, raw[k])), 4) for k in FEATURE_NAMES},
        "labeled": labeled,
    }


# ── normalisation helpers ─────────────────────────────────────────

def _norm10(v: float) -> float:
    """Normalise a 0-10 value to 0-1."""
    return round(float(v) / 10.0, 4)


def _normalise_sentiment(v: float) -> float:
    """Map -1..1 to 0..1."""
    return round((float(v) + 1.0) / 2.0, 4)


def _norm_cuts(v: float) -> float:
    """Normalise cuts/min: 0 → 0, ≥20 → 1."""
    return round(min(1.0, float(v) / 20.0), 4)


# ═══════════════════════════════════════════════════════════════════
#  VIRALITY SCORE  (cosine similarity)
# ═══════════════════════════════════════════════════════════════════

def compute_virality_score(
    user_vector:   list[float],
    viral_vectors: list[list[float]],
    has_face:      bool = False,
) -> float:
    """
    Computes cosine similarity between the user video vector and
    the average of all viral benchmark vectors.

    If has_face=False, emotion features (dims 8,9,10) are excluded from
    both vectors before comparison so no-face videos are not unfairly
    penalised for missing emotion data.

    Returns a float 0-100 (virality score percentage).
    """
    if not viral_vectors:
        logger.warning("No viral benchmarks available — returning default score 50.")
        return 50.0

    # Emotion feature indices: avg_positive_emotion(8), emotion_variance(9), smile_ratio(10)
    EMOTION_DIMS = {8, 9, 10}

    if not has_face:
        # Remove emotion dims from comparison — they are always 0 for no-face videos
        keep = [i for i in range(len(FEATURE_NAMES)) if i not in EMOTION_DIMS]
        u_vec = [user_vector[i] for i in keep]
        v_vecs = [[v[i] for i in keep] for v in viral_vectors]
        logger.info("No face detected — scoring on 11 non-emotion dimensions.")
    else:
        u_vec  = user_vector
        v_vecs = viral_vectors

    user_np   = np.array(u_vec).reshape(1, -1)
    viral_np  = np.array(v_vecs)
    avg_viral = viral_np.mean(axis=0).reshape(1, -1)

    similarity = cosine_similarity(user_np, avg_viral)[0][0]
    score      = round(float(similarity) * 100, 1)
    return min(100.0, max(0.0, score))


# ═══════════════════════════════════════════════════════════════════
#  GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def compute_gap_analysis(
    user_vector:   list[float],
    viral_vectors: list[list[float]],
    has_face:      bool = False,
) -> list[dict]:
    """
    Compares user feature values against viral average per dimension.

    Returns list of dicts sorted by gap (largest gap first):
        [
          {
            "feature":      str,   # internal key
            "label":        str,   # human readable
            "user_score":   float, # 0-1
            "viral_avg":    float, # 0-1
            "gap":          float, # viral_avg - user_score (positive = user is weaker)
            "status":       str,   # "critical" | "weak" | "ok" | "strong"
          },
          ...
        ]
    """
    if not viral_vectors:
        return []

    EMOTION_DIMS = {8, 9, 10}   # avg_positive_emotion, emotion_variance, smile_ratio

    viral_np  = np.array(viral_vectors)
    viral_avg = viral_np.mean(axis=0)

    gaps = []
    for i, feature_key in enumerate(FEATURE_NAMES):
        user_val  = float(user_vector[i])
        viral_val = float(viral_avg[i])

        # For no-face videos, mark emotion dims as not applicable
        if i in EMOTION_DIMS and not has_face:
            gaps.append({
                "feature":    feature_key,
                "label":      FEATURE_LABELS[feature_key],
                "user_score": 0.0,
                "viral_avg":  round(viral_val, 4),
                "gap":        0.0,
                "status":     "na",    # not applicable — no face in video
            })
            continue

        gap = round(viral_val - user_val, 4)

        if   gap > 0.3:  status = "critical"
        elif gap > 0.15: status = "weak"
        elif gap >= 0:   status = "ok"
        else:            status = "strong"

        gaps.append({
            "feature":    feature_key,
            "label":      FEATURE_LABELS[feature_key],
            "user_score": round(user_val,  4),
            "viral_avg":  round(viral_val, 4),
            "gap":        gap,
            "status":     status,
        })

    # Sort: critical first, then weak, ok, strong, na last
    order = {"critical": 0, "weak": 1, "ok": 2, "strong": 3, "na": 4}
    gaps.sort(key=lambda x: (order[x["status"]], -abs(x["gap"])))
    return gaps
