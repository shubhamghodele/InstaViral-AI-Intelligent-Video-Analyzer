"""
Emotion Analyzer Service
─────────────────────────
Step 1 — OpenCV Haar Cascade: detect if a human face is present.
Step 2 — EmotiEffLibRecognizerOnnx: classify emotion per face frame.

If no face is detected across any frame, emotion scores default to 0
and the feature vector skips emotion dimensions gracefully.
"""

import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)

# positive emotions contribute to virality
POSITIVE_EMOTIONS = {"happy", "surprise", "surprised"}

# Haar cascade path (bundled with opencv-python — always available)
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade  = None
_emoti_model   = None


# ── lazy loaders ─────────────────────────────────────────────────

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if _face_cascade.empty():
            logger.warning("Haar cascade failed to load.")
            _face_cascade = None
    return _face_cascade


def _get_emoti_model():
    global _emoti_model
    if _emoti_model is None:
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizerOnnx
            _emoti_model = EmotiEffLibRecognizerOnnx(model_name="enet_b0_8_best_afew")
            logger.info("EmotiEffLib model loaded (ONNX).")
        except Exception as e:
            logger.warning(f"EmotiEffLib not available: {e}")
            _emoti_model = None
    return _emoti_model


# ═══════════════════════════════════════════════════════════════════
#  FACE DETECTION  (OpenCV Haar Cascade)
# ═══════════════════════════════════════════════════════════════════

def has_face(frame_path: Path) -> bool:
    """Returns True if at least one face is detected in the frame."""
    cascade = _get_face_cascade()
    if cascade is None:
        return False

    img = cv2.imread(str(frame_path))
    if img is None:
        return False

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces) > 0


# ═══════════════════════════════════════════════════════════════════
#  EMOTION CLASSIFICATION  (EmotiEffLib ONNX)
# ═══════════════════════════════════════════════════════════════════

def classify_emotion(frame_path: Path) -> dict | None:
    """
    Classify the dominant emotion in a frame.
    Returns { "emotion": str, "score": float } or None.
    """
    model = _get_emoti_model()
    if model is None:
        return None

    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    try:
        emotions, scores = model.predict_emotions(img, logits=False)
        if not emotions:
            return None

        # emotions = list of label strings, scores = 2D numpy array
        scores_arr = np.array(scores).flatten()
        dominant_idx     = int(np.argmax(scores_arr))
        dominant_emotion = str(emotions[0]).lower() if isinstance(emotions, list) else str(emotions).lower()
        dominant_score   = float(scores_arr[dominant_idx])

        return {"emotion": dominant_emotion, "score": round(dominant_score, 3)}

    except Exception as e:
        logger.warning(f"Emotion classification failed for {frame_path.name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  FULL EMOTION ANALYSIS  (main entry point)
# ═══════════════════════════════════════════════════════════════════

def analyze_emotions(frame_paths: list[Path], fps_sample: float = 1.0) -> dict:
    """
    Run emotion analysis on all key frames.
    Returns summary scores + full emotion timeline.
    """
    if not frame_paths:
        return _default_emotion_result()

    emotion_data = []

    for i, fp in enumerate(frame_paths[:15]):
        if not has_face(fp):
            continue
        result = classify_emotion(fp)
        if result:
            emotion_data.append({
                "time_sec": round(i / fps_sample, 1),
                "emotion":  result["emotion"],
                "score":    result["score"],
            })

    if not emotion_data:
        logger.info("No face detected in any frame — skipping emotion scores.")
        return _default_emotion_result(has_face=False)

    logger.info(f"Emotion analysis: {len(emotion_data)} frames with faces.")

    positive_scores = [
        e["score"] for e in emotion_data
        if e["emotion"] in POSITIVE_EMOTIONS
    ]
    all_scores   = [e["score"] for e in emotion_data]
    avg_positive = round(float(np.mean(positive_scores)) if positive_scores else 0.0, 3)
    smile_ratio  = round(len(positive_scores) / len(emotion_data), 3)
    variance     = round(float(np.std(all_scores)) if len(all_scores) > 1 else 0.0, 3)
    emotion_var  = min(1.0, round(variance / 0.3, 3))

    return {
        "has_face":             True,
        "avg_positive_emotion": avg_positive,
        "emotion_variance":     emotion_var,
        "smile_ratio":          smile_ratio,
        "emotion_timeline":     emotion_data,
    }


def _default_emotion_result(has_face: bool = False) -> dict:
    return {
        "has_face":             has_face,
        "avg_positive_emotion": 0.0,
        "emotion_variance":     0.0,
        "smile_ratio":          0.0,
        "emotion_timeline":     [],
    }
