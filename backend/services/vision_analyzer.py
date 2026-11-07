"""
Vision Analyzer Service
────────────────────────
Uses Groq Llama-4-Maverick (multimodal) to analyse key video frames.
Returns structured JSON scores per frame, then averages them into
a single visual feature set.

Scores returned per frame:
  • visual_energy    (0-10)
  • lighting_quality (0-10)
  • composition      (0-10)
  • color_vibrancy   (0-10)
  • text_overlay     (0 or 1)
  • close_up_face    (0 or 1)
  • scene_description (string)
"""

import os
import re
import json
import base64
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_groq_client: Groq | None = None

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not set in .env file!")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _encode_image(image_path: Path) -> str:
    """Base64-encode an image file for the Groq Vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════
#  SINGLE FRAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_frame(frame_path: Path) -> dict:
    """
    Analyse a single frame with Groq Llama 4 Maverick Vision.
    Returns a dict of scores.
    """
    try:
        client     = _get_client()
        b64_image  = _encode_image(frame_path)

        prompt = """You are an expert Instagram Reels content analyst specialising in Indian short-form content (Reels, Shorts).

Analyse this video frame and return ONLY a valid JSON object with these exact keys:

{
  "visual_energy": <0-10, how exciting/dynamic does the frame look? Movement, action, and expressive faces score higher>,
  "lighting_quality": <0-10, how well-lit is the subject? Bright even lighting = 9-10, dark/shadowy = 1-3, backlit silhouette = 2-4>,
  "composition": <0-10, is the subject well-framed? Centred or rule-of-thirds = 8-10, cut-off subject or too much empty space = 3-5>,
  "color_vibrancy": <0-10, are colours rich, saturated and visually appealing? Bright vibrant palette = 8-10, muted/grey/dull = 2-4>,
  "text_overlay": <1 if ANY text, captions, subtitles, or on-screen graphics are clearly visible, else 0>,
  "close_up_face": <1 if a human face is clearly and prominently visible (not tiny in the background), else 0>,
  "scene_description": "<one specific sentence: who/what is in the frame, what are they doing, what is the setting/background>"
}

Scoring guide:
  9-10 = Professional quality — studio lighting, perfect framing, vibrant colors
  7-8  = Good — above average home creator content, clear and well-lit
  4-6  = Average — typical smartphone video, somewhat ok but not standout
  1-3  = Poor — dark, blurry, heavily backlit, shaky, or visually boring

Key factors for Indian Reels scoring:
  - Face visibility is crucial (Indian audiences connect with creator-on-camera content)
  - Bright natural lighting or ring-light quality scores high
  - Vibrant traditional clothing, colorful backgrounds score higher on color_vibrancy
  - On-screen text/captions are important (many watch without sound)

Return ONLY the JSON object. No explanation. No markdown fences. No code block."""

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":      "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=250,
        )

        raw   = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "visual_energy":    float(data.get("visual_energy",    5.0)),
                "lighting_quality": float(data.get("lighting_quality", 5.0)),
                "composition":      float(data.get("composition",      5.0)),
                "color_vibrancy":   float(data.get("color_vibrancy",   5.0)),
                "text_overlay":     float(data.get("text_overlay",     0)),
                "close_up_face":    float(data.get("close_up_face",    0)),
                "scene_description": str(data.get("scene_description", "")),
            }

    except Exception as e:
        logger.error(f"Frame analysis failed for {frame_path.name}: {e}")

    return _default_frame_scores()


def _default_frame_scores() -> dict:
    return {
        "visual_energy":    5.0,
        "lighting_quality": 5.0,
        "composition":      5.0,
        "color_vibrancy":   5.0,
        "text_overlay":     0.0,
        "close_up_face":    0.0,
        "scene_description": "Frame could not be analysed.",
    }


# ═══════════════════════════════════════════════════════════════════
#  BATCH FRAME ANALYSIS  (main entry point)
# ═══════════════════════════════════════════════════════════════════

def analyze_frames(frame_paths: list[Path]) -> dict:
    """
    Analyse all key frames and return averaged visual scores + timeline.

    Returns:
        {
          "visual_energy":     float,
          "lighting_quality":  float,
          "composition":       float,
          "color_vibrancy":    float,
          "text_overlay":      float,
          "close_up_face":     float,
          "scene_analysis":    list[dict]   ← per-frame data
        }
    """
    if not frame_paths:
        logger.warning("No frames to analyse.")
        return _default_visual_scores()

    # Use only 4 key frames: first, ~33%, ~66%, last
    # 4 frames give nearly the same accuracy as 12 but 3x faster + 3x cheaper
    n = len(frame_paths)
    if n <= 4:
        frames_to_use = frame_paths
    else:
        indices = [0, n // 3, (2 * n) // 3, n - 1]
        frames_to_use = [frame_paths[i] for i in indices]
    logger.info(f"Analysing {len(frames_to_use)} key frames with Groq Vision (fast mode)...")

    # Pre-warm Groq client before spawning threads (avoids lazy-init race)
    _get_client()

    # Fire all frame API calls simultaneously — they are independent
    per_frame_results = []
    with ThreadPoolExecutor(max_workers=len(frames_to_use)) as executor:
        future_to_idx = {
            executor.submit(analyze_frame, fp): (i, fp)
            for i, fp in enumerate(frames_to_use)
        }
        for future in as_completed(future_to_idx):
            i, fp = future_to_idx[future]
            result = future.result()
            result["frame_index"] = i
            result["frame_name"]  = fp.name
            per_frame_results.append(result)

    # Restore original frame order (as_completed returns in completion order)
    per_frame_results.sort(key=lambda r: r["frame_index"])
    for i, result in enumerate(per_frame_results):
        logger.info(f"  Frame {i+1}/{len(frames_to_use)}: energy={result['visual_energy']}")

    if not per_frame_results:
        return _default_visual_scores()

    # Average numeric scores across all frames
    numeric_keys = ["visual_energy", "lighting_quality", "composition",
                    "color_vibrancy", "text_overlay", "close_up_face"]
    averages = {}
    for key in numeric_keys:
        vals = [r[key] for r in per_frame_results]
        averages[key] = round(float(np.mean(vals)), 2)

    averages["scene_analysis"] = [
        {
            "frame_number": r["frame_index"],
            "frame_name":   r["frame_name"],
            "description":  r["scene_description"],
            "visual_score": r["visual_energy"],
        }
        for r in per_frame_results
    ]

    logger.info(f"Visual analysis complete. Avg energy: {averages['visual_energy']}")
    return averages


def _default_visual_scores() -> dict:
    return {
        "visual_energy":    5.0,
        "lighting_quality": 5.0,
        "composition":      5.0,
        "color_vibrancy":   5.0,
        "text_overlay":     0.0,
        "close_up_face":    0.0,
        "scene_analysis":   [],
    }
