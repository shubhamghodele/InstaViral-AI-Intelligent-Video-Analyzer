"""
Holistic Video Analyzer
─────────────────────────
Passes ALL key video frames SIMULTANEOUSLY to Groq Llama 4 Scout Vision
along with the computed gap analysis numbers for a unified qualitative
assessment.

This bridges the gap between numerical feature scores and specific visual
observations — so the suggestion LLM can say "your frame 1 shows dim
lighting near a dark wall — move 2 steps toward the window" instead of
just "lighting is below viral target".

Why this matters:
  The per-frame vision calls already score individual metrics (lighting=4.2,
  composition=5.1, etc.).  This holistic call gives the 70B suggestion model
  a DESCRIPTIVE understanding of what the video actually looks like, so its
  advice is specific to THIS creator's video, not generic tips.
"""

import os
import base64
import logging
from pathlib import Path
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
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def get_holistic_assessment(
    frame_paths:    list[Path],
    gap_analysis:   list[dict],
    niche:          str,
    virality_score: float,
) -> str:
    """
    Pass all key frames simultaneously to Groq Vision LLM for a qualitative
    assessment that combines what the model SEES with the gap numbers.

    Returns a descriptive paragraph (≤250 words) that is fed into the
    suggestion generator's context so the 70B model gives hyper-specific
    advice rooted in the actual video frames.

    Returns empty string on failure — the rest of the pipeline continues.
    """
    if not frame_paths:
        logger.warning("No frames available for holistic assessment — skipping.")
        return ""

    # Use up to 4 key frames
    frames_to_use = frame_paths[:4]

    # ── Format gap context ────────────────────────────────────────
    critical = [g for g in gap_analysis if g["status"] == "critical"]
    weak     = [g for g in gap_analysis if g["status"] == "weak"]
    strong   = [g for g in gap_analysis if g["status"] == "strong"]

    def _fmt(items: list[dict]) -> str:
        return ", ".join(g["label"] for g in items) if items else "None"

    gap_context = (
        f"Viral Readiness Score: {virality_score:.1f}%\n"
        f"Critical weaknesses (score much below viral videos): {_fmt(critical)}\n"
        f"Weak areas (slightly below viral average): {_fmt(weak)}\n"
        f"Strong points (at or above viral average): {_fmt(strong)}"
    )

    prompt = f"""You are an expert Instagram Reels video coach. I am showing you {len(frames_to_use)} key frames from a creator's {niche} reel — taken at the start, 33%, 66%, and end of the video.

The automated analysis system has flagged these issues:
{gap_context}

Now, looking at ALL the frames together:

WHAT YOU SEE (2-3 sentences): Describe the video content — who is the creator, what are they doing, what does the setting/background look like?

SPECIFIC VISUAL OBSERVATIONS (link to what you actually see):
- Lighting: Describe exactly what you see — bright/dim, natural light from window, artificial, shadows on face, etc.
- Framing & Composition: Is subject centred? Too much empty space? Camera distance — close-up or far away?
- Colors & Background: Are colors vibrant or muted? Is background clean, cluttered, or distracting?
- On-screen text/captions: Visible in any frame? Yes/No?
- Creator's energy: Do they look engaged and energetic, or static and stiff?

FRAME-SPECIFIC FIXES (2-3 very specific tips based on what you actually see):
Reference the actual frames — e.g. "Frame 1 shows harsh backlighting from the window behind you — swap positions so the window faces you" NOT generic advice like "improve lighting".

Keep total response under 250 words. Be direct, specific, and encouraging."""

    try:
        client = _get_client()

        # Build multi-image content block — all frames in ONE API call
        content = []
        for i, fp in enumerate(frames_to_use):
            try:
                b64 = _encode_image(fp)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
                logger.debug(f"  Holistic: added frame {i+1}: {fp.name}")
            except Exception as img_err:
                logger.warning(f"Could not encode frame {fp}: {img_err}")

        if not content:
            logger.warning("No frames could be encoded for holistic assessment.")
            return ""

        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=500,
        )

        assessment = response.choices[0].message.content.strip()
        logger.info(f"Holistic assessment complete: {len(assessment)} chars")
        return assessment

    except Exception as e:
        logger.error(f"Holistic assessment failed: {e}", exc_info=True)
        return ""
