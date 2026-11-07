"""
Audio Analyzer Service  (Multilingual — Marathi / Hindi / English)
──────────────────────────────────────────────────────────────────
Two-pass Whisper strategy:
  Pass 1  whisper-large-v3  task=transcribe  (no language lock)
          → original transcript in native script for display
          → also captures detected_language code

  Pass 2  whisper-large-v3  task=translate
          → English translation for downstream LLM scoring
          (Groq Whisper translates any language → English in one call)

Why two passes?
  The user sees the original Marathi/Hindi transcript on the dashboard.
  The LLM scores hook/sentiment/CTA on the English translation so it
  understands the content correctly regardless of input language.

CTA detection covers Hindi + Marathi + English phrases.
"""

import os
import re
import json
import logging
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_groq_client: Groq | None = None

# whisper-large-v3 is more accurate than turbo for Indian languages
WHISPER_MODEL = "whisper-large-v3"

# Groq returns full language names like "English", "Hindi", "Marathi"
# Normalise them to ISO-639-1 codes used throughout the app
_LANG_NAME_TO_CODE = {
    "english":    "en",
    "hindi":      "hi",
    "marathi":    "mr",
    "telugu":     "te",
    "tamil":      "ta",
    "kannada":    "kn",
    "gujarati":   "gu",
    "punjabi":    "pa",
    "bengali":    "bn",
    "urdu":       "ur",
    "odia":       "or",
    "malayalam":  "ml",
    "assamese":   "as",
    "nepali":     "ne",
}

# ── Indian-language CTA keywords (used in fallback rule-based check) ─
# These supplement the LLM — catches cases where the translated text
# still keeps Romanised words (very common in Hinglish/Marathlish).
INDIAN_CTA_KEYWORDS = [
    # English (standard)
    "follow", "like", "share", "comment", "subscribe",
    "save", "link in bio", "click", "watch", "tag",
    # Romanised Hindi
    "follow karo", "like karo", "share karo", "comment karo",
    "subscribe karo", "save kar", "notification", "bell dabao",
    "dekho", "batao", "jaroor", "zaroor",
    # Romanised Marathi
    "follow kara", "like kara", "share kara", "comment kara",
    "subscribe kara", "save kara", "bagha", "sanga", "nakki",
    # Common hybrid
    "tap karo", "tap kara", "click karo", "swipe",
]


def _normalise_lang(raw: str) -> str:
    """
    Convert Groq's full language name (e.g. 'Hindi', 'English') to ISO-639-1
    code (e.g. 'hi', 'en').  Falls back to lowercase of raw string.
    """
    if not raw:
        return "en"
    return _LANG_NAME_TO_CODE.get(raw.lower(), raw.lower()[:2])


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not set in .env file!")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ═══════════════════════════════════════════════════════════════════
#  PASS 1 — Transcribe in original language (for display)
# ═══════════════════════════════════════════════════════════════════

def transcribe_audio(wav_path: Path) -> dict:
    """
    Transcribe audio in its original language.
    Returns { "text": str, "language": str }
    Language is the ISO-639-1 code detected by Whisper (e.g. "hi", "mr", "en").
    """
    if wav_path is None or not wav_path.exists():
        logger.warning("No audio file — returning empty transcript.")
        return {"text": "", "language": "en"}

    try:
        client = _get_client()
        with open(wav_path, "rb") as f:
            # verbose_json gives us the detected language code
            response = client.audio.transcriptions.create(
                file=f,
                model=WHISPER_MODEL,
                response_format="verbose_json",
                # NO language parameter — let Whisper auto-detect
            )
        text        = getattr(response, "text",     "") or ""
        raw_lang    = getattr(response, "language", "en") or "en"
        # Groq returns full name e.g. "Hindi" — normalise to ISO code "hi"
        language    = _normalise_lang(raw_lang)
        logger.info(f"Transcription: lang={raw_lang}->{language}, length={len(text)} chars")
        return {"text": text.strip(), "language": language}

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"text": "", "language": "en"}


# ═══════════════════════════════════════════════════════════════════
#  PASS 2 — Translate to English (for LLM scoring)
# ═══════════════════════════════════════════════════════════════════

def translate_to_english(wav_path: Path) -> str:
    """
    Translate any language audio → English text using Groq Whisper.
    Used to feed English content to the LLM scorer regardless of
    what language the original video is in.

    Returns English translation string (empty string on failure).
    """
    if wav_path is None or not wav_path.exists():
        return ""

    try:
        client = _get_client()
        with open(wav_path, "rb") as f:
            response = client.audio.translations.create(
                file=f,
                model=WHISPER_MODEL,
                response_format="text",
            )
        translation = str(response).strip()
        logger.info(f"Translation to English: {len(translation)} chars")
        return translation

    except Exception as e:
        logger.error(f"Audio translation failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════
#  RULE-BASED CTA FALLBACK (language-aware)
# ═══════════════════════════════════════════════════════════════════

def _rule_based_cta(text: str) -> float:
    """
    Quick keyword scan for CTAs in Hindi, Marathi, and English.
    Returns 1.0 if any CTA keyword found, else 0.0.
    Used as a fallback / boost when LLM misses Indian CTAs.
    """
    if not text:
        return 0.0
    lower = text.lower()
    for kw in INDIAN_CTA_KEYWORDS:
        if kw in lower:
            return 1.0
    return 0.0


# ═══════════════════════════════════════════════════════════════════
#  TRANSCRIPT SCORING  (Llama 3.3-70B, language-aware)
# ═══════════════════════════════════════════════════════════════════

def score_transcript(
    english_text: str,
    original_text: str,
    detected_language: str,
) -> dict:
    """
    Score the transcript using Groq Llama 3.3-70B.

    Inputs:
      english_text      — Whisper English translation (reliable for LLM)
      original_text     — Native-language transcript (used for rule-based CTA)
      detected_language — ISO code like "hi", "mr", "en"

    Returns:
        {
          "hook_strength":         float,   # 0–10
          "transcript_sentiment":  float,   # -1 to 1
          "cta_present":           float,   # 0 or 1
          "hook_text":             str,
          "sentiment_reason":      str,
        }
    """
    # Use English translation for LLM; fall back to original if translation empty
    scoring_text = english_text if english_text.strip() else original_text
    if not scoring_text or len(scoring_text.strip()) < 5:
        return _default_audio_scores()

    lang_note = {
        "hi": "The original video is in Hindi (हिंदी). The transcript below is an English translation.",
        "mr": "The original video is in Marathi (मराठी). The transcript below is an English translation.",
        "en": "The video is in English.",
    }.get(detected_language,
          f"The original video is in language code '{detected_language}'. "
          "The transcript below is an English translation.")

    prompt = f"""You are an expert Instagram Reels content analyst specialising in Indian content (Hindi, Marathi, English).

{lang_note}

Analyse this transcript and return ONLY a valid JSON object with these exact keys:

{{
  "hook_strength": <number 0-10 — how compelling are the first 1-2 sentences as a scroll-stopping hook?>,
  "transcript_sentiment": <number -1.0 to 1.0 — overall emotional tone: 1.0=very positive/energetic/exciting, 0=neutral/informational, -1.0=negative/sad/angry>,
  "cta_present": <1 if ANY call-to-action is present anywhere in the transcript, else 0>,
  "hook_text": "<copy the first 15 words of the transcript exactly>",
  "sentiment_reason": "<one sentence explaining why you gave this sentiment score>"
}}

Hook scoring guide (for Indian Reels):
  10 = Instantly stops scrolling — bold claim, shocking reveal, strong promise, direct challenge
       Examples: "This one mistake is costing you views", "Nobody told you this about...", "Watch this before you post"
  8-9 = Very good hook — engaging question, surprising statement, creates curiosity
       Examples: "Did you know this about your phone?", "The #1 reason reels don't go viral is..."
  6-7 = Decent hook — somewhat interesting opening, not generic but not outstanding
  4-5 = Average — neutral opener like "Hey guys, welcome back", "Today I'm going to show you"
  1-3 = Weak hook — no opener, starts mid-sentence, boring intro with no energy

Sentiment guide:
  +0.8 to +1.0 = Highly energetic, excited, celebratory, motivational
  +0.4 to +0.7 = Positive, happy, encouraging, upbeat
  0.0 to +0.3  = Neutral or mildly positive (most informational/tutorial content)
  -0.3 to -0.1 = Slightly negative, serious, concerned
  -1.0 to -0.4 = Strongly negative, angry, sad, alarming

CTA detection — mark as 1 if ANY of these appear (including English translations of Indian CTAs):
  "Follow for more", "Like this", "Share with friends", "Comment below", "Save this",
  "Subscribe", "Tag someone", "Watch till the end", "Don't skip", "Click the link",
  "Try this", "Do this now", "Let me know", "Tell me in comments"

TRANSCRIPT (English translation):
\"\"\"{scoring_text[:1500]}\"\"\"

Return ONLY the JSON object. No explanation. No markdown fences."""

    try:
        client   = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fast + cheap for structured extraction
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        raw   = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            llm_cta = float(data.get("cta_present", 0))
            # Boost with rule-based check on BOTH original and translated text
            rule_cta = max(
                _rule_based_cta(original_text),
                _rule_based_cta(english_text),
            )
            final_cta = max(llm_cta, rule_cta)  # take the higher confidence
            return {
                "hook_strength":        float(data.get("hook_strength",        5.0)),
                "transcript_sentiment": float(data.get("transcript_sentiment", 0.0)),
                "cta_present":          final_cta,
                "hook_text":            str(data.get("hook_text",   "")),
                "sentiment_reason":     str(data.get("sentiment_reason", "")),
            }
    except Exception as e:
        logger.error(f"Transcript scoring failed: {e}")

    return _default_audio_scores()


def _default_audio_scores() -> dict:
    return {
        "hook_strength":        5.0,
        "transcript_sentiment": 0.0,
        "cta_present":          0.0,
        "hook_text":            "",
        "sentiment_reason":     "Could not analyse transcript.",
    }


# ═══════════════════════════════════════════════════════════════════
#  COMBINED  (main entry point)
# ═══════════════════════════════════════════════════════════════════

def analyze_audio(wav_path: Path) -> dict:
    """
    Full multilingual audio analysis pipeline.

    Returns:
        transcript          — original language text (shown on dashboard)
        english_translation — English version (used internally for scoring)
        detected_language   — ISO-639-1 code: "hi" / "mr" / "en" / etc.
        hook_strength, transcript_sentiment, cta_present, hook_text,
        sentiment_reason
    """
    # Pass 1 — original transcript + language detection
    transcription   = transcribe_audio(wav_path)
    original_text   = transcription["text"]
    lang_code       = transcription["language"]

    # Pass 2 — English translation (only if not already English)
    if lang_code == "en" or not original_text:
        english_text = original_text          # already English, no extra API call
        logger.info("Audio is English — skipping translation pass.")
    else:
        english_text = translate_to_english(wav_path)
        if not english_text:
            english_text = original_text      # fallback: score original
            logger.warning("Translation empty — scoring original text.")

    # Score using English text
    scores = score_transcript(english_text, original_text, lang_code)

    return {
        **scores,
        "transcript":          original_text,
        "english_translation": english_text,
        "detected_language":   lang_code,
    }
