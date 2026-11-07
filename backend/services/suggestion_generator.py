"""
Suggestion Generator Service  (Multilingual — Marathi / Hindi / English)
─────────────────────────────────────────────────────────────────────────
Two separate Groq API calls to avoid JSON-with-newlines parsing issues:
  Call 1 — Llama 3.3-70B  → full plain-text improvement report
            (language-aware: adjusts advice for Hindi/Marathi/English content)
            (scene-aware: receives holistic visual assessment from LLM Vision)
            (niche-aware: receives viral patterns specific to each content type)
  Call 2 — Llama 3.1-8B   → structured JSON
            (caption + hashtags in detected language, Indian hashtags included)

Context passed to Call 1 (v2 — enriched):
  • Numerical gap analysis (all 14 features)
  • Scene descriptions per frame (from Vision AI per-frame analysis)
  • Holistic visual assessment (LLM saw all 4 frames simultaneously)
  • English translation of transcript (even for Marathi/Hindi videos)
  • Niche-specific viral patterns for Indian content
  • Cultural language context (Marathi / Hindi / regional)
"""

import os
import re
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_groq_client: Groq | None = None

# ── Language display names ──────────────────────────────────────────
LANG_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "en": "English",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "gu": "Gujarati",
    "pa": "Punjabi",
}

# ── Indian niche hashtag packs ──────────────────────────────────────
INDIAN_HASHTAGS = {
    "dance": [
        "#bollywoododance", "#indiandance", "#marathidance",
        "#hindinritya", "#danceindia", "#reelsindia",
    ],
    "comedy": [
        "#indiancomedy", "#hindicomedy", "#marathicomedy",
        "#desifunny", "#indianhumour", "#reelscomedy",
    ],
    "tutorial": [
        "#indiantips", "#hindijugaad", "#jugaad",
        "#desihacks", "#marathitips", "#learningvibes",
    ],
    "motivation": [
        "#indianmotivation", "#hindimotivation", "#marathimotivation",
        "#desimotivation", "#successtips", "#gyaan",
    ],
    "other": [
        "#indianreels", "#reelsindia", "#desivibes",
        "#indiacontent", "#viral", "#trending",
    ],
}

# ── Universal Indian Reels hashtags (always added for Indian content) ─
UNIVERSAL_INDIAN = [
    "#reelsindia", "#indianreels", "#viral", "#trending",
    "#instareels", "#fyp", "#reelsviral",
]

# ── Niche-specific viral patterns for Indian short-form content ──────
# These tell the 70B LLM what actually drives virality in each niche
# for Indian Instagram audiences — not generic advice but real patterns.
NICHE_VIRAL_PATTERNS = {
    "dance": (
        "VIRAL PATTERNS — Indian Dance Reels:\n"
        "• Trending Bollywood or regional audio is the #1 virality driver — "
        "videos using chart-topping songs get 3-5x more reach via the audio discovery feed\n"
        "• Face visible throughout — dance reels with creator's face visible outperform faceless ones significantly\n"
        "• Hook in first 1.5 seconds — start mid-move, not with a static pose or setup\n"
        "• Editing pace 10-15 cuts/min — fast cuts synced to beat drops are expected\n"
        "• Vibrant, single-color backgrounds (yellow, orange, pink walls) perform better than cluttered rooms\n"
        "• Marathi Lavani, Rajasthani folk, or devotional-themed dance gets strong regional reach\n"
        "• CTA at end: 'Ye step try karo aur video banao — tag me!' drives huge comments\n"
        "• Duration sweet spot: 15-30 seconds for pure dance; up to 60s for step-by-step tutorials"
    ),
    "comedy": (
        "VIRAL PATTERNS — Indian Comedy Reels:\n"
        "• The HOOK is everything — first 2 seconds must create a question or surprise; "
        "no setup allowed, start at the punchline moment\n"
        "• Relatable situations go viral — college life, family dynamics, office situations, "
        "Indian middle-class problems (jugaad, mom's threats, train travel)\n"
        "• Hinglish dialogue outperforms pure Hindi or pure English — mix both\n"
        "• Twist/subversion endings — the best Indian comedy reels have an unexpected final line\n"
        "• On-screen text captions increase completion rate — many watch with no audio\n"
        "• 20-45 second duration — shorter punchy clips outperform long skits\n"
        "• CTA: 'Tag karo us dost ko jo aisa hi karta hai' (tag that friend) drives massive shares\n"
        "• Trending audio under the comedy even if it doesn't match — the audio boosts discoverability"
    ),
    "tutorial": (
        "VIRAL PATTERNS — Indian Tutorial Reels:\n"
        "• PROBLEM → SOLUTION format in first 3 seconds: state a specific painful problem, "
        "then promise the fix. 'Ye mistake mat karo!' performs extremely well\n"
        "• On-screen numbered steps (Step 1, Step 2) are critical — "
        "30-40% of viewers watch without sound\n"
        "• Close-up shots of hands/product/demonstration are essential — "
        "wide shots of the person explaining are weaker\n"
        "• Before/After format — show the result first, then how to achieve it\n"
        "• Save rate is the #1 metric for tutorials — always say 'Save this for later' / 'Save karo'\n"
        "• Duration: 30-60 seconds; longer tutorials should be broken into series\n"
        "• Bright, clean background — kitchen counter, white wall, table setup\n"
        "• End with a question to drive comments: 'Tumhara bhi aisa hota hai?' / 'Kya try karoge?'"
    ),
    "motivation": (
        "VIRAL PATTERNS — Indian Motivation Reels:\n"
        "• Powerful opening quote — first text overlay or spoken word must be a striking line "
        "that stops the scroll (philosophical, aspirational, or emotionally resonant)\n"
        "• Cinematic/dramatic background music — motivational reels NEED strong audio\n"
        "• Text overlays of key quotes are essential — the quote displayed = saves + shares\n"
        "• Emotional arc: problem/struggle → insight → hope/action. Flat delivery kills motivation content\n"
        "• Hindi quotes and shayari perform extremely well for Indian audiences — "
        "'Zindagi mein ek cheez pakki hai...' type hooks\n"
        "• 30-60 seconds is the sweet spot; under 20s feels shallow\n"
        "• Call-to-action: 'Save karo, kal zarurat padegi' (Save this, you'll need it tomorrow) "
        "is the highest-save CTA in motivational content\n"
        "• Creator on camera builds trust — talking directly to camera > voiceover only"
    ),
    "other": (
        "VIRAL PATTERNS — Indian Lifestyle/General Reels:\n"
        "• Strong visual hook in first 1.5 seconds — something surprising, beautiful, or satisfying\n"
        "• Trending audio is mandatory for general content — it's the discovery engine\n"
        "• Aesthetic visual quality: well-lit, vibrant colors, clean framing\n"
        "• On-screen text to guide viewers who watch without sound\n"
        "• Authentic personality — Indian audiences connect with real, unpolished creators "
        "more than overly produced content\n"
        "• 15-45 seconds performs best across general content\n"
        "• End with a CTA tied to the content: question, 'follow for more', or challenge"
    ),
}


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not set in .env file!")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_lang_name(lang_code: str) -> str:
    return LANG_NAMES.get(lang_code, lang_code.upper() if lang_code else "Indian")


def _is_indian_language(lang_code: str) -> bool:
    return lang_code in ("hi", "mr", "te", "ta", "kn", "gu", "pa")


# ═══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def generate_suggestions(
    gap_analysis:        list[dict],
    transcript:          str,
    virality_score:      float,
    niche:               str,
    duration_sec:        float,
    has_face:            bool,
    detected_language:   str        = "en",
    english_translation: str        = "",
    scene_analysis:      list[dict] | None = None,
    holistic_assessment: str        = "",
) -> dict:
    """
    Generates improvement report, optimised caption, hashtags, and posting time.

    New in v2:
      english_translation  — Whisper English translation of the transcript
                             (ensures LLM understands Marathi/Hindi content)
      scene_analysis       — per-frame descriptions from Vision AI
                             (tells LLM what is actually shown in each frame)
      holistic_assessment  — qualitative assessment from the multi-frame LLM call
                             (LLM saw ALL frames simultaneously with gap context)

    Returns:
        { suggestions, caption, hashtags, posting_time }
    """
    critical  = [g for g in gap_analysis if g["status"] == "critical"]
    weak      = [g for g in gap_analysis if g["status"] == "weak"]
    strong    = [g for g in gap_analysis if g["status"] == "strong"]
    gap_text  = _format_gaps(gap_analysis)
    lang_name = _get_lang_name(detected_language)

    # ── Build scene descriptions string ──────────────────────────
    if scene_analysis:
        scene_lines = [
            f"  Frame {s.get('frame_number', i+1)} "
            f"(visual score {s.get('visual_score', '?')}/10): "
            f"{s.get('description', 'No description.')}"
            for i, s in enumerate(scene_analysis)
        ]
        scene_text = "\n".join(scene_lines)
    else:
        scene_text = "  Scene descriptions not available."

    # ── Build translation note ─────────────────────────────────
    if english_translation and english_translation.strip() and \
       english_translation.strip() != transcript.strip():
        translation_note = (
            f"ENGLISH TRANSLATION OF TRANSCRIPT: "
            f"{english_translation[:500]}\n"
        )
    else:
        translation_note = ""

    # ── Build niche patterns ───────────────────────────────────
    niche_key     = niche.lower() if niche.lower() in NICHE_VIRAL_PATTERNS else "other"
    niche_pattern = NICHE_VIRAL_PATTERNS[niche_key]

    # ── Assemble full context ──────────────────────────────────
    context = (
        f"NICHE: {niche} | VIRAL SCORE: {virality_score:.1f}% | "
        f"DURATION: {duration_sec:.0f}s | FACE ON CAMERA: {'Yes' if has_face else 'No'} | "
        f"CONTENT LANGUAGE: {lang_name}\n\n"
        f"TRANSCRIPT (original {lang_name}):\n"
        f"  {transcript[:500] if transcript else 'No speech detected'}\n"
        + (translation_note + "\n" if translation_note else "\n")
        + f"SCENE-BY-SCENE ANALYSIS (Vision AI per-frame):\n{scene_text}\n\n"
        + (f"HOLISTIC VISUAL ASSESSMENT (AI viewed all 4 frames together):\n"
           f"{holistic_assessment}\n\n"
           if holistic_assessment else "")
        + f"FEATURE GAP ANALYSIS (your score vs viral average, 0-1 scale):\n"
        f"{gap_text}\n\n"
        f"CRITICAL ISSUES:  {[g['label'] for g in critical] or 'None'}\n"
        f"WEAK AREAS:       {[g['label'] for g in weak] or 'None'}\n"
        f"STRENGTHS:        {[g['label'] for g in strong] or 'None'}\n\n"
        f"{niche_pattern}"
    )

    try:
        client      = _get_client()
        suggestions = _call_suggestions(client, context, niche, detected_language)
        structured  = _call_structured(client, context, niche, detected_language)

        # Merge LLM hashtags with curated Indian hashtag packs
        llm_hashtags   = structured.get("hashtags", [])
        final_hashtags = _build_hashtags(llm_hashtags, niche, detected_language)

        return {
            "suggestions":  suggestions,
            "caption":      structured.get("caption", ""),
            "hashtags":     final_hashtags,
            "posting_time": structured.get("posting_time",
                                           "Tuesday & Thursday, 7-9 PM IST"),
        }
    except Exception as e:
        logger.error(f"generate_suggestions failed: {e}")
        return _default_suggestions(niche, detected_language)


# ═══════════════════════════════════════════════════════════════════
#  CALL 1 — Full text report  (language-aware + scene-aware)
# ═══════════════════════════════════════════════════════════════════

def _call_suggestions(
    client:    Groq,
    context:   str,
    niche:     str,
    lang_code: str,
) -> str:
    lang_name = _get_lang_name(lang_code)
    is_indian = _is_indian_language(lang_code)

    # ── Language-specific cultural context ────────────────────
    if lang_code == "mr":
        lang_context = (
            "CONTENT LANGUAGE CONTEXT — MARATHI:\n"
            "- Marathi audience responds strongly to authenticity, regional pride, and local culture\n"
            "- Most effective Marathi hooks: 'Hey bagha!' (Hey look!), 'Tumhala mahit ahe ka?' "
            "(Did you know?), 'He ekda baghaच' (Watch this once)\n"
            "- Strong Marathi CTAs: 'follow kara' (follow us), 'share kara' (share this), "
            "'like kara' (like it), 'comment madhe sanga' (tell us in comments)\n"
            "- Bollywood music AND Marathi folk music (Lavani, Powada, Natya Sangeet) both work well\n"
            "- Mixing Marathi with English (Marathlish) in captions feels natural and relatable\n"
            "- Strong regional identity markers: Maharashtra pride, Pune/Mumbai references, "
            "Ganpati, traditional elements perform well\n"
            "- Best posting times for Marathi audience: 8-10 PM IST (after dinner), Sunday 2-4 PM"
        )
    elif lang_code == "hi":
        lang_context = (
            "CONTENT LANGUAGE CONTEXT — HINDI:\n"
            "- Hindi audience is the largest on Indian Instagram — massive reach potential\n"
            "- Most effective Hindi hooks: 'Yeh video end tak dekho!' (Watch till the very end!), "
            "'Kya aap jaante hain yeh cheez?' (Did you know this?), "
            "'Galti mat karna yeh kaam mein' (Don't make this mistake)\n"
            "- Strong Hindi CTAs: 'follow karo yaar' (follow, buddy), 'like karo agar agree ho' "
            "(like if you agree), 'tag karo us dost ko' (tag that friend), "
            "'comment mein batao' (tell us in comments)\n"
            "- Hinglish (Hindi + English mix) captions ALWAYS outperform pure Hindi\n"
            "- Trending Bollywood audio — even if it doesn't perfectly match content — "
            "dramatically increases discoverability via the audio page\n"
            "- Best posting times: 8-11 PM IST weekdays, 12-2 PM Saturday/Sunday"
        )
    elif is_indian:
        lang_context = (
            f"CONTENT LANGUAGE CONTEXT — {lang_name.upper()}:\n"
            f"- {lang_name} audience responds to regional authenticity and cultural references\n"
            "- Mix regional language with English for wider reach (Hinglish/regional-lish style)\n"
            "- Use regional-language CTAs for better connection with local audience\n"
            "- Regional + Hindi trending audio both work for discoverability\n"
            "- Best posting times: 8-10 PM IST"
        )
    else:
        lang_context = (
            "CONTENT LANGUAGE CONTEXT — ENGLISH:\n"
            "- English content reaches pan-India educated urban audience\n"
            "- Mix of English + Hindi slang (Hinglish) often performs better than pure English\n"
            "- Trending international audio works well alongside Bollywood crossover tracks"
        )

    prompt = f"""You are a top Instagram growth expert who has helped hundreds of Indian creators go viral. You have access to BOTH the numerical analysis scores AND descriptions of what the video actually looks like.

Your job: give brutally honest, hyper-specific advice that THIS creator can act on TODAY — not generic tips.

━━━ VIDEO ANALYSIS ━━━
{context}

━━━ CULTURAL CONTEXT ━━━
{lang_context}

Write a detailed improvement report with EXACTLY these sections:

## About This Video
In 2-3 sentences: what type of content is this, who is the creator, and what do they appear to be doing? (Use the Scene Analysis and Holistic Assessment above — prove you understand their specific content, not a generic creator.)

## Critical Issues — Fix Before You Post
For EACH critical issue listed above:
- State what the data shows (gap from viral average)
- Reference what was actually SEEN in the frames (use Holistic Assessment if available)
- Give ONE very specific fix with a {lang_name} content example
  (e.g. NOT "improve lighting" — instead "Frame 1 shows you're standing with a window behind you causing backlighting — turn around so the window faces your face")

## Quick Wins — Apply Today
For each weak area: one specific tip with an example that makes sense for {lang_name} {niche} creators.

## What You're Doing Right ✓
Highlight genuine strengths with specific encouragement. Don't be generic — reference what was actually seen.

## Your Top 3 Action Items
The 3 highest-impact changes to make RIGHT NOW, ranked by potential score increase:
1. [Most impactful] ...
2. ...
3. ...

RULES:
- Reference what you actually see in frames/scene descriptions whenever possible
- Give Indian-specific, {lang_name}-specific advice — not global generic tips
- Minimum 350 words. Plain text only. No JSON, no code blocks, no markdown tables.
- Be direct and encouraging — like a coach who believes in the creator"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Suggestions text call failed: {e}")
        return _default_suggestions(niche, lang_code)["suggestions"]


# ═══════════════════════════════════════════════════════════════════
#  CALL 2 — Structured data  (caption + hashtags in right language)
# ═══════════════════════════════════════════════════════════════════

def _call_structured(
    client:    Groq,
    context:   str,
    niche:     str,
    lang_code: str,
) -> dict:
    lang_name = _get_lang_name(lang_code)

    # Caption instruction changes based on language
    if lang_code == "mr":
        caption_instruction = (
            "Write a SHORT caption (under 80 words) in Marathlish style "
            "(natural mix of Marathi and English). "
            "Start with a Marathi attention-grabber like 'Hey bagha!', 'Ekdum mast!', "
            "or 'He try kela ka?'. End with a clear CTA in Marathi or English."
        )
        hashtag_instruction = (
            "Include relevant Marathi hashtags like #marathicontent #marathireels "
            "#maharashtra #puneri and niche-specific tags for Marathi audience."
        )
    elif lang_code == "hi":
        caption_instruction = (
            "Write a SHORT caption (under 80 words) in Hinglish style "
            "(natural mix of romanised Hindi and English — e.g. 'Yaar, yeh try karo!'). "
            "Start with a Hindi hook that stops the scroll. "
            "End with a CTA in Hindi or English."
        )
        hashtag_instruction = (
            "Include relevant Hindi hashtags like #hindireels #hindicontent "
            "#india #bollywood and niche-specific Indian tags."
        )
    else:
        caption_instruction = (
            "Write a SHORT caption (under 80 words) in English. "
            "Start with a strong attention-grabbing hook. End with a clear CTA."
        )
        hashtag_instruction = "Include relevant English niche hashtags that target Indian audiences."

    prompt = f"""Instagram content expert specialising in Indian {lang_name} reels. Based on this {niche} reel analysis:
{context[:800]}

{caption_instruction}
{hashtag_instruction}

Reply with ONLY this JSON (no markdown fences, no extra text, no explanation):
{{"caption":"your caption here","hashtags":["#tag1","#tag2","#tag3","#tag4","#tag5","#tag6","#tag7"],"posting_time":"best day and time for Indian audience in IST"}}"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        raw   = resp.choices[0].message.content.strip()
        raw   = re.sub(r'```(?:json)?|```', '', raw).strip()
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            data     = json.loads(match.group())
            raw_tags = data.get("hashtags", [])
            return {
                "caption":      str(data.get("caption", "")),
                "hashtags":     [h if h.startswith("#") else f"#{h}" for h in raw_tags][:7],
                "posting_time": str(data.get("posting_time",
                                             "Tuesday & Thursday, 8-10 PM IST")),
            }
    except Exception as e:
        logger.error(f"Structured data call failed: {e}")

    return {
        "caption":      _default_caption(niche, lang_code),
        "hashtags":     [],
        "posting_time": "Tuesday & Thursday, 8-10 PM IST",
    }


# ═══════════════════════════════════════════════════════════════════
#  HASHTAG BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_hashtags(
    llm_tags:  list[str],
    niche:     str,
    lang_code: str,
) -> list[str]:
    """
    Merge LLM-generated tags + curated Indian niche tags + universal tags.
    Deduplicate, normalise, and return max 15 hashtags.
    """
    combined = list(llm_tags)

    # Add Indian niche pack only for Indian-language content
    if _is_indian_language(lang_code):
        niche_key = niche.lower() if niche.lower() in INDIAN_HASHTAGS else "other"
        combined += INDIAN_HASHTAGS[niche_key]
        combined += UNIVERSAL_INDIAN

        # Add language-specific tag
        if lang_code == "mr":
            combined += ["#marathicontent", "#marathireels", "#maharashtra"]
        elif lang_code == "hi":
            combined += ["#hindireels", "#hindicontent", "#india"]
    else:
        # English content — standard tags
        combined += ["#reels", "#viral", "#trending", "#instareels",
                     "#fyp", "#reelsviral", "#explore"]

    # Normalise: ensure # prefix, lowercase, strip spaces
    seen   = set()
    result = []
    for tag in combined:
        tag = tag.strip().lower()
        if not tag.startswith("#"):
            tag = f"#{tag}"
        tag = tag.replace(" ", "")
        if tag not in seen and len(tag) > 1:
            seen.add(tag)
            result.append(tag)

    return result[:15]  # Instagram allows up to 30 but 10-15 performs best


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _format_gaps(gap_analysis: list[dict]) -> str:
    lines = []
    for g in gap_analysis:
        sym = {
            "critical": "[!!]", "weak": "[!]",
            "ok": "[ok]", "strong": "[+]", "na": "[--]",
        }.get(g["status"], "")
        user_str  = f"{g['user_score']:.2f}" if g["status"] != "na" else "N/A"
        gap_str   = f"{g['gap']:+.2f}"      if g["status"] != "na" else "N/A"
        lines.append(
            f"  {sym} {g['label']:<35} "
            f"you={user_str}  viral={g['viral_avg']:.2f}  gap={gap_str}"
        )
    return "\n".join(lines)


def _default_caption(niche: str, lang_code: str) -> str:
    if lang_code == "mr":
        return f"Ekdum mast {niche} content! Follow kara anikhi asya videos sathi! 🔥"
    elif lang_code == "hi":
        return f"Kamaal ka {niche} content! Follow karo aur aisi videos ke liye! 🔥"
    else:
        return f"Amazing {niche} content! Follow for more. 🔥"


def _default_suggestions(niche: str, lang_code: str = "en") -> dict:
    lang_name = _get_lang_name(lang_code)
    return {
        "suggestions": (
            f"## General Tips for Viral {niche.title()} Reels ({lang_name} Content)\n\n"
            "**Hook**: Start with something surprising, visual, or exciting in the first 2 seconds. "
            "Don't begin with a static intro — start mid-action.\n\n"
            "**Editing Pace**: Keep cuts fast — aim for 8-15 cuts per minute. "
            "Boring sections lose viewers instantly.\n\n"
            "**Audio**: Use trending audio or clear background music. "
            + ("For Indian content, trending Bollywood or regional music increases "
               "discoverability via the audio page dramatically.\n\n"
               if _is_indian_language(lang_code) else "\n\n")
            + "**Lighting**: Bright, even, natural light makes a huge difference. "
            "Face a window — don't stand with a window behind you.\n\n"
            "**On-Screen Text/Captions**: Add text captions — "
            "60% of viewers watch reels without sound.\n\n"
            "**CTA**: Always end with a clear call-to-action: "
            + ("'follow karo', 'share karo', 'comment mein batao'.\n\n"
               if lang_code in ("hi", "mr") else "'follow', 'save', 'share'.\n\n")
            + "**Face**: Show your face on camera — creators on camera "
            "consistently outperform voiceover-only content."
        ),
        "caption":      _default_caption(niche, lang_code),
        "hashtags":     _build_hashtags([], niche, lang_code),
        "posting_time": "Tuesday & Thursday, 8-10 PM IST",
    }
