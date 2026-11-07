"""
Main Analysis Orchestrator
───────────────────────────
Ties all services together into a single pipeline.
Called by the FastAPI /analyze endpoint.

Pipeline order:
  1. Download viral videos (yt-dlp)
  2. For each video (user + viral):
       a. Extract audio + frames
       b. Detect key frames (PySceneDetect)
       c. Audio analysis (Groq Whisper + Llama 3.1-8B)
       d. Visual analysis (Groq Llama 4 Scout Vision — 4 frames)
       e. Emotion analysis (OpenCV face detect + EmotiEffLib)
       f. Structure analysis (motion, color, music, pacing)
       g. Build feature vector
  3. Compare user vector vs viral vectors (cosine similarity)
  4. Gap analysis
  5. Holistic visual assessment — all 4 frames + gap numbers → Groq Vision
     (gives suggestion LLM specific "what we see" context)
  6. Generate AI suggestions (Groq Llama 3.3-70B)
     — now receives: scene descriptions, English translation, holistic assessment,
       niche-specific viral patterns
  7. Save to MongoDB
"""

import logging
import asyncio
import concurrent.futures
from pathlib import Path

from .video_processor  import (
    download_video, extract_audio, detect_key_frames,
    compute_motion_intensity,
    compute_color_vibrancy, detect_music, get_duration,
    duration_score, cleanup_temp, TEMP_DIR
)
from .audio_analyzer    import analyze_audio
from .vision_analyzer   import analyze_frames
from .emotion_analyzer  import analyze_emotions
from .feature_extractor import (
    build_feature_vector, compute_virality_score, compute_gap_analysis
)
from .holistic_analyzer    import get_holistic_assessment
from .suggestion_generator import generate_suggestions
from ..database.crud        import (
    update_job_status, save_result, get_benchmarks
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  SINGLE VIDEO ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_single_video(video_path: Path) -> dict:
    """
    Run the full multimodal pipeline on one video file.
    Returns a dict with all extracted features + raw analysis data.

    Pipeline runs in two parallel phases to cut latency:
      Phase 1: audio extraction + frame detection + motion (all independent)
      Phase 2: audio analysis + vision + emotion + music (all independent,
               each using outputs from Phase 1)
    """
    logger.info(f"Analysing: {video_path.name}")

    # Duration is fast — get it first synchronously
    duration = get_duration(video_path)

    # ── Phase 1: independent I/O tasks run in parallel ────────────────
    # extract_audio, detect_key_frames, compute_motion_intensity all only
    # need video_path and have zero dependency on each other.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fut_wav    = ex.submit(extract_audio, video_path)
        fut_frames = ex.submit(detect_key_frames, video_path)
        fut_motion = ex.submit(compute_motion_intensity, video_path)

        wav_path             = fut_wav.result()
        key_frames, scene_count = fut_frames.result()
        motion               = fut_motion.result()

    logger.info(f"  Phase 1 done. Frames: {len(key_frames)}, Motion: {motion}")

    # ── Phase 2: all analysis tasks run in parallel ───────────────────
    # Each task uses only its own Phase-1 output and has no cross-dependency.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        fut_audio   = ex.submit(analyze_audio,          wav_path)
        fut_visual  = ex.submit(analyze_frames,         key_frames)
        fut_emotion = ex.submit(analyze_emotions,       key_frames)
        fut_color   = ex.submit(compute_color_vibrancy, key_frames)
        fut_music   = ex.submit(detect_music,           wav_path)

        audio_result   = fut_audio.result()
        visual_result  = fut_visual.result()
        emotion_result = fut_emotion.result()
        color          = fut_color.result()
        music_data     = fut_music.result()

    logger.info(f"  Audio done. Hook: {audio_result.get('hook_strength')}, "
                f"lang: {audio_result.get('detected_language')}")
    logger.info(f"  Vision done. Energy: {visual_result.get('visual_energy')}")
    logger.info(f"  Emotion done. Has face: {emotion_result.get('has_face')}")

    # ── Pacing: computed from scene_count returned by detect_key_frames ──
    # No second PySceneDetect scan needed.
    cuts_pm   = round((scene_count / duration) * 60, 2) if duration > 0 else 0.0
    dur_score = duration_score(duration)

    structure = {
        "cuts_per_minute":    cuts_pm,
        "motion_intensity":   motion,
        "music_present":      music_data["music_present"],
        "bpm":                music_data["bpm"],
        "color_vibrancy_raw": color,
        "duration_sec":       duration,
        "duration_score":     dur_score,
    }

    # Override color_vibrancy from vision with raw OpenCV value if vision gave default
    if visual_result.get("color_vibrancy", 5.0) == 5.0:
        visual_result["color_vibrancy"] = color

    # ── Build feature vector ──────────────────────────────────────────
    fv = build_feature_vector(audio_result, visual_result, emotion_result, structure)

    return {
        "feature_vector":    fv["vector"],
        "feature_raw":       fv["raw"],
        "feature_labeled":   fv["labeled"],
        "transcript":        audio_result.get("transcript", ""),
        "english_translation":  audio_result.get("english_translation", ""),
        "detected_language":    audio_result.get("detected_language", "en"),
        "hook_text":         audio_result.get("hook_text", ""),
        "emotion_timeline":  emotion_result.get("emotion_timeline", []),
        "scene_analysis":    visual_result.get("scene_analysis", []),
        "has_face":          emotion_result.get("has_face", False),
        "duration_sec":      duration,
        "structure":         structure,
        "audio_scores": {
            k: audio_result.get(k)
            for k in ["hook_strength", "transcript_sentiment", "cta_present"]
        },
        # key_frames returned so holistic_analyzer can use them after gap analysis
        "key_frames": key_frames,
    }


# ═══════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

async def run_full_analysis(
    video_id:   str,
    video_path: Path,
    niche:      str,
    viral_urls: list[str],
) -> dict:
    """
    Main async entry point called by FastAPI.
    Orchestrates the complete pipeline and saves results to MongoDB.
    """
    await update_job_status(video_id, "processing")

    try:
        # ── Step 1: Analyse the user's video ─────────────────────
        # Use asyncio.to_thread so the heavy sync pipeline does NOT
        # block the FastAPI event loop during processing.
        logger.info(f"=== Analysing USER video: {video_path.name} ===")
        user_analysis = await asyncio.to_thread(analyze_single_video, video_path)

        # ── Step 2: Get viral benchmark vectors ───────────────────
        viral_vectors, benchmark_label = await _get_viral_vectors(
            niche, viral_urls, video_id
        )

        # ── Step 3: Virality score + gap analysis ─────────────────
        has_face = user_analysis.get("has_face", False)
        virality_score = compute_virality_score(
            user_analysis["feature_vector"], viral_vectors,
            has_face=has_face,
        )
        gap_analysis = compute_gap_analysis(
            user_analysis["feature_vector"], viral_vectors,
            has_face=has_face,
        )

        logger.info(f"=== Virality Score: {virality_score:.1f}% ===")

        # ── Step 4: Holistic visual assessment ────────────────────
        # Pass ALL key frames + gap numbers to Vision LLM in ONE call.
        # Returns a descriptive paragraph about what the video actually
        # looks like, so the 70B suggestion model can give specific advice.
        key_frames = user_analysis.get("key_frames", [])
        holistic_assessment = ""
        if key_frames:
            logger.info("=== Running holistic visual assessment ===")
            holistic_assessment = get_holistic_assessment(
                frame_paths=key_frames,
                gap_analysis=gap_analysis,
                niche=niche,
                virality_score=virality_score,
            )
        else:
            logger.warning("No key frames returned — skipping holistic assessment.")

        # ── Step 5: Generate AI suggestions ───────────────────────
        logger.info("=== Generating AI suggestions ===")
        suggestions_data = generate_suggestions(
            gap_analysis=gap_analysis,
            transcript=user_analysis["transcript"],
            virality_score=virality_score,
            niche=niche,
            duration_sec=user_analysis["duration_sec"],
            has_face=user_analysis["has_face"],
            detected_language=user_analysis.get("detected_language", "en"),
            # v2 enriched context ─────────────────────────────────
            english_translation=user_analysis.get("english_translation", ""),
            scene_analysis=user_analysis.get("scene_analysis", []),
            holistic_assessment=holistic_assessment,
        )

        # ── Step 6: Assemble final result ─────────────────────────
        result = {
            "video_id":            video_id,
            "niche":               niche,
            "virality_score":      virality_score,
            "feature_vector":      user_analysis["feature_vector"],
            "feature_raw":         user_analysis["feature_raw"],
            "feature_labeled":     user_analysis["feature_labeled"],
            "gap_analysis":        gap_analysis,
            "transcript":          user_analysis["transcript"],
            "english_translation": user_analysis.get("english_translation", ""),
            "detected_language":   user_analysis.get("detected_language", "en"),
            "hook_text":           user_analysis["hook_text"],
            "emotion_timeline":    user_analysis["emotion_timeline"],
            "scene_analysis":      user_analysis["scene_analysis"],
            "holistic_assessment": holistic_assessment,
            "has_face":            user_analysis["has_face"],
            "duration_sec":        user_analysis["duration_sec"],
            "suggestions":         suggestions_data["suggestions"],
            "caption":             suggestions_data["caption"],
            "hashtags":            suggestions_data["hashtags"],
            "posting_time":        suggestions_data["posting_time"],
            "viral_benchmarks_used": len(viral_vectors),
            "benchmark_label":       benchmark_label,
        }

        # ── Step 7: Save to MongoDB ───────────────────────────────
        await save_result(video_id, result)
        await update_job_status(video_id, "done")

        # ── Step 8: Cleanup temp files ────────────────────────────
        cleanup_temp(video_id)

        logger.info(f"=== Analysis complete for {video_id} ===")
        return result

    except Exception as e:
        logger.error(f"Analysis failed for {video_id}: {e}", exc_info=True)
        await update_job_status(video_id, "failed", error=str(e))
        raise


# ═══════════════════════════════════════════════════════════════════
#  VIRAL VECTOR HELPER
# ═══════════════════════════════════════════════════════════════════

async def _get_viral_vectors(
    niche:      str,
    viral_urls: list[str],
    job_id:     str,
) -> tuple[list[list[float]], str]:
    """
    Returns (vectors, benchmark_label) where:
      - vectors is the list of feature vectors to compare against
      - benchmark_label describes what was used (shown on dashboard)

    Priority:
      1. User-provided viral URLs → download + analyse (MOST LEGITIMATE)
         Works best with YouTube Shorts URLs (yt-dlp handles them reliably)
      2. Pre-seeded MongoDB benchmarks → fallback (clearly labelled as estimates)
    """
    user_viral_vectors = []
    db_vectors         = []

    # ── Priority 1: User-provided viral URLs ──────────────────────
    for i, url in enumerate(viral_urls[:3]):          # max 3 viral videos
        try:
            logger.info(f"Downloading reference viral video {i+1}: {url}")
            vpath    = download_video(url, save_dir=TEMP_DIR)
            analysis = analyze_single_video(vpath)
            user_viral_vectors.append(analysis["feature_vector"])
            logger.info(f"  Reference video {i+1} analysed successfully.")
        except Exception as e:
            logger.warning(f"Could not process viral URL '{url}': {e}")

    if user_viral_vectors:
        label = f"{len(user_viral_vectors)} viral reference video(s) you provided"
        logger.info(f"Using {len(user_viral_vectors)} user-provided viral vectors.")
        return user_viral_vectors, label

    # ── Priority 2: Pre-seeded MongoDB benchmarks ─────────────────
    db_benchmarks = await get_benchmarks(niche)
    for b in db_benchmarks:
        if b.get("feature_vector"):
            db_vectors.append(b["feature_vector"])
    logger.info(f"Loaded {len(db_vectors)} seeded benchmarks for niche '{niche}'")

    if db_vectors:
        label = f"industry average estimates ({niche} niche)"
        return db_vectors, label

    # ── Last resort fallback ──────────────────────────────────────
    logger.warning("No benchmarks available — using default midpoint vector.")
    return [[0.7] * 14], "default estimates"
