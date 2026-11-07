"""
Video Processor Service
────────────────────────
Handles all video file operations:
  • Download viral videos via yt-dlp
  • Extract audio as .wav  (via moviepy)
  • Extract frames as JPEG (via OpenCV)
  • Detect key scene-change frames (via PySceneDetect)
  • Compute motion intensity (optical flow)
  • Compute colour vibrancy (HSV histogram)
  • Detect music / beat presence (librosa)
  • Compute video duration
"""

import os
import uuid
import logging
import subprocess
import numpy as np
import cv2

# ── ensure moviepy finds ffmpeg via imageio_ffmpeg ────────────────
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ.setdefault("FFMPEG_BINARY", _ffmpeg_exe)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", _ffmpeg_exe)
except Exception:
    _ffmpeg_exe = "ffmpeg"

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TEMP_DIR   = Path(os.getenv("TEMP_DIR",   "./temp"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  VIDEO DOWNLOAD  (yt-dlp)
# ═══════════════════════════════════════════════════════════════════

def download_video(url: str, save_dir: Path = TEMP_DIR) -> Path:
    """
    Download a video from any URL supported by yt-dlp.
    Returns the local file path.
    """
    import yt_dlp

    out_path = save_dir / "%(id)s.%(ext)s"
    ydl_opts = {
        "outtmpl":         str(out_path),
        "format":          "best[ext=mp4]/best",
        "quiet":           True,
        "no_warnings":     True,
        "noplaylist":      True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    filepath = Path(filename)
    if not filepath.exists():
        # yt-dlp sometimes converts to mp4 — find it
        for f in save_dir.glob(f"{info['id']}.*"):
            filepath = f
            break

    logger.info(f"Downloaded video: {filepath}")
    return filepath


# ═══════════════════════════════════════════════════════════════════
#  AUDIO EXTRACTION  (moviepy)
# ═══════════════════════════════════════════════════════════════════

def extract_audio(video_path: Path) -> Path:
    """
    Extract audio from a video file as a 16-kHz mono WAV.
    Uses imageio_ffmpeg binary directly — no system ffmpeg needed.
    Returns the .wav file path, or None if no audio track.
    """
    wav_path = TEMP_DIR / f"{video_path.stem}_audio.wav"
    if wav_path.exists():
        return wav_path

    try:
        result = subprocess.run(
            [
                _ffmpeg_exe, "-y",
                "-i", str(video_path),
                "-vn",                      # no video
                "-acodec", "pcm_s16le",
                "-ar", "16000",             # 16kHz
                "-ac", "1",                 # mono
                str(wav_path),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not wav_path.exists():
            logger.warning(f"No audio track in {video_path.name} (ffmpeg: {result.stderr[-100:]})")
            return None
        logger.info(f"Audio extracted: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return None

    return wav_path


# ═══════════════════════════════════════════════════════════════════
#  FRAME EXTRACTION  (OpenCV)
# ═══════════════════════════════════════════════════════════════════

def extract_frames(video_path: Path, fps: int = 1) -> list[Path]:
    """
    Extract frames at <fps> frames per second.
    Returns list of JPEG file paths.
    """
    frames_dir = TEMP_DIR / f"{video_path.stem}_frames"
    frames_dir.mkdir(exist_ok=True)

    cap    = cv2.VideoCapture(str(video_path))
    v_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    step   = max(1, int(v_fps / fps))
    paths  = []
    idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            paths.append(out_path)
        idx += 1

    cap.release()
    logger.info(f"Extracted {len(paths)} frames from {video_path.name}")
    return paths


# ═══════════════════════════════════════════════════════════════════
#  KEY FRAME SELECTION  (PySceneDetect)
# ═══════════════════════════════════════════════════════════════════

def detect_key_frames(video_path: Path, threshold: float = 27.0) -> tuple[list[Path], int]:
    """
    Use PySceneDetect ContentDetector to find scene-change frames.
    Falls back to uniform sampling if fewer than 4 scenes detected.

    Returns (key_frame_paths, scene_count).
    scene_count is returned so the caller can compute pacing directly
    without running PySceneDetect a second time.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video      = open_video(str(video_path))
        scene_mgr  = SceneManager()
        scene_mgr.add_detector(ContentDetector(threshold=threshold))
        scene_mgr.detect_scenes(video, show_progress=False)
        scene_list = scene_mgr.get_scene_list()
        scene_count = len(scene_list)

        if scene_count < 4:
            logger.info("Few scenes detected — using uniform frame sampling.")
            return extract_frames(video_path, fps=1)[:12], scene_count

        # Save the first frame of every scene
        frames_dir = TEMP_DIR / f"{video_path.stem}_keyframes"
        frames_dir.mkdir(exist_ok=True)

        cap    = cv2.VideoCapture(str(video_path))
        paths  = []
        for scene in scene_list:
            frame_num = scene[0].frame_num
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                p = frames_dir / f"kf_{frame_num:06d}.jpg"
                cv2.imwrite(str(p), frame)
                paths.append(p)
        cap.release()

        logger.info(f"Key frames selected: {len(paths)}")
        return paths, scene_count

    except Exception as e:
        logger.warning(f"SceneDetect failed ({e}), falling back to uniform sampling.")
        return extract_frames(video_path, fps=1)[:12], 0


# ═══════════════════════════════════════════════════════════════════
#  SCENE PACING  (cuts per minute)
# ═══════════════════════════════════════════════════════════════════

def compute_pacing(video_path: Path) -> float:
    """
    Returns the number of scene cuts per minute (editing pace score).
    Viral short-form videos typically have 8–14 cuts/min.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video     = open_video(str(video_path))
        mgr       = SceneManager()
        mgr.add_detector(ContentDetector(threshold=27.0))
        mgr.detect_scenes(video, show_progress=False)
        scenes    = mgr.get_scene_list()
        duration  = get_duration(video_path)

        if duration > 0:
            return round((len(scenes) / duration) * 60, 2)
        return 0.0
    except Exception as e:
        logger.warning(f"Pacing compute failed: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════════════
#  MOTION INTENSITY  (optical flow)
# ═══════════════════════════════════════════════════════════════════

def compute_motion_intensity(video_path: Path, sample_frames: int = 30) -> float:
    """
    Computes average optical-flow magnitude as a motion proxy.
    Returns a 0-10 normalised score.
    """
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // sample_frames)

    prev_gray = None
    mags      = []

    for i in range(0, min(total, sample_frames * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(float(np.mean(mag)))
        prev_gray = gray

    cap.release()

    if not mags:
        return 0.0
    avg  = np.mean(mags)
    # Normalise: 0 → 0, ≥8 → 10
    score = min(10.0, round((avg / 8.0) * 10, 2))
    return score


# ═══════════════════════════════════════════════════════════════════
#  COLOUR VIBRANCY  (HSV histogram)
# ═══════════════════════════════════════════════════════════════════

def compute_color_vibrancy(frame_paths: list[Path]) -> float:
    """
    Average saturation across key frames → colour vibrancy score 0-10.
    Higher saturation = more vibrant = tends to perform better.
    """
    saturations = []
    for fp in frame_paths[:10]:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat  = hsv[:, :, 1].mean()          # 0-255
        saturations.append(sat)

    if not saturations:
        return 5.0
    avg   = np.mean(saturations)
    score = min(10.0, round((avg / 255.0) * 10, 2))
    return score


# ═══════════════════════════════════════════════════════════════════
#  MUSIC / BEAT DETECTION  (librosa)
# ═══════════════════════════════════════════════════════════════════

def detect_music(wav_path: Path) -> dict:
    """
    Detects if background music is present and estimates BPM.
    Returns { "music_present": bool, "bpm": float }
    """
    if wav_path is None or not wav_path.exists():
        return {"music_present": False, "bpm": 0.0}

    try:
        import librosa
        y, sr   = librosa.load(str(wav_path), sr=None, mono=True, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # If tempo is a numpy array, get the scalar
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)
        music   = tempo > 60.0          # BPM > 60 strongly suggests music
        return {"music_present": music, "bpm": round(tempo, 1)}
    except Exception as e:
        logger.warning(f"Music detection failed: {e}")
        return {"music_present": False, "bpm": 0.0}


# ═══════════════════════════════════════════════════════════════════
#  VIDEO DURATION
# ═══════════════════════════════════════════════════════════════════

def get_duration(video_path: Path) -> float:
    """Returns video duration in seconds."""
    cap      = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    frames   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return round(frames / fps, 2) if fps > 0 else 0.0


def duration_score(seconds: float) -> float:
    """
    Score the video duration.
    Sweet spot for Reels: 15-30s = 10, 30-45s = 8, 45-60s = 6, >60s = 4
    """
    if   15 <= seconds <= 30: return 10.0
    elif 30 <  seconds <= 45: return 8.0
    elif 45 <  seconds <= 60: return 6.0
    elif seconds < 15:        return 7.0
    else:                     return 4.0


# ═══════════════════════════════════════════════════════════════════
#  CLEANUP
# ═══════════════════════════════════════════════════════════════════

def cleanup_temp(video_id: str) -> None:
    """Remove all temp files related to a video_id after analysis."""
    import shutil
    for item in TEMP_DIR.iterdir():
        if video_id in item.name:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    logger.info(f"Cleaned temp files for {video_id}")
