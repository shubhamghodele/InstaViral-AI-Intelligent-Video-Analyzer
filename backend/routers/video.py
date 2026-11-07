"""
Video Router
─────────────
POST /api/video/upload     — upload the user's video
GET  /api/video/status/{id} — check job status
"""

import uuid
import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..database.crud   import create_job, get_job
from ..models.schemas  import UploadResponse, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/video", tags=["video"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_SIZE_MB        = 100


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload the user's video file.
    Returns a unique video_id used for all subsequent calls.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Generate unique ID + save
    video_id  = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{video_id}{suffix}"

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Validate size
    size_mb = save_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_SIZE_MB} MB"
        )

    logger.info(f"Video uploaded: {video_id} | {file.filename} | {size_mb:.1f} MB")

    return UploadResponse(
        video_id=video_id,
        message="Video uploaded successfully. Call /api/analyze to start analysis.",
        filename=file.filename,
    )


@router.get("/status/{video_id}", response_model=JobStatus)
async def get_status(video_id: str):
    """Check the current status of an analysis job."""
    job = await get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(
        video_id=video_id,
        status=job["status"],
        created_at=job.get("created_at"),
        error=job.get("error"),
    )
