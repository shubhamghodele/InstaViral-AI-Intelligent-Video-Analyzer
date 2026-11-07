"""
Analysis Router
────────────────
POST /api/analyze          — trigger full analysis pipeline
GET  /api/results/{id}     — get analysis results
"""

import logging
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..models.schemas  import AnalyzeRequest
from ..database.crud   import create_job, get_job, get_result
from ..services.analyzer import run_full_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])

UPLOAD_DIR = Path("./uploads")


@router.post("/analyze")
async def start_analysis(
    request:          AnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the full multimodal analysis pipeline.
    Runs in background — poll /api/results/{video_id} for completion.
    """
    video_id = request.video_id

    # Find the uploaded video file
    video_path = _find_video(video_id)
    if not video_path:
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found for id '{video_id}'. Upload first."
        )

    # Check if job already exists
    existing = await get_job(video_id)
    if existing and existing["status"] in ("processing", "done"):
        return {
            "message": f"Job already {existing['status']}.",
            "video_id": video_id,
            "status":   existing["status"],
        }

    # Create job record in MongoDB
    await create_job(
        video_id=video_id,
        niche=request.niche,
        viral_urls=request.viral_urls,
    )

    # Launch analysis in background
    background_tasks.add_task(
        run_full_analysis,
        video_id=video_id,
        video_path=video_path,
        niche=request.niche,
        viral_urls=request.viral_urls,
    )

    logger.info(f"Analysis started for {video_id} | niche={request.niche}")

    return {
        "message":  "Analysis started. Poll /api/results/{video_id} for status.",
        "video_id": video_id,
        "status":   "processing",
    }


@router.get("/results/{video_id}")
async def get_results(video_id: str):
    """
    Get analysis results.
    Returns job status if still processing, full results if done.
    """
    job = await get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job found for this video_id.")

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {job.get('error', 'Unknown error')}"
        )

    if job["status"] in ("pending", "processing"):
        return {"status": job["status"], "message": "Analysis in progress..."}

    # Fetch full results
    result = await get_result(video_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found in database.")

    return result


def _find_video(video_id: str) -> Path | None:
    """Find the uploaded video file by video_id prefix."""
    for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        p = UPLOAD_DIR / f"{video_id}{ext}"
        if p.exists():
            return p
    return None
