"""
All MongoDB read / write operations in one place.
Every function is async (Motor-based).
"""

from datetime import datetime, timezone
from bson import ObjectId
from .mongo import jobs_collection, results_collection, benchmarks_collection


# ── helpers ───────────────────────────────────────────────────────
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _str_id(doc: dict) -> dict:
    """Convert ObjectId _id to string so it can be JSON-serialised."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ═══════════════════════════════════════════════════════════════════
#  ANALYSIS JOBS
# ═══════════════════════════════════════════════════════════════════

async def create_job(video_id: str, niche: str, viral_urls: list[str]) -> dict:
    """Create a new analysis job document."""
    doc = {
        "video_id":   video_id,
        "niche":      niche,
        "viral_urls": viral_urls,
        "status":     "pending",      # pending | processing | done | failed
        "created_at": _now(),
        "updated_at": _now(),
    }
    result = await jobs_collection().insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return doc


async def get_job(video_id: str) -> dict | None:
    doc = await jobs_collection().find_one({"video_id": video_id})
    return _str_id(doc) if doc else None


async def update_job_status(video_id: str, status: str, error: str = "") -> None:
    update = {"status": status, "updated_at": _now()}
    if error:
        update["error"] = error
    await jobs_collection().update_one(
        {"video_id": video_id}, {"$set": update}
    )


# ═══════════════════════════════════════════════════════════════════
#  ANALYSIS RESULTS
# ═══════════════════════════════════════════════════════════════════

async def save_result(video_id: str, result_data: dict) -> None:
    """Insert or replace the analysis result for a video."""
    result_data["video_id"]   = video_id
    result_data["created_at"] = _now()
    await results_collection().replace_one(
        {"video_id": video_id},
        result_data,
        upsert=True,
    )


async def get_result(video_id: str) -> dict | None:
    doc = await results_collection().find_one({"video_id": video_id})
    return _str_id(doc) if doc else None


# ═══════════════════════════════════════════════════════════════════
#  VIRAL BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

async def save_benchmark(
    category: str,
    video_url: str,
    feature_vector: list[float],
    metadata: dict,
) -> None:
    """Insert or update a viral benchmark entry."""
    doc = {
        "category":       category,
        "video_url":      video_url,
        "feature_vector": feature_vector,
        "metadata":       metadata,
        "analyzed_at":    _now(),
    }
    await benchmarks_collection().replace_one(
        {"video_url": video_url}, doc, upsert=True
    )


async def get_benchmarks(category: str) -> list[dict]:
    """Fetch all pre-seeded viral benchmarks for a niche category."""
    # Normalise to title-case so "dance" matches stored "Dance"
    cursor = benchmarks_collection().find({"category": category.title()})
    docs   = await cursor.to_list(length=50)
    return [_str_id(d) for d in docs]


async def get_all_benchmarks() -> list[dict]:
    cursor = benchmarks_collection().find({})
    docs   = await cursor.to_list(length=200)
    return [_str_id(d) for d in docs]


async def count_benchmarks(category: str) -> int:
    return await benchmarks_collection().count_documents({"category": category.title()})
