"""
Benchmarks Router
──────────────────
GET  /api/benchmarks/{category}   — list benchmarks for a niche
GET  /api/benchmarks/count/{cat}  — count benchmarks per category
"""

import logging
from fastapi import APIRouter, HTTPException
from ..database.crud import get_benchmarks, get_all_benchmarks, count_benchmarks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])

VALID_CATEGORIES = {"Dance", "Comedy", "Tutorial", "Motivation", "Other"}


@router.get("/{category}")
async def list_benchmarks(category: str):
    """List all pre-seeded viral benchmarks for a category."""
    docs = await get_benchmarks(category)
    return {
        "category": category,
        "count":    len(docs),
        "benchmarks": [
            {
                "video_url":   d.get("video_url", ""),
                "analyzed_at": str(d.get("analyzed_at", "")),
            }
            for d in docs
        ],
    }


@router.get("/count/{category}")
async def benchmark_count(category: str):
    count = await count_benchmarks(category)
    return {"category": category, "count": count}


@router.get("/")
async def all_benchmarks():
    docs = await get_all_benchmarks()
    summary = {}
    for d in docs:
        cat = d.get("category", "Unknown")
        summary[cat] = summary.get(cat, 0) + 1
    return {"total": len(docs), "by_category": summary}
