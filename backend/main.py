"""
InstaViral AI — FastAPI Application Entry Point
─────────────────────────────────────────────────
Run with:  uvicorn backend.main:app --reload --port 8000
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database.mongo    import connect_db, close_db
from .routers.video     import router as video_router
from .routers.analysis  import router as analysis_router
from .routers.benchmarks import router as benchmarks_router

# ── logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="InstaViral AI",
    description="Intelligent Video Analyser and Content Optimiser for Instagram Reels",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit frontend (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── lifecycle ─────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("InstaViral AI starting up...")
    await connect_db()


@app.on_event("shutdown")
async def shutdown():
    await close_db()
    logger.info("InstaViral AI shut down.")


# ── routers ───────────────────────────────────────────────────────
app.include_router(video_router)
app.include_router(analysis_router)
app.include_router(benchmarks_router)


# ── health check ──────────────────────────────────────────────────
@app.get("/", tags=["health"])
async def root():
    return {
        "app":     "InstaViral AI",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
