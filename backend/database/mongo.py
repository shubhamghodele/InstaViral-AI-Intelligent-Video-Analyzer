"""
MongoDB connection manager using Motor (async driver).
Provides a single shared client instance across the app.
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# ── connection state ──────────────────────────────────────────────
_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return (or create) the shared Motor client."""
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(uri)
    return _client


def get_database():
    """Return the instaviral database handle."""
    db_name = os.getenv("DB_NAME", "instaviral")
    return get_client()[db_name]


# ── collection helpers ────────────────────────────────────────────
def jobs_collection():
    return get_database()["analysis_jobs"]


def results_collection():
    return get_database()["analysis_results"]


def benchmarks_collection():
    return get_database()["viral_benchmarks"]


# ── lifecycle ─────────────────────────────────────────────────────
async def connect_db():
    """Called on FastAPI startup — verifies connection."""
    client = get_client()
    await client.admin.command("ping")
    print("[OK] MongoDB connected successfully.")


async def close_db():
    """Called on FastAPI shutdown."""
    global _client
    if _client:
        _client.close()
        _client = None
        print("[CLOSED] MongoDB connection closed.")
