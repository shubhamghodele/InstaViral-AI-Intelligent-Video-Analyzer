"""
Benchmark Seeder Script
────────────────────────
Run this ONCE before demoing the project to pre-seed viral benchmark
feature vectors into MongoDB.

Two modes:
  1. URL mode   — provide real viral video URLs → full analysis pipeline
  2. Manual mode — directly insert pre-defined feature vectors (fast, offline)

Usage:
  python seed_benchmarks.py --mode manual     ← recommended for first setup
  python seed_benchmarks.py --mode url        ← requires internet + viral URLs
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════
#  PRE-DEFINED BENCHMARK VECTORS  (manual mode)
#  These are representative feature vectors for viral Reels.
#  Dimensions (in order):
#  sentiment, hook, cta, visual_energy, lighting, composition,
#  color_vibrancy, text_overlay, pos_emotion, emotion_var,
#  smile_ratio, cuts_pm, motion, music
# ═══════════════════════════════════════════════════════════════════

MANUAL_BENCHMARKS = {
    "Dance": [
        # High energy, bright, fast cuts, music always present
        [0.75, 0.85, 0.6, 0.90, 0.80, 0.82, 0.88, 0.70, 0.82, 0.75, 0.78, 0.75, 0.90, 1.0],
        [0.70, 0.80, 0.5, 0.85, 0.78, 0.80, 0.85, 0.65, 0.79, 0.72, 0.75, 0.70, 0.88, 1.0],
        [0.80, 0.90, 0.7, 0.92, 0.83, 0.85, 0.90, 0.75, 0.85, 0.80, 0.82, 0.80, 0.92, 1.0],
    ],
    "Comedy": [
        # Strong hook, expressive face, moderate cuts
        [0.80, 0.92, 0.75, 0.78, 0.72, 0.80, 0.75, 0.80, 0.88, 0.85, 0.85, 0.55, 0.65, 0.70],
        [0.75, 0.88, 0.70, 0.75, 0.70, 0.78, 0.72, 0.75, 0.85, 0.82, 0.82, 0.50, 0.60, 0.65],
        [0.82, 0.90, 0.78, 0.80, 0.75, 0.82, 0.78, 0.82, 0.90, 0.87, 0.88, 0.60, 0.68, 0.72],
    ],
    "Tutorial": [
        # Text overlay, steady pace, clear narration
        [0.70, 0.82, 0.80, 0.75, 0.85, 0.88, 0.72, 0.92, 0.70, 0.60, 0.70, 0.45, 0.45, 0.60],
        [0.65, 0.78, 0.78, 0.72, 0.82, 0.85, 0.70, 0.90, 0.68, 0.58, 0.68, 0.42, 0.42, 0.58],
        [0.72, 0.85, 0.82, 0.78, 0.88, 0.90, 0.75, 0.95, 0.72, 0.62, 0.72, 0.48, 0.48, 0.62],
    ],
    "Motivation": [
        # Positive sentiment, strong CTA, good lighting
        [0.90, 0.88, 0.85, 0.82, 0.88, 0.85, 0.80, 0.78, 0.85, 0.78, 0.82, 0.50, 0.60, 0.75],
        [0.88, 0.85, 0.82, 0.80, 0.85, 0.82, 0.78, 0.75, 0.82, 0.75, 0.80, 0.48, 0.58, 0.72],
        [0.92, 0.90, 0.88, 0.85, 0.90, 0.88, 0.82, 0.80, 0.88, 0.80, 0.85, 0.52, 0.62, 0.78],
    ],
    "Other": [
        # General viral content averages
        [0.72, 0.82, 0.70, 0.80, 0.78, 0.80, 0.78, 0.72, 0.75, 0.70, 0.72, 0.58, 0.68, 0.72],
        [0.70, 0.80, 0.68, 0.78, 0.76, 0.78, 0.76, 0.70, 0.73, 0.68, 0.70, 0.56, 0.66, 0.70],
    ],
}


async def seed_manual():
    """Insert pre-defined benchmark vectors into MongoDB."""
    from backend.database.mongo import connect_db, close_db
    from backend.database.crud  import save_benchmark

    print("Connecting to MongoDB...")
    await connect_db()

    total = 0
    for category, vectors in MANUAL_BENCHMARKS.items():
        for i, vector in enumerate(vectors):
            await save_benchmark(
                category=category,
                video_url=f"seed://{category.lower()}/benchmark_{i+1}",
                feature_vector=vector,
                metadata={"source": "manual_seed", "index": i + 1},
            )
            print(f"  [OK] Seeded {category} benchmark {i+1}")
            total += 1

    await close_db()
    print(f"\nDone! Seeded {total} benchmark vectors across {len(MANUAL_BENCHMARKS)} niches.")
    print("You can now run the full app and comparisons will work from the first analysis.")


async def seed_from_urls(urls_by_category: dict):
    """Download and analyse real viral videos, save their vectors."""
    from backend.database.mongo     import connect_db, close_db
    from backend.database.crud      import save_benchmark
    from backend.services.analyzer  import analyze_single_video
    from backend.services.video_processor import download_video, TEMP_DIR

    await connect_db()

    for category, urls in urls_by_category.items():
        print(f"\nSeeding category: {category}")
        for url in urls:
            try:
                print(f"  Downloading: {url}")
                vpath    = download_video(url, save_dir=TEMP_DIR)
                print(f"  Analysing...")
                analysis = analyze_single_video(vpath)
                await save_benchmark(
                    category=category,
                    video_url=url,
                    feature_vector=analysis["feature_vector"],
                    metadata={"source": "url_seed"},
                )
                print(f"  [OK] Saved benchmark for {category}")
            except Exception as e:
                print(f"  [FAILED] {url}: {e}")

    await close_db()
    print("\n✅ URL seeding complete.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed viral benchmark vectors.")
    parser.add_argument(
        "--mode",
        choices=["manual", "url"],
        default="manual",
        help="'manual' uses pre-defined vectors (fast). 'url' downloads real videos (slow).",
    )
    args = parser.parse_args()

    if args.mode == "manual":
        print("Seeding with pre-defined benchmark vectors (manual mode)...")
        asyncio.run(seed_manual())
    else:
        print("URL mode: Edit this script and add your viral video URLs.")
        print("   Then re-run with --mode url")
        # Example — replace with real viral URLs:
        # urls = {
        #     "Dance":      ["https://www.instagram.com/reel/...", "..."],
        #     "Comedy":     ["https://www.instagram.com/reel/...", "..."],
        #     "Tutorial":   ["https://www.instagram.com/reel/...", "..."],
        #     "Motivation": ["https://www.instagram.com/reel/...", "..."],
        # }
        # asyncio.run(seed_from_urls(urls))
