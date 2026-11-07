"""
Page 1 — Upload & Analyze
──────────────────────────
User uploads their video, pastes viral URLs, picks niche, hits Analyze.
Polls the FastAPI backend until the job is done, then redirects to results.
"""

import time
import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Upload & Analyze — InstaViral AI", page_icon="📁", layout="wide")

st.title("📁 Upload & Analyze")
st.caption("Upload your Reel/Short and compare it against real viral videos for an accurate Viral Readiness Score.")
st.divider()

# ── shared session state ──────────────────────────────────────────
if "video_id"   not in st.session_state: st.session_state.video_id   = None
if "results"    not in st.session_state: st.session_state.results    = None
if "niche"      not in st.session_state: st.session_state.niche      = "Dance"
if "viral_urls" not in st.session_state: st.session_state.viral_urls = []

# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — Upload video
# ═══════════════════════════════════════════════════════════════════

st.subheader("Step 1 — Upload Your Reel")
uploaded_file = st.file_uploader(
    "Choose your video file (.mp4, .mov)",
    type=["mp4", "mov", "avi", "mkv"],
    help="Max file size: 100 MB",
)

# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — Niche selection
# ═══════════════════════════════════════════════════════════════════

st.subheader("Step 2 — Select Your Niche")
niche = st.selectbox(
    "What type of content is this?",
    options=["Dance", "Comedy", "Tutorial", "Motivation", "Other"],
    index=0,
    help="This helps us match your video against the right viral benchmarks.",
)
st.session_state.niche = niche

# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — Viral video URLs
# ═══════════════════════════════════════════════════════════════════

st.subheader("Step 3 — Paste Viral Video URLs (Strongly Recommended)")

st.info(
    "**Why this matters:** When you provide URLs, your video is compared against "
    "REAL viral videos you choose — making the Viral Readiness Score meaningful and accurate. "
    "Without URLs, we use industry average estimates as a fallback.",
    icon="💡"
)

st.markdown("""
**How to find viral reference URLs:**
1. Go to **YouTube** and search `[your niche] viral shorts 2024` (e.g. `marathi dance viral shorts 2024`)
2. Open a video that went viral (millions of views)
3. Copy its URL from the browser address bar
4. Paste it below

> YouTube Shorts, YouTube videos, and most public video platforms work.
""")

url1 = st.text_input("Viral Reference Video URL 1", placeholder="https://www.youtube.com/watch?v=...")
url2 = st.text_input("Viral Reference Video URL 2", placeholder="https://www.youtube.com/watch?v=...")
url3 = st.text_input("Viral Reference Video URL 3", placeholder="https://www.youtube.com/watch?v=...")

viral_urls = [u.strip() for u in [url1, url2, url3] if u.strip()]
st.session_state.viral_urls = viral_urls

if viral_urls:
    st.success(
        f"✅ {len(viral_urls)} viral reference video(s) added. "
        "Score will be compared against these real viral videos."
    )
else:
    st.warning(
        "No viral URLs provided. Score will use industry average estimates. "
        "Add YouTube URLs above for a more accurate comparison.",
        icon="⚠️"
    )

# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — Analyze button
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Step 4 — Start Analysis")

analyze_btn = st.button(
    "🚀 Analyze My Reel",
    type="primary",
    disabled=(uploaded_file is None),
    use_container_width=True,
)

if uploaded_file is None:
    st.caption("⬆️ Please upload a video file first.")

# ═══════════════════════════════════════════════════════════════════
#  ANALYSIS PIPELINE  (runs when button clicked)
# ═══════════════════════════════════════════════════════════════════

if analyze_btn and uploaded_file:

    progress = st.progress(0, text="Starting...")
    status   = st.empty()

    # ── 1. Upload video to FastAPI ────────────────────────────────
    status.info("📤 Uploading video to server...")
    progress.progress(10, text="Uploading video...")

    try:
        with httpx.Client(timeout=120) as client:
            upload_resp = client.post(
                f"{API_BASE}/api/video/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")},
            )
        if upload_resp.status_code != 200:
            st.error(f"Upload failed: {upload_resp.text}")
            st.stop()

        video_id = upload_resp.json()["video_id"]
        st.session_state.video_id = video_id
        progress.progress(20, text="Video uploaded ✅")

    except Exception as e:
        st.error(f"Could not connect to FastAPI server. Is it running?\n\n{e}")
        st.code("uvicorn backend.main:app --reload --port 8000")
        st.stop()

    # ── 2. Start analysis ─────────────────────────────────────────
    status.info("🧠 Starting AI analysis pipeline...")
    progress.progress(30, text="Starting analysis...")

    try:
        with httpx.Client(timeout=60) as client:
            analyze_resp = client.post(
                f"{API_BASE}/api/analyze",
                json={
                    "video_id":   video_id,
                    "niche":      niche,
                    "viral_urls": viral_urls,
                },
            )
        if analyze_resp.status_code != 200:
            st.error(f"Analysis trigger failed: {analyze_resp.text}")
            st.stop()
    except Exception as e:
        st.error(f"Analysis request failed: {e}")
        st.stop()

    # ── 3. Poll for completion ────────────────────────────────────
    progress_steps = [
        (40, "🎵 Transcribing audio with Groq Whisper..."),
        (55, "👁️ Analysing frames with Groq Llama 4 Maverick..."),
        (70, "😊 Running emotion analysis..."),
        (82, "📊 Computing virality score..."),
        (90, "💡 Generating AI suggestions with Groq Llama 3.3-70B..."),
        (95, "💾 Saving results to MongoDB..."),
    ]
    step_idx  = 0
    max_polls = 120  # 4 minutes max

    for poll in range(max_polls):
        time.sleep(2)

        try:
            with httpx.Client(timeout=10) as client:
                status_resp = client.get(f"{API_BASE}/api/results/{video_id}")
        except Exception:
            continue

        data = status_resp.json()

        if data.get("status") == "processing" or data.get("status") == "pending":
            if step_idx < len(progress_steps):
                pct, msg = progress_steps[step_idx]
                progress.progress(pct, text=msg)
                status.info(msg)
                step_idx = min(step_idx + 1, len(progress_steps) - 1)
            continue

        if status_resp.status_code == 500:
            st.error(f"Analysis failed: {data.get('detail', 'Unknown error')}")
            st.stop()

        if "virality_score" in data:
            # ✅ Done!
            progress.progress(100, text="Analysis complete! ✅")
            status.success("✅ Analysis complete!")
            st.session_state.results = data
            st.balloons()
            time.sleep(1)
            st.switch_page("pages/2_Results_Dashboard.py")
            break
    else:
        st.warning("Analysis is taking longer than expected. Check the Results page manually.")
