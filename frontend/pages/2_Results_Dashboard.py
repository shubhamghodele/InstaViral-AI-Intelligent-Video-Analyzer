"""
Page 2 — Results Dashboard
"""

import httpx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Results — InstaViral AI",
    page_icon="📊",
    layout="wide",
)

if "video_id" not in st.session_state: st.session_state.video_id = None
if "results"  not in st.session_state: st.session_state.results  = None

_LANG_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "en": "English",
    "te": "Telugu", "ta": "Tamil",  "kn": "Kannada",
    "gu": "Gujarati","pa": "Punjabi",
}

# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _score_color(score: float) -> str:
    if score >= 65: return "#28a745"
    if score >= 45: return "#ffc107"
    return "#dc3545"


def _score_label_and_action(score: float) -> tuple:
    """
    Returns (label, action_text) based on score range.
    Label is honest, action_text tells user exactly what to do next.
    """
    if score >= 80:
        return (
            "Strong",
            "Your video is well-optimized on most features. "
            "Post it — focus your energy on the caption, hashtags and posting time below."
        )
    elif score >= 65:
        return (
            "Good",
            "Good foundation. Fix the RED (Critical) issues in the table below "
            "before posting — those changes will have the biggest impact."
        )
    elif score >= 50:
        return (
            "Fair",
            "Several key features need improvement. "
            "Work through the Critical and Weak issues below, then re-analyze."
        )
    else:
        return (
            "Needs Work",
            "Multiple important features are below target. "
            "Read the full AI Report below — follow each action item before posting."
        )


def _render_radar(results: dict, benchmark_source: str):
    gap_analysis = results.get("gap_analysis", [])
    if not gap_analysis:
        st.info("Gap analysis data not available.")
        return

    # Exclude N/A emotion features from radar so chart is not misleading
    active = [g for g in gap_analysis if g["status"] != "na"]
    if not active:
        st.info("No active features to chart.")
        return

    labels     = [g["label"]      for g in active]
    user_vals  = [g["user_score"] for g in active]
    target_vals= [g["viral_avg"]  for g in active]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_vals + [user_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="Your Video",
        line_color="#4361ee",
        fillcolor="rgba(67,97,238,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=target_vals + [target_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=f"Target ({benchmark_source})",
        line_color="#f72585",
        fillcolor="rgba(247,37,133,0.1)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=380,
        margin=dict(t=20, b=20, l=30, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_emotion_timeline(results: dict):
    timeline = results.get("emotion_timeline", [])
    if not timeline:
        if not results.get("has_face"):
            st.info("No face detected — emotion timeline not available.")
        else:
            st.info("Emotion data not available.")
        return

    df = pd.DataFrame(timeline)
    if df.empty:
        return

    EMOTION_SCORE = {
        "happy": 1.0, "surprise": 0.8, "surprised": 0.8, "neutral": 0.5,
        "sad": 0.2, "angry": 0.1, "disgust": 0.1, "fear": 0.2,
        "fearful": 0.2, "contempt": 0.15,
    }
    df["sentiment"] = df["emotion"].map(lambda e: EMOTION_SCORE.get(e.lower(), 0.5))
    df["Emotion"]   = df["emotion"].str.capitalize()

    fig = px.line(
        df, x="time_sec", y="sentiment", color="Emotion", markers=True,
        labels={"time_sec": "Time (s)", "sentiment": "Positivity (0-1)"},
        height=350,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(margin=dict(t=10, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_gap_table(results: dict, benchmark_source: str):
    gap_analysis = results.get("gap_analysis", [])
    if not gap_analysis:
        st.info("No gap analysis data available.")
        return

    # Explain where Target column comes from — right above the table
    if "you provided" in benchmark_source:
        st.caption(
            f"Target values = measured from {benchmark_source}. "
            "This is a REAL comparison against actual viral videos."
        )
    else:
        st.caption(
            "Target values = research-based estimates of what high-performing content "
            "in this niche typically looks like. These are NOT measured from real viral videos. "
            "For a real comparison, paste YouTube URLs in the Upload page."
        )

    rows = []
    for g in gap_analysis:
        status_map = {
            "critical": "🔴 Critical",
            "weak":     "🟡 Weak",
            "ok":       "🟢 OK",
            "strong":   "💪 Strong",
            "na":       "⚪ N/A (no face)",
        }
        is_na = g["status"] == "na"
        rows.append({
            "Feature":            g["label"],
            "Your Score (0-1)":   "—" if is_na else f"{g['user_score']:.2f}",
            "Target (0-1)":       f"{g['viral_avg']:.2f}",
            "Gap":                "—" if is_na else f"{g['gap']:+.2f}",
            "Priority":           status_map.get(g["status"], g["status"]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature":           st.column_config.TextColumn(width="large"),
            "Your Score (0-1)":  st.column_config.TextColumn(width="small"),
            "Target (0-1)":      st.column_config.TextColumn(width="small"),
            "Gap":               st.column_config.TextColumn(width="small"),
            "Priority":          st.column_config.TextColumn(width="medium"),
        },
    )

    # Plain English guide
    st.markdown("""
**How to read this table:**
- **Your Score**: 0 = feature completely missing, 1 = feature is perfect
- **Target**: what this feature looks like in high-performing content for your niche
- **Gap**: how far you are from the target (negative gap = you are ABOVE target — great!)
- **🔴 Critical**: fix this before posting (gap > 0.3)
- **🟡 Weak**: improve if you can (gap 0.15–0.3)
- **🟢 OK / 💪 Strong**: doing well — keep it up
- **⚪ N/A**: not applicable to your video type (e.g. emotion features for no-face videos)
""")


# ═══════════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════════

st.title("📊 Analysis Results Dashboard")

# Manual lookup
with st.expander("🔍 Look up results by Video ID"):
    manual_id = st.text_input("Video ID", placeholder="Paste your video_id here")
    if st.button("Fetch Results") and manual_id:
        try:
            with httpx.Client(timeout=15) as c:
                resp = c.get(f"{API_BASE}/api/results/{manual_id}")
            if resp.status_code == 200 and "virality_score" in resp.json():
                st.session_state.results  = resp.json()
                st.session_state.video_id = manual_id
                st.rerun()
            else:
                st.error("Results not found or still processing.")
        except Exception as e:
            st.error(f"Could not fetch: {e}")

results = st.session_state.results
if not results:
    st.info("No results yet. Go to **Upload & Analyze** to analyse a video first.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  EXTRACT ALL VALUES
# ═══════════════════════════════════════════════════════════════════

virality          = results.get("virality_score", 0)
duration          = results.get("duration_sec",   0)
has_face          = results.get("has_face",        False)
benchmarks        = results.get("viral_benchmarks_used", 0)
niche             = results.get("niche", "Unknown")
benchmark_label   = results.get("benchmark_label", "research-based estimates")
detected_lang     = results.get("detected_language", "en")
detected_lang_name = _LANG_NAMES.get(detected_lang, detected_lang.upper())

# Is this comparison against real videos or estimates?
is_real_comparison = "you provided" in benchmark_label
benchmark_source   = "Real Viral Videos" if is_real_comparison else "Estimates"

score_label, score_action = _score_label_and_action(virality)

# ═══════════════════════════════════════════════════════════════════
#  TOP ROW
# ═══════════════════════════════════════════════════════════════════

col1, col2 = st.columns([1, 2])

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=virality,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Viral Readiness Score", "font": {"size": 18}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": _score_color(virality)},
            "steps": [
                {"range": [0,  50], "color": "#ffcccc"},
                {"range": [50, 65], "color": "#fff3cd"},
                {"range": [65, 100],"color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 65,
            },
        },
    ))
    fig_gauge.update_layout(height=220, margin=dict(t=40, b=5, l=5, r=5))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    # Score label + honest action
    color_map = {"Strong": "🟢", "Good": "🔵", "Fair": "🟡", "Needs Work": "🔴"}
    icon = color_map.get(score_label, "⚪")
    st.markdown(f"## {icon} {score_label}")
    st.info(score_action)

    # Where does this score come from?
    if is_real_comparison:
        st.success(
            f"This score compares your video against **{benchmark_label}**. "
            "This is a real, meaningful comparison."
        )
    else:
        st.warning(
            f"This score compares your video against **research-based target values** "
            f"for the '{niche}' niche — NOT measured from real viral videos. "
            "To get a real comparison, paste YouTube viral video URLs when uploading."
        )

st.divider()

# ─── Key Metrics Row ─────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Niche",            niche.title())
m2.metric("Duration",         f"{duration:.0f}s")
m3.metric("Face Detected",    "Yes" if has_face else "No")
m4.metric("Language",         detected_lang_name)
m5.metric("Best Post Time",   results.get("posting_time", "—"))

st.divider()

# ═══════════════════════════════════════════════════════════════════
#  HOW IS THIS CALCULATED?  (Transparent explanation)
# ═══════════════════════════════════════════════════════════════════

with st.expander("How is this score calculated? (Click to understand)"):
    st.markdown(f"""
**What we measure (14 features of your video):**

| # | Feature | How it is measured |
|---|---------|-------------------|
| 1 | Audio Sentiment | Groq Whisper transcribes your speech, Groq AI scores the tone |
| 2 | Hook Strength | AI judges how compelling your first 2 seconds of speech are |
| 3 | Call-to-Action | AI checks if you say "follow", "share", "like" etc. |
| 4 | Visual Energy | Groq Vision AI rates how exciting your frames look (0-10) |
| 5 | Lighting Quality | Groq Vision AI rates your lighting (0-10) |
| 6 | Frame Composition | Groq Vision AI rates framing/composition (0-10) |
| 7 | Color Vibrancy | OpenCV measures color saturation of your frames |
| 8 | On-Screen Text | Groq Vision detects if captions are visible on screen |
| 9 | Positive Emotion | EmotiEffLib measures happiness/surprise on detected face |
| 10 | Emotion Variety | How much emotion changes across frames |
| 11 | Smile Ratio | % of face frames showing positive emotion |
| 12 | Editing Pace | PySceneDetect counts scene cuts per minute |
| 13 | Motion Intensity | OpenCV optical flow measures how much movement is in the video |
| 14 | Background Music | Librosa detects if music/beat is present in audio |

**What are the Target values?**

{"This analysis used **real viral videos you provided** as the target. The 'Target' column shows the average feature values of those videos." if is_real_comparison else f"""
The **Target** column shows research-based estimates of what high-performing **{niche}** content typically looks like.

**Important: these targets were NOT measured from real viral videos.** They are informed estimates based on content research. They are reasonable, but they are not exact. The gap analysis and AI suggestions are still useful — they tell you what to improve. But the score number itself is an estimate, not a scientific measurement.

**How to get a real score:** On the Upload page, paste 2-3 YouTube Shorts URLs of viral content in your niche. The system will download and analyze those real videos and use their actual feature values as your target.
"""}

**How is the final score calculated?**

We compute **cosine similarity** between your 14-feature vector and the target vector.
Cosine similarity measures the angle between two vectors — if your video has the same feature pattern as high-performing content, the angle is small and the score is high.
Result is scaled to 0-100%.
    """)

st.divider()

# ═══════════════════════════════════════════════════════════════════
#  RADAR CHART + EMOTION TIMELINE
# ═══════════════════════════════════════════════════════════════════

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Feature Radar Chart")
    st.caption("Blue = your video | Pink = target for your niche")
    _render_radar(results, benchmark_source)

with col_right:
    st.subheader("Emotion Timeline")
    _render_emotion_timeline(results)

st.divider()

# ═══════════════════════════════════════════════════════════════════
#  FEATURE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════

st.subheader("Feature-by-Feature Breakdown")
_render_gap_table(results, benchmark_label)

st.divider()

# ═══════════════════════════════════════════════════════════════════
#  AI SUGGESTIONS
# ═══════════════════════════════════════════════════════════════════

st.subheader("AI Optimization Report")
st.caption("Generated by Groq Llama 3.3-70B based on your video's specific gaps.")
suggestions = results.get("suggestions", "")
if suggestions:
    st.markdown(suggestions)
else:
    st.info("No suggestions generated.")

st.divider()

# ═══════════════════════════════════════════════════════════════════
#  CAPTION + HASHTAGS
# ═══════════════════════════════════════════════════════════════════

cap_col, hash_col = st.columns(2)

with cap_col:
    st.subheader("Suggested Caption")
    st.caption("AI-generated caption based on your niche and language.")
    caption = results.get("caption", "")
    st.text_area("Copy and paste this caption:", value=caption, height=150)

with hash_col:
    st.subheader("Recommended Hashtags")
    st.caption("Includes Indian niche hashtags for regional reach.")
    hashtags = results.get("hashtags", [])
    if hashtags:
        hashtag_str = "  ".join(hashtags)
        st.text_area("Copy these hashtags:", value=hashtag_str, height=150)
        st.caption(f"{len(hashtags)} hashtags")

st.divider()

# ═══════════════════════════════════════════════════════════════════
#  TRANSCRIPT
# ═══════════════════════════════════════════════════════════════════

with st.expander("Full Transcript"):
    transcript      = results.get("transcript", "")
    eng_translation = results.get("english_translation", "")
    hook_text       = results.get("hook_text", "")

    if hook_text:
        st.info(f"**Hook (first few seconds):** {hook_text}")

    if detected_lang != "en" and transcript:
        t1, t2 = st.columns(2)
        with t1:
            st.caption(f"Original ({detected_lang_name})")
            st.text(transcript)
        with t2:
            st.caption("English Translation (used for AI scoring)")
            st.text(eng_translation if eng_translation else "Translation not available.")
    else:
        st.text(transcript if transcript else "No speech detected in video.")

# ═══════════════════════════════════════════════════════════════════
#  SCENE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

with st.expander("Scene-by-Scene Visual Analysis"):
    st.caption("Groq Llama 4 Scout Vision describes what it sees in each sampled frame.")
    scenes = results.get("scene_analysis", [])
    if scenes:
        for s in scenes:
            st.markdown(
                f"**Frame {s.get('frame_number','?')}** "
                f"(Visual Energy: {s.get('visual_score',0):.1f}/10): "
                f"{s.get('description','')}"
            )
    else:
        st.info("No scene data available.")

holistic = results.get("holistic_assessment", "")
if holistic:
    with st.expander("🎬 Holistic Video Assessment (AI viewed all frames together)"):
        st.caption(
            "Groq Llama 4 Scout Vision analysed ALL 4 key frames simultaneously "
            "alongside your gap analysis — giving specific visual observations about "
            "your actual video, not just abstract scores."
        )
        st.markdown(holistic)
