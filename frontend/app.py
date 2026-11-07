"""
InstaViral AI — Streamlit Main App
────────────────────────────────────
Entry point. Sets up page config, navigation, and shared session state.

Run with:  streamlit run frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="InstaViral AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared session state defaults ────────────────────────────────
if "video_id"   not in st.session_state: st.session_state.video_id   = None
if "results"    not in st.session_state: st.session_state.results    = None
if "niche"      not in st.session_state: st.session_state.niche      = "Dance"
if "viral_urls" not in st.session_state: st.session_state.viral_urls = []

# ── sidebar branding ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/instagram-reel.png", width=80)
    st.title("InstaViral AI")
    st.caption("Intelligent Video Analyser\nfor Instagram Reels")
    st.divider()
    st.markdown("""
    **How it works:**
    1. 📁 Upload your Reel
    2. 🔗 Paste viral video URLs
    3. 🎯 Choose your niche
    4. 🚀 Click Analyze
    5. 📊 Get your Virality Score
    6. 💡 Read AI suggestions
    """)
    st.divider()
    st.caption("Built with Groq AI + FastAPI + Streamlit")

# ── main landing ──────────────────────────────────────────────────
st.title("🎬 InstaViral AI")
st.subheader("Analyse your Instagram Reel and discover how to make it go viral.")

st.info(
    "👈 Use the **sidebar navigation** to go to **Upload & Analyze** or view **Results**.\n\n"
    "Or use the pages directly from the sidebar menu.",
    icon="ℹ️"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("AI Models Used", "4 Groq Models")
with col2:
    st.metric("Analysis Dimensions", "14 Features")
with col3:
    st.metric("Output", "Virality Score + Report")
