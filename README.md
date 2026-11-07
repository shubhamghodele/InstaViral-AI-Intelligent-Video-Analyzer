# InstaViral AI
### Intelligent Video Analyser and Content Optimiser for Instagram Reels

---

## Quick Start

### Step 1 — Set up your API keys

Open `.env` and fill in:
```
GROQ_API_KEY=your_key_from_console.groq.com
MONGODB_URI=your_atlas_connection_string
```

### Step 2 — Seed viral benchmarks (run once)
```bash
python seed_benchmarks.py --mode manual
```

### Step 3 — Start the app
Double-click `start.bat` OR run manually:

**Terminal 1 (Backend):**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
python -m streamlit run frontend/app.py
```

Open: http://localhost:8501

---

## Project Structure
```
instaviral/
├── backend/
│   ├── main.py                    # FastAPI app
│   ├── routers/                   # API endpoints
│   ├── services/                  # All AI analysis logic
│   │   ├── video_processor.py     # FFmpeg, OpenCV, PySceneDetect
│   │   ├── audio_analyzer.py      # Groq Whisper
│   │   ├── vision_analyzer.py     # Groq Llama 4 Maverick Vision
│   │   ├── emotion_analyzer.py    # EmotiEffLib + MediaPipe
│   │   ├── feature_extractor.py   # 14-dim vector + cosine similarity
│   │   ├── suggestion_generator.py# Groq Llama 3.3-70B
│   │   └── analyzer.py            # Main orchestrator
│   ├── database/                  # MongoDB CRUD
│   └── models/                    # Pydantic schemas
├── frontend/
│   ├── app.py                     # Main Streamlit app
│   └── pages/
│       ├── 1_Upload_and_Analyze.py
│       └── 2_Results_Dashboard.py
├── seed_benchmarks.py             # One-time benchmark seeder
├── start.bat                      # Start both servers (Windows)
└── requirements.txt
```

## Tech Stack
| Component | Technology |
|---|---|
| Frontend | Streamlit + Plotly |
| Backend | FastAPI + Uvicorn |
| Database | MongoDB Atlas (Motor async) |
| Audio STT | Groq Whisper Large V3 Turbo |
| Vision AI | Groq Llama 4 Maverick (multimodal) |
| Suggestions | Groq Llama 3.3-70B Versatile |
| Emotion | EmotiEffLib (ONNX) + MediaPipe |
| Scene Detection | PySceneDetect |
| Video Download | yt-dlp |
| Video Processing | OpenCV + moviepy |
| Scoring | Cosine Similarity (scikit-learn) |
