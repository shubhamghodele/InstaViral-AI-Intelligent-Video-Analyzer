# InstaViral AI – Intelligent Video Analyzer and Content Optimizer for Instagram Reels

## Project Overview

InstaViral AI is an AI-powered video analysis application designed to analyse Instagram Reels and provide insights to help content creators improve their videos.

The application examines audio, visual elements, emotions, and video structure to generate a Viral Readiness Score and AI-based improvement suggestions.

## Key Features

- **Video Analysis:** Analyse uploaded videos using audio and visual features.
- **Audio Analysis:** Examine speech, sentiment, and call-to-action (CTA) presence.
- **Visual Analysis:** Evaluate lighting, composition, colour vibrancy, and on-screen text.
- **Emotion Detection:** Analyse facial expressions and emotional variations.
- **Video Metrics:** Examine scene cuts, motion intensity, and music presence.
- **Viral Readiness Score:** Generate a score from 0 to 100 using comparison with viral benchmark data.
- **AI Recommendations:** Generate suggestions to improve video content.
- **Multilingual Support:** Provide reports and content suggestions in Hindi, Marathi, and English.
- **Content Suggestions:** Generate captions, hashtags, and recommended posting times.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Database | MongoDB Atlas |
| AI Integration | Groq API |
| Data Analysis | Pandas, NumPy |
| Video Processing | OpenCV, FFmpeg |
| Visualisation | Plotly |
| Machine Learning | Scikit-learn |

## How It Works

1. Upload a video through the Streamlit application.
2. The backend processes the video and extracts relevant audio and visual information.
3. AI and analysis modules examine the extracted features.
4. The system compares video features with available viral benchmark data.
5. A Viral Readiness Score is generated.
6. The application displays analysis results and AI-generated improvement suggestions.

## Project Structure

```text
InstaViral-AI/
│
├── backend/
│   ├── database/
│   ├── models/
│   ├── routers/
│   ├── services/
│   └── main.py
│
├── frontend/
│   ├── pages/
│   └── app.py
│
├── temp/
├── uploads/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── seed_benchmarks.py
├── start.bat
├── README.md
└── SETUP_GUIDE.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shubhamghodele/InstaViral-AI.git
cd InstaViral-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a template.

Add your own credentials locally:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_connection_string
DB_NAME=instaviral
```

**Important:** Never upload your actual `.env` file, API keys, or database credentials to GitHub.

### 5. Run the Application

Start the backend:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
python -m streamlit run frontend/app.py
```

Open the application:

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/video/upload` | Upload a video |
| POST | `/api/analyze` | Analyse a video |
| GET | `/api/results/{video_id}` | Retrieve analysis results |
| GET | `/api/benchmarks/{category}` | Retrieve benchmark data |
| GET | `/health` | Check backend status |

## Future Enhancements

- Real-time video analysis
- Support for additional social media platforms
- Trend detection and analysis
- Creator performance dashboard
- Cloud deployment

## Author

**Shubham Ghodele**

GitHub: [@shubhamghodele](https://github.com/shubhamghodele)

---
