# InstaViral AI — Setup & Run Guide
**Dhole Patil College of Engineering, Pune — Group 014**

---

## PART 1 — INSTALL THESE FIRST (Do Once on Any New Laptop)

Before touching the project, install these 3 things on your laptop.

---

### 1. Python 3.11

**Why:** The entire project is written in Python. Must be version 3.11 — not 3.12 or 3.13.

**Download:** https://www.python.org/downloads/release/python-3119/
→ Click **"Windows installer (64-bit)"** → Run it
→ ⚠️ On first screen: **tick the checkbox "Add Python to PATH"** then click Install Now

**Check it worked — open PowerShell and run:**
```
python --version
```
Should show: `Python 3.11.x`

---

### 2. MongoDB Community Server 7.0

**Why:** The project stores all analysis results and benchmark data in MongoDB running on your laptop.

**Download:** https://www.mongodb.com/try/download/community
→ Select Version: **7.0**, Platform: **Windows**, Package: **MSI**
→ Run the installer → Choose **Complete**
→ ✅ Check **"Install MongoDB as a Service"** (so it auto-starts with Windows)
→ Click Install

**Check it worked — open PowerShell and run:**
```
mongosh --eval "db.runCommand({ ping: 1 })"
```
Should show: `{ ok: 1 }`

> If not working: Press Win+R → type `services.msc` → find **MongoDB** → right-click → Start

---

### 3. Groq API Key (Free — No Credit Card)

**Why:** All AI features (audio transcription, video frame analysis, suggestion generation) use Groq's free AI API.

**How to get:**
1. Go to https://console.groq.com → Sign up (free)
2. Click **"API Keys"** in sidebar → **"Create API Key"**
3. Copy the key (starts with `gsk_...`)
4. Open the file `.env` in your project folder and paste it:
```
GROQ_API_KEY=gsk_your_key_here
```

---

## PART 2 — ONE-TIME SETUP COMMANDS (Run These Once Only)

Open PowerShell and run these commands **in this exact order:**

---

### Command 1 — Go to the project folder
```
cd C:\Users\Admin\Documents\instaviral
```

---

### Command 2 — Install all Python packages
```
pip install -r requirements.txt
```
⏳ Takes 5–10 minutes. Wait until it fully finishes before going to next command.

---

### Command 3 — Seed the database with benchmark data
```
python seed_benchmarks.py --mode manual
```
Expected output:
```
[OK] Seeded Dance benchmark 1
[OK] Seeded Dance benchmark 2
[OK] Seeded Comedy benchmark 1
...
Done! Seeded 14 benchmark vectors across 5 niches.
```

> This fills MongoDB with reference data so the app can compare your video
> against what viral videos typically look like.

---

## PART 3 — HOW TO RUN THE PROJECT (Every Time You Want to Use It)

You need **2 terminal windows open at the same time.**
Both must keep running while you use the app. Do NOT close them.

---

### Command 4 — Start MongoDB (only if it stopped)
```
net start MongoDB
```
> Skip this if MongoDB is already running (it runs automatically on Windows startup if installed as a service).

---

### Command 5 — Open Terminal 1 and Start the Backend FIRST
```
cd C:\Users\Admin\Documents\instaviral
python -m uvicorn backend.main:app --reload --port 8000
```

✅ Wait until you see this before moving to Command 6:
```
[OK] MongoDB connected successfully.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### Command 6 — Open Terminal 2 and Start the Frontend SECOND
```
cd C:\Users\Admin\Documents\instaviral
python -m streamlit run frontend/app.py
```

✅ Wait until you see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

### Command 7 — Open the App in Your Browser
```
http://localhost:8501
```

---

### OR — Start Both With One Click (Easiest Way)

Double-click **`start.bat`** in the project folder.
It automatically opens both Terminal 1 and Terminal 2 for you.
Then open your browser and go to: **http://localhost:8501**

---

## PART 4 — USING THE APP (Step by Step)

| Step | What to do |
|------|-----------|
| **1** | Go to http://localhost:8501 → click **"1 Upload and Analyze"** in sidebar |
| **2** | Click **Browse files** → upload your Reel (`.mp4`, `.mov`, max 100MB) |
| **3** | Select your **content niche** (Dance / Comedy / Tutorial / Motivation / Other) |
| **4** | Paste 1–3 YouTube viral video URLs for accurate comparison (optional but recommended) |
| **5** | Click **"🚀 Analyze My Reel"** |
| **6** | Wait **2–5 minutes** — results appear automatically |
| **7** | View your Virality Score, Gap Table, AI Report, Caption, and Hashtags |

---

## PART 5 — COMMAND ORDER SUMMARY (Cheat Sheet)

```
══════════════════════════════════════════════════════════════
  FIRST TIME ONLY — Run in this order:
══════════════════════════════════════════════════════════════

  Command 1:   cd C:\Users\Admin\Documents\instaviral
  Command 2:   pip install -r requirements.txt
  Command 3:   python seed_benchmarks.py --mode manual

══════════════════════════════════════════════════════════════
  EVERY TIME YOU WANT TO RUN — Run in this order:
══════════════════════════════════════════════════════════════

  Command 4:   net start MongoDB
               (only if MongoDB stopped — usually not needed)

  Command 5:   [Terminal 1 — run FIRST, keep open]
               cd C:\Users\Admin\Documents\instaviral
               python -m uvicorn backend.main:app --reload --port 8000

               Wait for: "Application startup complete."

  Command 6:   [Terminal 2 — run SECOND, keep open]
               cd C:\Users\Admin\Documents\instaviral
               python -m streamlit run frontend/app.py

               Wait for: "Local URL: http://localhost:8501"

  Command 7:   Open browser → http://localhost:8501

══════════════════════════════════════════════════════════════
  SHORTCUT — Skip Commands 5, 6, 7:
══════════════════════════════════════════════════════════════

  Double-click start.bat → then open http://localhost:8501

══════════════════════════════════════════════════════════════
  VERIFY MONGODB IS RUNNING:
══════════════════════════════════════════════════════════════

  mongosh --eval "db.runCommand({ ping: 1 })"
  Expected: { ok: 1 }

══════════════════════════════════════════════════════════════
```

---

## PART 6 — COMMON ERRORS & FIXES

| Error Message | What it means | Fix |
|---------------|---------------|-----|
| `Could not connect to FastAPI server` | Backend not running | Run Command 5 first |
| `GROQ_API_KEY is not set` | Missing API key in .env | Add your key to `.env` file |
| `MongoDB connection failed` | MongoDB service stopped | Run `net start MongoDB` |
| `ModuleNotFoundError` | Packages not installed | Run Command 2 again |
| `Port 8000 already in use` | Old server still running | Double-click `start.bat` |
| Analysis stuck over 10 min | Viral URL failed to download | Remove URLs or try different ones |

---

*InstaViral AI — Dhole Patil College of Engineering, Pune, Group 014*
