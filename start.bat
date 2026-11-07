@echo off
echo ================================
echo   InstaViral AI - Starting Up
echo ================================

echo.
echo Killing any old Python processes on ports 8000 and 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 " ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak > nul

echo.
echo [1/2] Starting FastAPI backend on port 8000...
start "InstaViral Backend" cmd /k "cd /d %~dp0 && set PYTHONIOENCODING=utf-8 && python -m uvicorn backend.main:app --reload --port 8000"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo [2/2] Starting Streamlit frontend on port 8501...
start "InstaViral Frontend" cmd /k "cd /d %~dp0 && set PYTHONIOENCODING=utf-8 && python -m streamlit run frontend/app.py --server.port 8501"

echo.
echo ================================
echo   Both servers are running!
echo   Backend:  http://localhost:8000/docs
echo   Frontend: http://localhost:8501
echo ================================
pause
