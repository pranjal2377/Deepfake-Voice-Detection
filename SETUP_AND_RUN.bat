@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo  Deepfake Voice ^& Scam Detection
echo  Windows Setup / Run
echo ==============================================

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  py -m venv .venv
  if errorlevel 1 python -m venv .venv
)

echo [2/3] Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo [3/3] Choose an action:
echo   1 - Web Dashboard
echo   2 - Real-time microphone
echo   3 - Demo analysis
echo   4 - Native desktop wrapper
echo   5 - Mobile PWA wrapper
echo   6 - Train model
echo   7 - Evaluate model
echo   8 - Run tests
echo   9 - Exit
echo.
set /p choice=Enter choice [1-9]: 

if "%choice%"=="1" .venv\Scripts\python.exe -m streamlit run src\ui\dashboard.py
if "%choice%"=="2" .venv\Scripts\python.exe scripts\realtime.py --mode mic
if "%choice%"=="3" .venv\Scripts\python.exe scripts\realtime.py --mode demo
if "%choice%"=="4" .venv\Scripts\python.exe desktop_app.py
if "%choice%"=="5" .venv\Scripts\python.exe run_mobile_app.py
if "%choice%"=="6" .venv\Scripts\python.exe scripts\train.py
if "%choice%"=="7" .venv\Scripts\python.exe scripts\evaluate.py
if "%choice%"=="8" .venv\Scripts\python.exe -m pytest tests/ -v --tb=short
if "%choice%"=="9" exit /b 0
pause
