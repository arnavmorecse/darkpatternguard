@echo off
title DarkPatternGuard
cd /d "%~dp0"
set PYTHONUTF8=1
echo ============================================================
echo   DarkPatternGuard - setting up and starting the demo
echo ============================================================
set PY=
where py >nul 2>nul && set PY=py
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python was not found. Install Python 3.10+ from python.org ^(tick "Add to PATH"^), then run this again.
  pause & exit /b 1
)
%PY% --version
echo.
echo [1/3] Installing packages (first run only takes a few minutes)...
%PY% -m pip install --disable-pip-version-check -q -r requirements-core.txt || (echo pip install failed & pause & exit /b 1)
if not exist models\roberta\binary\config.json goto :noroberta
if exist models\roberta\.deps_installed goto :noroberta
echo [1b] RoBERTa models found - installing PyTorch + Transformers (one time, ~250 MB download)...
%PY% -m pip install --disable-pip-version-check -q -r requirements-roberta.txt || (echo PyTorch install failed - the app will use TF-IDF only & goto :noroberta)
echo done> models\roberta\.deps_installed
:noroberta
if exist models\tfidf\.trained_here goto :skiptrain
echo [2/3] Training TF-IDF models with your local library versions (about 1-2 min)...
%PY% src\train_baselines.py || (echo training failed & pause & exit /b 1)
echo done> models\tfidf\.trained_here
:skiptrain
echo [3/3] Starting the app - your browser will open at http://localhost:8501
echo        Keep this window open while you use the app. Close it to stop.
start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8501"
%PY% -m streamlit run app.py --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
pause
