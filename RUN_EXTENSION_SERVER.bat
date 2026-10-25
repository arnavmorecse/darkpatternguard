@echo off
title DarkPatternGuard - extension server
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=python
where py >nul 2>nul && set PY=py
echo ============================================================
echo   DarkPatternGuard - local model server for the browser extension
echo   Keep this window open while browsing. Close it to stop.
echo ============================================================
if not exist models\tfidf\binary.joblib (
  echo Models not found - run RUN_DEMO.bat once first.
  pause & exit /b 1
)
%PY% server.py
pause
