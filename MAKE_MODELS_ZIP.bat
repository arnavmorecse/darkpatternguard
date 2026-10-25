@echo off
rem Packs models\roberta into deploy_release\roberta_models.zip for the GitHub release (about 480 MB, takes a minute).
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py
if not defined PY set PY=python
%PY% tools\make_models_zip.py
if errorlevel 1 (echo Something went wrong. & pause & exit /b 1)
echo.
echo Done: deploy_release\roberta_models.zip
pause
