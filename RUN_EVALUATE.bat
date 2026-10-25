@echo off
title DarkPatternGuard - evaluation
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=python
where py >nul 2>nul && set PY=py
echo Rebuilding metrics, confusion matrices, comparison chart and sample audit reports...
echo (RoBERTa runs on the CPU here, so this takes a few minutes.)
%PY% src\evaluate.py
%PY% src\rule_coverage.py >nul
if exist models\roberta\binary\config.json %PY% src\eval_agreement.py
echo.
echo Done. See results\metrics\summary_table.csv and results\figures\model_comparison.png
pause
