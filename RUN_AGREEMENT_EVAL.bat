@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=python
where py >nul 2>nul && set PY=py
%PY% src\eval_agreement.py > results\metrics\agreement_rule.txt 2>&1
type results\metrics\agreement_rule.txt
