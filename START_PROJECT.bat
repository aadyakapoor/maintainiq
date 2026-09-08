@echo off
setlocal
cd /d "%~dp0"
set NUMBA_DISABLE_JIT=1
set PYTHONDONTWRITEBYTECODE=1
if exist "..\..\work\venv\Scripts\python.exe" (
  "..\..\work\venv\Scripts\python.exe" -B -m streamlit run app.py --global.developmentMode=false --server.port=8501 --browser.serverPort=8501 --browser.serverAddress=127.0.0.1 --server.headless=false
) else (
  call setup_and_run.bat
)
