@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  if exist "C:\ProgramData\anaconda3\python.exe" (
    "C:\ProgramData\anaconda3\python.exe" -m venv .venv
  ) else (
    py -3.10 -m venv .venv
  )
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -m pip install --no-compile -r requirements.txt
if errorlevel 1 goto fail
if not exist "artifacts\models.pkl" (
  ".venv\Scripts\python.exe" train.py
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -m streamlit run app.py --global.developmentMode=false --server.port=8501 --browser.serverPort=8501 --browser.serverAddress=127.0.0.1 --server.headless=false
exit /b
:fail
echo Setup failed. Check the message above and README.md.
pause
exit /b 1
