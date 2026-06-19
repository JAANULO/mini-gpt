@echo off
echo =========================================
echo   Uruchamianie Projektu Mini-GPT (Web)
echo =========================================

echo [1/3] Upewnianie sie, ze nie ma starych procesow blokujacych port...
taskkill /F /IM python.exe 2>nul
timeout /t 1 >nul

echo [2/3] Otwieranie przegladarki...
start http://127.0.0.1:5000

echo [3/3] Uruchamianie serwera (Flask + React)...
python app.py

pause
