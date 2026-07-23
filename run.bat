@echo off
echo =========================================
echo   Uruchamianie Projektu Mini-GPT (Web)
echo =========================================

echo [1/4] Upewnianie sie, ze port 5000 jest wolny...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do (
    echo Zamykanie starego procesu na porcie 5000, PID: %%a...
    taskkill /F /PID %%a 2>nul
)
timeout /t 1 >nul

echo [2/4] Sprawdzanie i budowanie frontendu...
if not exist "app_frontend\dist" (
    echo [!] Brak skompilowanego frontendu. Sprawdzanie dostepnosci npm...
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [BLAD] Narzedzie npm - Node.js - nie jest zainstalowane w systemie!
        echo Skompiluj frontend recznie lub zainstaluj Node.js ze strony https://nodejs.org/
        pause
        exit /b 1
    )
    echo [!] Budowanie frontendu...
    cd app_frontend
    echo Instalowanie zaleznosci...
    call npm install
    if errorlevel 1 (
        echo [BLAD] Instalacja zaleznosci npm nie powiodla sie!
        cd ..
        pause
        exit /b 1
    )
    echo Kompilacja kodu produkcyjnego...
    call npm run build
    if errorlevel 1 (
        echo [BLAD] Kompilacja frontendu - npm run build - nie powiodla sie!
        cd ..
        pause
        exit /b 1
    )
    cd ..
)

echo [3/4] Konfiguracja srodowiska Pythona...
set PYTHON_CMD=python
if exist ".venv\Scripts\activate.bat" (
    echo Aktywowanie srodowiska wirtualnego .venv...
    call .venv\Scripts\activate.bat
    set PYTHON_CMD=python
) else if exist "venv\Scripts\activate.bat" (
    echo Aktywowanie srodowiska wirtualnego venv...
    call venv\Scripts\activate.bat
    set PYTHON_CMD=python
)

echo [4/4] Otwieranie przegladarki w tle (z opoznieniem)...
start /b cmd /c "timeout /t 4 >nul && start http://127.0.0.1:5000"

echo Uruchamianie serwera Flask...
%PYTHON_CMD% app.py

pause


