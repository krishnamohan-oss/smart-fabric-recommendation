@echo off
title Smart Fabric Recommendation System
echo =========================================================
echo    Smart Fabric Recommendation System - Launching
echo =========================================================
echo.
cd /d "%~dp0"

where streamlit >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    streamlit run app.py
    goto end
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -m streamlit run app.py
    goto end
)

if exist "C:\Users\krish\Downloads\FOSSEE\KiCad\bin\python.exe" (
    "C:\Users\krish\Downloads\FOSSEE\KiCad\bin\python.exe" -m streamlit run app.py
    goto end
)

echo [ERROR] Neither streamlit nor python was found on PATH.
pause

:end
