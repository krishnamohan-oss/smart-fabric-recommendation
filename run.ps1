Write-Host "=========================================================" -ForegroundColor Green
Write-Host "   Smart Fabric Recommendation System - Launching" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Set-Location -Path $PSScriptRoot

if (Get-Command streamlit -ErrorAction SilentlyContinue) {
    streamlit run app.py
} elseif (Test-Path "C:\Users\krish\Downloads\FOSSEE\KiCad\bin\python.exe") {
    & "C:\Users\krish\Downloads\FOSSEE\KiCad\bin\python.exe" -m streamlit run app.py
} else {
    python -m streamlit run app.py
}
