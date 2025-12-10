@echo off
cd /d %~dp0
git add -A
for /f "delims=" %%i in ('powershell -command "Get-Date -Format o"') do set MSG=%%i
git commit -m "Auto-update: %MSG%" || echo No changes to commit
git push origin main
