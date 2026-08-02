@echo off
REM Teste rapido do pipeline (sem validacao cruzada e sem radiomica).
chcp 65001 >nul
cd /d "%~dp0"
"D:\Projetos\FIAP\.venv\Scripts\python.exe" "%~dp0executar_projeto.py" --rapido --sem-radiomica %*
echo.
pause
