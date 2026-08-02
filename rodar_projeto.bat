@echo off
REM Roda o projeto INTEIRO (testes + treino + previsao + radiomica).
chcp 65001 >nul
cd /d "%~dp0"
"D:\Projetos\FIAP\.venv\Scripts\python.exe" "%~dp0executar_projeto.py" %*
echo.
pause
