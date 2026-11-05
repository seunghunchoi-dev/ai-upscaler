@echo off
rem 설치 파일 다시 만들기 (Claude 없이 실행 가능, 약 10분)
rem 1) vup\__init__.py 의 __version__ 을 올린 뒤  2) 이 파일을 더블클릭
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build_installer.ps1
if errorlevel 1 (echo. & echo 빌드 실패 - 위 오류 내용을 확인하세요) else (echo. & echo 완료 - dist\installer 폴더를 확인하세요 & start "" explorer dist\installer)
pause
