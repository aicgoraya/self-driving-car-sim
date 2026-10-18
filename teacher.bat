@echo off
REM Stop any running driver, then start the teacher (records to data\teacher). Simulator must be open.
cd /d "%~dp0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :4567 ^| findstr LISTENING') do taskkill /f /pid %%a
if exist data\teacher if not exist old_runs mkdir old_runs
if exist data\teacher move data\teacher old_runs\teacher_%RANDOM%
call venv\Scripts\activate.bat
python car\expert_drive.py data\teacher --speed 9 --frames 6000 > teacher_log.txt 2>&1
