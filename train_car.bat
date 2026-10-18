@echo off
REM Train the steering network on every recorded folder under data\
cd /d "%~dp0"
call venv\Scripts\activate.bat
python car\train.py --epochs 15 > train_log.txt 2>&1
echo TRAIN_FINISHED>> train_log.txt
