@echo off
REM One-time setup: Python 3.11 venv, PyTorch with CUDA, libraries, and the Udacity simulator.
cd /d "%~dp0"
py -3.11 --version || winget install -e --id Python.Python.3.11
py -3.11 -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r car\requirements.txt
if not exist sim powershell -NoProfile -Command "Invoke-WebRequest https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip -OutFile sim.zip; Expand-Archive sim.zip -DestinationPath sim; Remove-Item sim.zip"
python -c "import torch; print('GPU:', torch.cuda.is_available())"
pause
