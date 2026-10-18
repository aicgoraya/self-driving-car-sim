@echo off
REM Stop any running driver, start the trained network as the driver. In the simulator pick Autonomous Mode.
cd /d "%~dp0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :4567 ^| findstr LISTENING') do taskkill /f /pid %%a
tasklist | find /i "beta_simulator.exe" >nul || start "" "sim\beta_simulator_windows\beta_simulator.exe"
call venv\Scripts\activate.bat
python car\drive.py model.pt --speed 12 > drive_log.txt 2>&1
