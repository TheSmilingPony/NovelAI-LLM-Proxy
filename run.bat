@echo off
REM Activate the virtual environment
call venv\Scripts\activate

REM Run the inference script
uvicorn proxy:app --host 0.0.0.0 --port 8001

REM Deactivate the virtual environment
deactivate