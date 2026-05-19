#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the inference script
uvicorn proxy:app --host 0.0.0.0 --port 8001

# Deactivate the virtual environment
deactivate