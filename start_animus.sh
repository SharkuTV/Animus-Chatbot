#!/bin/bash

# Navigate to the directory where this script is located
cd "$(dirname "$0")"

echo "Starting Animus server..."

# Run Uvicorn via the virtual environment to keep the backend live on port 8001
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8001 --reload
