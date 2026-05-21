#!/bin/bash
set -euo pipefail

echo "Setting up RAG Backend API..."

# Use python3 so this works on macOS/Linux where `python` may be missing.
PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Error: no Python 3 found. Install Python 3.8+ or set PY=/path/to/python3"
  exit 1
fi

# Create virtual environment (always use the venv's pip — avoids pip/python mismatch)
"$PY" -m venv venv

# Install dependencies into THIS venv only (do not rely on global `pip`)
if [ -f "venv/Scripts/python.exe" ]; then
  VENV_PY="venv/Scripts/python.exe"
else
  VENV_PY="venv/bin/python"
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p users_data

echo "Setup complete! Start the server with:"
echo "  ./venv/bin/python app.py"