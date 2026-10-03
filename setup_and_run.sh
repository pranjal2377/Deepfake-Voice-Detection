#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV=".venv"
PYTHON="$VENV/bin/python"

echo "=============================================="
echo " Deepfake Voice & Scam Detection"
echo " Setup / Run"
echo "=============================================="

if [ ! -d "$VENV" ]; then
  echo "[1/4] Creating virtual environment..."
  python3 -m venv "$VENV"
else
  echo "[1/4] Virtual environment already exists."
fi

echo "[2/4] Installing/updating dependencies..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
"$PYTHON" -m pip install -r requirements.txt

echo "[3/4] Setup complete."
echo
echo "Choose an action:"
echo "  1) Web Dashboard"
echo "  2) Real-time microphone"
echo "  3) Demo analysis"
echo "  4) Native desktop wrapper"
echo "  5) Mobile PWA wrapper"
echo "  6) Train model"
echo "  7) Evaluate model"
echo "  8) Run tests"
echo "  9) Exit"
echo

read -r -p "Enter choice [1-9]: " choice

case "$choice" in
  1) exec "$PYTHON" -m streamlit run src/ui/dashboard.py ;;
  2) exec "$PYTHON" scripts/realtime.py --mode mic ;;
  3) exec "$PYTHON" scripts/realtime.py --mode demo ;;
  4) exec "$PYTHON" desktop_app.py ;;
  5) exec "$PYTHON" run_mobile_app.py ;;
  6) exec "$PYTHON" scripts/train.py ;;
  7) exec "$PYTHON" scripts/evaluate.py ;;
  8) exec "$PYTHON" -m pytest tests/ -v --tb=short ;;
  9) exit 0 ;;
  *) echo "Invalid choice."; exit 1 ;;
esac
