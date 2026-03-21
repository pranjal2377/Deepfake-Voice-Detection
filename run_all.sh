#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# Deepfake Voice Detection — One-Command Setup & Demo
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Steps:
#   1. Create / activate virtual environment
#   2. Install Python dependencies (CPU-only PyTorch)
#   3. Generate demo audio data
#   4. Train the CNN model
#   5. Evaluate on test split
#   6. Launch the Streamlit dashboard
# ──────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# ── Colours ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

step() { echo -e "\n${CYAN}═══ $1 ═══${NC}\n"; }

# ── 1. Virtual environment ────────────────────────────────
step "Step 1/6 — Virtual Environment"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}→ Virtual environment already exists${NC}"
fi

# ── 2. Install dependencies ──────────────────────────────
step "Step 2/6 — Installing Dependencies"
"$PIP" install --upgrade pip -q
"$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q
"$PIP" install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── 3. Generate demo data ────────────────────────────────
step "Step 3/6 — Generating Demo Data"
"$PYTHON" scripts/prepare_dataset.py --generate-demo
echo -e "${GREEN}✓ Demo data ready${NC}"

# ── 4. Train model ───────────────────────────────────────
step "Step 4/6 — Training Model"
"$PYTHON" scripts/train.py --epochs 20
echo -e "${GREEN}✓ Model trained${NC}"

# ── 5. Evaluate ──────────────────────────────────────────
step "Step 5/6 — Evaluating Model"
"$PYTHON" scripts/evaluate.py
echo -e "${GREEN}✓ Evaluation complete${NC}"

# ── 6. Run tests ─────────────────────────────────────────
step "Step 6/6 — Running Tests"
"$PYTHON" -m pytest tests/ -v --tb=short
echo -e "${GREEN}✓ All tests passed${NC}"

# ── Done ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo ""
echo -e "To launch the dashboard:"
echo -e "  ${CYAN}source $VENV_DIR/bin/activate${NC}"
echo -e "  ${CYAN}streamlit run src/ui/dashboard.py${NC}"
echo ""
echo -e "To run real-time detection:"
echo -e "  ${CYAN}$PYTHON scripts/realtime.py --mode demo${NC}"
echo ""
