#!/usr/bin/env bash
# =============================================================================
#  MotionAGFormer — One-Command Setup & Run (Linux/macOS)
#
#  This script:
#    1. Creates a conda environment with Python 3.8 + CUDA + all dependencies
#    2. Downloads the datasets (Human3.6M and MPI-INF-3DHP)
#    3. Trains all 4 model sizes (xsmall, small, base, large) on both datasets
#    4. Evaluates all models and computes 6 temporal metrics
#    5. Saves a consolidated results report to results/full_results.json
#
#  Prerequisites:
#    - conda or miniconda installed (https://docs.conda.io/en/latest/miniconda.html)
#    - NVIDIA GPU with drivers installed (nvidia-smi should work)
#    - ~50 GB disk space for datasets + checkpoints
#
#  Usage:
#    chmod +x setup_and_run.sh
#    ./setup_and_run.sh                              # run everything
#    ./setup_and_run.sh --skip-download              # skip dataset download
#    ./setup_and_run.sh --eval-only                  # only evaluate (skip training)
#    ./setup_and_run.sh --sizes xsmall base          # only specific model sizes
#    ./setup_and_run.sh --datasets h36m              # only Human3.6M
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="motionagformer"
PYTHON_VER="3.8"

# ─── Colors for output ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ─── Step 0: Check prerequisites ─────────────────────────────────────────────
echo ""
echo "================================================================="
echo "  MotionAGFormer — Full Pipeline Setup & Run"
echo "================================================================="
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    error "conda is not installed or not in PATH."
    echo "  Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    echo "  Then re-run this script."
    exit 1
fi
ok "conda found: $(conda --version)"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    ok "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    warn "nvidia-smi not found. Training will use CPU (very slow)."
fi

# ─── Step 1: Create conda environment ────────────────────────────────────────
echo ""
echo "================================================================="
echo "  Step 1: Setting up conda environment '${ENV_NAME}'"
echo "================================================================="

if conda env list | grep -q "^${ENV_NAME} "; then
    info "Environment '${ENV_NAME}' already exists."
    read -p "  Recreate it? (y/N): " -r REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        info "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
        info "Creating environment from environment.yml..."
        conda env create -f environment.yml
    else
        info "Using existing environment."
    fi
else
    info "Creating environment from environment.yml..."
    conda env create -f environment.yml
fi

ok "Environment '${ENV_NAME}' is ready."

# ─── Step 2: Activate and run the pipeline ───────────────────────────────────
echo ""
echo "================================================================="
echo "  Step 2: Running the pipeline"
echo "================================================================="
echo ""

# Use conda run to execute inside the environment (avoids conda activate issues in scripts)
info "Launching run_pipeline.py with arguments: $@"
echo ""

conda run --no-capture-output -n "$ENV_NAME" python run_pipeline.py "$@"

echo ""
ok "Pipeline complete! Results are in: ${SCRIPT_DIR}/results/"
echo ""
