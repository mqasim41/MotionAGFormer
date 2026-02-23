@echo off
REM =============================================================================
REM  MotionAGFormer — One-Command Setup ^& Run (Windows)
REM
REM  This script:
REM    1. Creates a conda environment with Python 3.8 + CUDA + all dependencies
REM    2. Downloads the datasets (Human3.6M and MPI-INF-3DHP)
REM    3. Trains all 4 model sizes (xsmall, small, base, large) on both datasets
REM    4. Evaluates all models and computes 6 temporal metrics
REM    5. Saves a consolidated results report to results\full_results.json
REM
REM  Prerequisites:
REM    - conda or miniconda installed (https://docs.conda.io/en/latest/miniconda.html)
REM    - NVIDIA GPU with drivers installed (nvidia-smi should work)
REM    - ~50 GB disk space for datasets + checkpoints
REM
REM  Usage:
REM    setup_and_run.bat                              # run everything
REM    setup_and_run.bat --skip-download              # skip dataset download
REM    setup_and_run.bat --eval-only                  # only evaluate (skip training)
REM    setup_and_run.bat --sizes xsmall base          # only specific model sizes
REM    setup_and_run.bat --datasets h36m              # only Human3.6M
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "ENV_NAME=motionagformer"

echo.
echo =================================================================
echo   MotionAGFormer — Full Pipeline Setup ^& Run
echo =================================================================
echo.

REM ─── Check conda ───────────────────────────────────────────────────────────
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda is not installed or not in PATH.
    echo   Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
    echo   Then re-run this script.
    exit /b 1
)
echo [OK] conda found.

REM ─── Check GPU ─────────────────────────────────────────────────────────────
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARN] nvidia-smi not found. Training will use CPU (very slow).
) else (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
)

REM ─── Create conda environment ─────────────────────────────────────────────
echo.
echo =================================================================
echo   Step 1: Setting up conda environment '%ENV_NAME%'
echo =================================================================

conda env list | findstr /B "%ENV_NAME% " >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Environment '%ENV_NAME%' already exists. Using it.
    echo   To recreate, run: conda env remove -n %ENV_NAME% -y
    echo   Then re-run this script.
) else (
    echo [INFO] Creating environment from environment.yml...
    conda env create -f environment.yml
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)
echo [OK] Environment '%ENV_NAME%' is ready.

REM ─── Run the pipeline ─────────────────────────────────────────────────────
echo.
echo =================================================================
echo   Step 2: Running the pipeline
echo =================================================================
echo.

echo [INFO] Launching run_pipeline.py with arguments: %*

conda run --no-capture-output -n %ENV_NAME% python run_pipeline.py %*

echo.
echo [OK] Pipeline complete! Results are in: %SCRIPT_DIR%results\
echo.

endlocal
