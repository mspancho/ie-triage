#!/bin/bash
#SBATCH --job-name=ie_ml_pipe
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
module purge
source ~/.bashrc
conda activate symile-env

# ── Ensure dependencies ──────────────────────────────────────────────────────
pip install --upgrade xgboost scikit-learn matplotlib >/dev/null 2>&1

# ── Ensure log directory exists ──────────────────────────────────────────────
mkdir -p /users/mspancho/Downloads/proj1_ie_triage/logs

# ── Run pipeline ─────────────────────────────────────────────────────────────
cd /users/mspancho/Downloads/proj1_ie_triage
python ml_pipeline.py
