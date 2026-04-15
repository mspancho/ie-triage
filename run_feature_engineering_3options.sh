#!/bin/bash
#SBATCH --job-name=ie_feat_3opt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load cuda cudnn
source ~/.bashrc
conda activate symile-env

# ── Ensure dependencies are up to date ────────────────────────────────────────
pip install --upgrade transformers scikit-learn >/dev/null 2>&1

# ── Ensure log directory exists ────────────────────────────────────────────
mkdir -p /users/mspancho/Downloads/proj1_ie_triage/logs

# ── Run pipeline ─────────────────────────────────────────────────────────────
cd /users/mspancho/Downloads/proj1_ie_triage
python feature_engineering_3options.py
