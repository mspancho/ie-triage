#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# IE Triage — Full Analysis Pipeline
#
# Runs the complete analysis end-to-end: cohort creation, feasibility,
# EDA, feature engineering, ML evaluation, and summary tables.
# All output is logged to analysis_report.txt.
#
# Usage:
#   bash run_all.sh [OPTIONS]
#
# Required:
#   --mimic-hosp DIR      Path to MIMIC-IV hosp directory
#   --mimic-ed DIR        Path to MIMIC-IV ED directory
#
# Optional:
#   --conda-env ENV       Conda environment name (default: ie-triage)
#   --partition-gpu PART  Slurm GPU partition (default: gpu)
#   --partition-cpu PART  Slurm CPU partition (default: batch)
#   --gres GRES           GPU resource spec (default: gpu:1)
#   --cpus-gpu N          CPUs for GPU jobs (default: 4)
#   --cpus-cpu N          CPUs for CPU jobs (default: 16)
#   --mem MEM             Memory per job (default: 32G)
#   --time-feat TIME      Time limit for feature engineering (default: 02:00:00)
#   --time-ml TIME        Time limit for ML pipeline (default: 00:30:00)
#   --local               Run locally instead of via Slurm
#═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Project root (directory containing this script) ──────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
MIMIC_HOSP=""
MIMIC_ED=""
CONDA_ENV="ie-triage"
PARTITION_GPU="gpu"
PARTITION_CPU="batch"
GRES="gpu:1"
CPUS_GPU=4
CPUS_CPU=16
MEM="32G"
TIME_FEAT="02:00:00"
TIME_ML="00:30:00"
LOCAL=false

# ── Derived paths ────────────────────────────────────────────────────────────
CSV_DIR="${PROJECT_ROOT}/data/csv"
NPZ_DIR="${PROJECT_ROOT}/data/npz"
ML_DIR="${PROJECT_ROOT}/results/ml"
EDA_DIR="${PROJECT_ROOT}/results/eda"
LOG_DIR="${PROJECT_ROOT}/logs"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mimic-hosp)     MIMIC_HOSP="$2"; shift 2 ;;
        --mimic-ed)       MIMIC_ED="$2"; shift 2 ;;
        --conda-env)      CONDA_ENV="$2"; shift 2 ;;
        --partition-gpu)  PARTITION_GPU="$2"; shift 2 ;;
        --partition-cpu)  PARTITION_CPU="$2"; shift 2 ;;
        --gres)           GRES="$2"; shift 2 ;;
        --cpus-gpu)       CPUS_GPU="$2"; shift 2 ;;
        --cpus-cpu)       CPUS_CPU="$2"; shift 2 ;;
        --mem)            MEM="$2"; shift 2 ;;
        --time-feat)      TIME_FEAT="$2"; shift 2 ;;
        --time-ml)        TIME_ML="$2"; shift 2 ;;
        --local)          LOCAL=true; shift ;;
        -h|--help)
            head -28 "$0" | tail -26
            exit 0 ;;
        *) echo "Error: unknown option '$1'"; exit 1 ;;
    esac
done

# ── Validate required args ───────────────────────────────────────────────────
if [[ -z "${MIMIC_HOSP}" || -z "${MIMIC_ED}" ]]; then
    echo "Error: --mimic-hosp and --mimic-ed are required."
    echo "Usage: bash run_all.sh --mimic-hosp /path/to/hosp --mimic-ed /path/to/ed [OPTIONS]"
    exit 1
fi

# ── Create output directories ────────────────────────────────────────────────
mkdir -p "${CSV_DIR}" "${NPZ_DIR}" "${ML_DIR}" "${EDA_DIR}" "${LOG_DIR}"

# ── Log all output to analysis_report.txt ────────────────────────────────────
REPORT="${PROJECT_ROOT}/analysis_report.txt"
exec > >(tee "${REPORT}") 2>&1

echo "═══════════════════════════════════════════════════════════════════"
echo "  IE Triage — Full Analysis Pipeline"
echo "═══════════════════════════════════════════════════════════════════"
echo "  MIMIC-IV hosp:  ${MIMIC_HOSP}"
echo "  MIMIC-IV ED:    ${MIMIC_ED}"
echo "  Conda env:      ${CONDA_ENV}"
echo "  Local mode:     ${LOCAL}"
echo "  Project root:   ${PROJECT_ROOT}"
echo "═══════════════════════════════════════════════════════════════════"

# ── Step 1: Feasibility analysis ─────────────────────────────────────────────
echo ""
echo "[Step 1/6] Feasibility analysis..."
python "${PROJECT_ROOT}/src/pipeline/proj_feasibility.py" \
    --mimic-hosp "${MIMIC_HOSP}" \
    --mimic-ed "${MIMIC_ED}" \
    --output-dir "${CSV_DIR}"

# ── Step 2: Cohort creation ──────────────────────────────────────────────────
echo ""
echo "[Step 2/6] Cohort creation..."
python "${PROJECT_ROOT}/src/pipeline/cohort_creation.py" \
    --mimic-hosp "${MIMIC_HOSP}" \
    --mimic-ed "${MIMIC_ED}" \
    --output-dir "${CSV_DIR}"

COHORT_CSV="${CSV_DIR}/proj1_ie_triage_cohort.csv"

# ── Step 3: Table 1 ─────────────────────────────────────────────────────────
echo ""
echo "[Step 3/6] Generating Table 1..."
python "${PROJECT_ROOT}/src/pipeline/table1.py" \
    --cohort-csv "${COHORT_CSV}" \
    --csv-dir "${CSV_DIR}" \
    --plot-dir "${EDA_DIR}"

# ── Step 4: Demographic EDA ─────────────────────────────────────────────────
echo ""
echo "[Step 4/6] Demographic EDA (case vs. control)..."
python "${PROJECT_ROOT}/src/pipeline/demographic_eda.py" \
    --cohort-csv "${COHORT_CSV}" \
    --output-dir "${EDA_DIR}"

# ── Step 5: Feature engineering ──────────────────────────────────────────────
echo ""
echo "[Step 5/6] Feature engineering (4 options)..."
if [[ "${LOCAL}" == true ]]; then
    python "${PROJECT_ROOT}/src/pipeline/feature_engineering.py" \
        --cohort-csv "${COHORT_CSV}" \
        --output-dir "${NPZ_DIR}"
else
    echo "  Submitting Slurm job (waiting for completion)..."
    sbatch --wait \
        --partition="${PARTITION_GPU}" \
        --gres="${GRES}" \
        --cpus-per-task="${CPUS_GPU}" \
        --mem="${MEM}" \
        --time="${TIME_FEAT}" \
        --output="${LOG_DIR}/%x_%j.out" \
        --error="${LOG_DIR}/%x_%j.err" \
        --job-name=ie_feat_4opt \
        --wrap="module load anaconda3/2023.09-0-aqbc && eval \"\$(conda shell.bash hook)\" && conda activate ${CONDA_ENV} && \
                python ${PROJECT_ROOT}/src/pipeline/feature_engineering.py \
                    --cohort-csv ${COHORT_CSV} \
                    --output-dir ${NPZ_DIR}"
    echo "  Slurm job finished."
fi

# ── Step 6: ML pipeline ─────────────────────────────────────────────────────
echo ""
echo "[Step 6/6] ML pipeline..."
if [[ "${LOCAL}" == true ]]; then
    python "${PROJECT_ROOT}/src/pipeline/ml_pipeline.py" \
        --data-dir "${NPZ_DIR}" \
        --output-dir "${ML_DIR}"
else
    echo "  Submitting Slurm job (waiting for completion)..."
    sbatch --wait \
        --partition="${PARTITION_CPU}" \
        --cpus-per-task="${CPUS_CPU}" \
        --mem="${MEM}" \
        --time="${TIME_ML}" \
        --output="${LOG_DIR}/%x_%j.out" \
        --error="${LOG_DIR}/%x_%j.err" \
        --job-name=ie_ml_pipe \
        --wrap="module load anaconda3/2023.09-0-aqbc && eval \"\$(conda shell.bash hook)\" && conda activate ${CONDA_ENV} && \
                python ${PROJECT_ROOT}/src/pipeline/ml_pipeline.py \
                    --data-dir ${NPZ_DIR} \
                    --output-dir ${ML_DIR}"
    echo "  Slurm job finished."
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  Cohort & summaries: ${CSV_DIR}/"
echo "  EDA plots & tables: ${EDA_DIR}/"
echo "  Feature matrices:   ${NPZ_DIR}/"
echo "  ML results & plots: ${ML_DIR}/"
echo "  Full report:        ${REPORT}"
echo "═══════════════════════════════════════════════════════════════════"
