# IE Triage

Predicting infective endocarditis (IE) from emergency department triage data using machine learning. This project builds a validated cohort from MIMIC-IV, engineers features (including BioClinical-ModernBERT chief complaint embeddings), evaluates multiple classifiers for early IE detection, and externally validates the categorical chief-complaint Random Forest on the [MC-MED](https://physionet.org/content/mc-med/1.0.1/) dataset.

## Description

Infective endocarditis is a life-threatening cardiac infection that is difficult to diagnose at ED triage. This project implements an end-to-end pipeline that:

1. Extracts and validates an IE cohort from MIMIC-IV ED and hospital records
2. Engineers feature representations using clinical NLP embeddings, vitals, and demographics
3. Trains and evaluates Logistic Regression, Random Forest, and XGBoost models with stratified cross-validation
4. Visualizes feature importances for each model and feature set
5. Externally validates the categorical-CC Random Forest on the MC-MED dataset, training on the full internal cohort and evaluating on the held-out external cohort

The goal is to assess whether ED triage data alone can flag patients at risk for IE — including under distribution shift across institutions — supporting earlier diagnostic workup.

## Directory Structure

```
ie-triage/
├── run_all.sh                 # Master script — runs the full pipeline
├── README.md
├── LICENSE
├── .gitignore
├── env/
│   ├── environment.yml        # Conda environment specification
│   └── requirements.txt       # Pip requirements
├── data/
│   ├── csv/                   # Internal & external cohort CSVs and summary tables
│   └── npz/                   # Feature matrices (.npz) — internal 4 options + external Option 1
├── results/
│   ├── eda/                   # EDA plots, Table 1, demographic comparisons
│   ├── ml/                    # Internal ML evaluation plots (ROC, PR, DCA curves)
│   │   └── feature_importance/ # Feature importance bar charts per model/option
│   └── ext_val/               # External-validation results (MC-MED): metrics CSV, ROC/PR/DCA, feature importance
├── logs/                      # Slurm job logs
└── src/
    ├── pipeline/              # Internal analysis scripts (MIMIC-IV)
    │   ├── cohort_creation.py
    │   ├── proj_feasibility.py
    │   ├── feature_engineering.py
    │   ├── ml_pipeline.py
    │   ├── feature_importance.py
    │   ├── table1.py
    │   └── demographic_eda.py
    └── ext_val_pipeline/      # External-validation scripts (MC-MED)
        ├── cohort_creation.py
        ├── table2.py
        ├── feature_engineering.py
        ├── ml_pipeline.py
        └── feature_importance.py
```

## Programs

### Internal Pipeline (`src/pipeline/`)

| File | Description |
|------|-------------|
| `cohort_creation.py` | Builds the IE triage cohort from MIMIC-IV: identifies IE-positive cases by ICD code, samples hard-negative controls across three clinical strata, validates vital signs, and outputs the labeled dataset. |
| `proj_feasibility.py` | Preliminary feasibility analysis that estimates IE prevalence in the ED and assesses whether the MIMIC-IV data supports the study design. |
| `feature_engineering.py` | Feature engineering that produces four alternative representations: (1) categorical chief complaints, (2) PCA-reduced BERT embeddings, (3) a hybrid of both, and (4) no chief complaint features. |
| `ml_pipeline.py` | Runs Logistic Regression (L1), Random Forest, and XGBoost over all feature sets with 5-fold stratified CV; reports AUROC, AUPRC, sensitivity, specificity, and generates ROC, PR, and decision curve plots. |
| `feature_importance.py` | Computes mean ± SD feature importances across 5-fold CV for each (feature set, model) combination and saves horizontal bar charts of the top 20 features to `results/ml/feature_importance/`. Uses \|coefficient\| for Logistic Regression and Gini/gain importances for tree-based models. |
| `table1.py` | Generates Table 1 descriptive statistics (means, medians, counts, percentages) stratified by IE label, following Hayes-Larson et al. (2019) guidelines. |
| `demographic_eda.py` | Case vs. control demographic comparison using seaborn: age, gender, race, acuity, and vitals distributions with box plots, violin plots, bar charts, and correlation heatmaps. |

### External-Validation Pipeline (`src/ext_val_pipeline/`)

External validation uses the [MC-MED](https://physionet.org/content/mc-med/1.0.1/) dataset (Stanford ED). The exact same cohort-creation procedure (IE ICD code set, 10:1 control ratio, 40/35/25% A/B/C stratified hard-negative sampling, identical vital-sign validation ranges, identical race normalization) is applied to MC-MED's `visits.csv`, with `pmh.csv` providing per-patient prior diagnoses for stratum-A flagging (analog of MIMIC's `diagnoses_icd` lookup). MC-MED's Celsius temperatures are converted to Fahrenheit before validation. The Random Forest is trained on the **full** internal Option 1 (categorical chief-complaint) cohort using the **exact same hyperparameters** as the internal CV (`n_estimators=500`, `max_depth=None`, `class_weight="balanced"`, `random_state=42`) and evaluated once on the held-out external cohort.

| File | Description |
|------|-------------|
| `cohort_creation.py` | Builds the external-validation cohort from MC-MED following the same procedure as the internal cohort. Writes `proj1_ie_triage_extval_cohort.csv` and a stratum summary CSV. |
| `table2.py` | External-cohort analog of Table 1: same Hayes-Larson-style summary statistics stratified by IE label, applied to the MC-MED cohort. Reuses the internal `table1.py` statistic functions for exact format/methodology parity. Writes `data/csv/table2_output.csv` and `results/ext_val/table2_output.png`. |
| `feature_engineering.py` | Categorical CC encoding only — produces a feature matrix schema-aligned to the internal Option 1 npz (loads internal `feature_names`, fills 0 for missing dummy levels, drops levels not present internally, asserts column order matches). |
| `ml_pipeline.py` | Trains the RF on the full internal Option 1 cohort, predicts on the MC-MED matrix, writes external metrics (Sensitivity, Specificity, PPV, NPV, AUROC, AUPRC), per-visit predictions, and ROC / PR / Decision-Curve plots to `results/ext_val/`. |
| `feature_importance.py` | Single RF trained on the full internal Option 1 cohort; saves a top-20 feature importance bar chart for the model used in external validation. |

### Bash Scripts

| File | Description |
|------|-------------|
| `run_all.sh` | Master script that orchestrates the entire analysis pipeline (internal + external validation, 12 steps) with configurable arguments for MIMIC-IV / MC-MED paths, Slurm job specs, and conda environment. Handles Slurm submission for GPU/CPU steps automatically. |

## Installation

Requires Python 3.10+ and access to MIMIC-IV and MC-MED on Oscar (Brown CCV).

### Using conda (recommended)

```bash
conda env create -f env/environment.yml
conda activate ie-triage
```

### Using pip

```bash
conda create -n ie-triage python=3.11
conda activate ie-triage
pip install -r env/requirements.txt
```

## Usage

### Full pipeline (single command)

Run the entire analysis end-to-end (internal cohort, training, evaluation, and external validation on MC-MED). All output is logged to `analysis_report.txt`:

```bash
bash run_all.sh \
    --mimic-hosp /oscar/data/shared/ursa/mimic-iv/hosp/3.1 \
    --mimic-ed /oscar/data/shared/ursa/mimic-iv/ed/2.2 \
    --mc-med-dir /oscar/data/shared/ursa/mc-med/1.0.1/data \
    --local
```

For Slurm submission (GPU feature engineering, CPU ML pipeline):

```bash
bash run_all.sh \
    --mimic-hosp /oscar/data/shared/ursa/mimic-iv/hosp/3.1 \
    --mimic-ed /oscar/data/shared/ursa/mimic-iv/ed/2.2 \
    --mc-med-dir /oscar/data/shared/ursa/mc-med/1.0.1/data \
    --conda-env ie-triage \
    --partition-gpu gpu \
    --partition-cpu batch
```

### Step-by-step

```bash
# 1. Cohort creation
python src/pipeline/cohort_creation.py \
    --mimic-hosp /oscar/data/shared/ursa/mimic-iv/hosp/3.1 \
    --mimic-ed /oscar/data/shared/ursa/mimic-iv/ed/2.2 \
    --output-dir data/csv

# 2. Table 1 and demographic EDA
python src/pipeline/table1.py \
    --cohort-csv data/csv/proj1_ie_triage_cohort.csv \
    --csv-dir data/csv \
    --plot-dir results/eda

python src/pipeline/demographic_eda.py \
    --cohort-csv data/csv/proj1_ie_triage_cohort.csv \
    --output-dir results/eda

# 3. Feature engineering (GPU recommended)
python src/pipeline/feature_engineering.py \
    --cohort-csv data/csv/proj1_ie_triage_cohort.csv \
    --output-dir data/npz

# 4. ML evaluation
python src/pipeline/ml_pipeline.py \
    --data-dir data/npz \
    --output-dir results/ml

# 5. Feature importance plots
python src/pipeline/feature_importance.py \
    --data-dir data/npz \
    --output-dir results/ml/feature_importance

# 6. External validation (MC-MED)
python src/ext_val_pipeline/cohort_creation.py \
    --mc-med-dir /oscar/data/shared/ursa/mc-med/1.0.1/data \
    --output-dir data/csv

python src/ext_val_pipeline/table2.py \
    --cohort-csv data/csv/proj1_ie_triage_extval_cohort.csv \
    --csv-dir data/csv \
    --plot-dir results/ext_val

python src/ext_val_pipeline/feature_engineering.py \
    --cohort-csv data/csv/proj1_ie_triage_extval_cohort.csv \
    --internal-npz data/npz/features_option1_categorical.npz \
    --output-dir data/npz

python src/ext_val_pipeline/ml_pipeline.py \
    --internal-npz data/npz/features_option1_categorical.npz \
    --external-npz data/npz/extval_features_option1_categorical.npz \
    --output-dir results/ext_val

python src/ext_val_pipeline/feature_importance.py \
    --internal-npz data/npz/features_option1_categorical.npz \
    --output-dir results/ext_val
```

### `run_all.sh` options

| Option | Default | Description |
|--------|---------|-------------|
| `--mimic-hosp` | *(required)* | Path to MIMIC-IV hosp directory |
| `--mimic-ed` | *(required)* | Path to MIMIC-IV ED directory |
| `--mc-med-dir` | *(required)* | Path to MC-MED data directory (for external validation) |
| `--conda-env` | `ie-triage` | Conda environment name |
| `--partition-gpu` | `gpu` | Slurm GPU partition |
| `--partition-cpu` | `batch` | Slurm CPU partition |
| `--gres` | `gpu:1` | GPU resource specification |
| `--cpus-gpu` | `4` | CPUs for GPU jobs |
| `--cpus-cpu` | `16` | CPUs for CPU jobs |
| `--mem` | `32G` | Memory per job |
| `--time-feat` | `02:00:00` | Time limit for feature engineering |
| `--time-ml` | `00:30:00` | Time limit for ML pipeline |
| `--local` | `false` | Run locally instead of via Slurm |

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

The author(s) of this project thank Dr. Neil Sarkar for the opportunity to work on this problem as part of the final project assignment for *BIOL 1595: Artificial Intelligence in Healthcare*, offered during the Spring 2026 semester at Brown University.

## License

[MIT](LICENSE)
