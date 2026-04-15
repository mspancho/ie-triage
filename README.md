# IE Triage

Predicting infective endocarditis (IE) from emergency department triage data using machine learning. This project builds a validated cohort from MIMIC-IV, engineers features (including BioClinical-ModernBERT chief complaint embeddings), and evaluates multiple classifiers for early IE detection.

## Description

Infective endocarditis is a life-threatening cardiac infection that is difficult to diagnose at ED triage. This project implements an end-to-end pipeline that:

1. Extracts and validates an IE cohort from MIMIC-IV ED and hospital records
2. Engineers feature representations using clinical NLP embeddings, vitals, and demographics
3. Trains and evaluates Logistic Regression, Random Forest, and XGBoost models with stratified cross-validation

The goal is to assess whether ED triage data alone can flag patients at risk for IE, supporting earlier diagnostic workup.

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
│   ├── csv/                   # Cohort CSVs and summary tables
│   └── npz/                   # Feature matrices (.npz)
├── results/
│   ├── eda/                   # EDA plots, Table 1, demographic comparisons
│   └── ml/                    # ML evaluation plots (ROC, PR, DCA curves)
├── logs/                      # Slurm job logs
└── src/
    └── pipeline/              # Python analysis scripts
        ├── cohort_creation.py
        ├── proj_feasibility.py
        ├── feature_engineering.py
        ├── ml_pipeline.py
        ├── table1.py
        └── demographic_eda.py
```

## Programs

### Python Pipeline (`src/pipeline/`)

| File | Description |
|------|-------------|
| `cohort_creation.py` | Builds the IE triage cohort from MIMIC-IV: identifies IE-positive cases by ICD code, samples hard-negative controls across three clinical strata, validates vital signs, and outputs the labeled dataset. |
| `proj_feasibility.py` | Preliminary feasibility analysis that estimates IE prevalence in the ED and assesses whether the MIMIC-IV data supports the study design. |
| `feature_engineering.py` | Feature engineering that produces four alternative representations: (1) categorical chief complaints, (2) PCA-reduced BERT embeddings, (3) a hybrid of both, and (4) no chief complaint features. |
| `ml_pipeline.py` | Runs Logistic Regression (L1), Random Forest, and XGBoost over all feature sets with 5-fold stratified CV; reports AUROC, AUPRC, sensitivity, specificity, and generates ROC, PR, and decision curve plots. |
| `table1.py` | Generates Table 1 descriptive statistics (means, medians, counts, percentages) stratified by IE label, following Hayes-Larson et al. (2019) guidelines. |
| `demographic_eda.py` | Case vs. control demographic comparison using seaborn: age, gender, race, acuity, and vitals distributions with box plots, violin plots, bar charts, and correlation heatmaps. |

### Bash Scripts

| File | Description |
|------|-------------|
| `run_all.sh` | Master script that orchestrates the entire analysis pipeline with configurable arguments for MIMIC-IV paths, Slurm job specs, and conda environment. Handles Slurm submission for GPU/CPU steps automatically. |

## Installation

Requires Python 3.10+ and access to MIMIC-IV on Oscar (Brown CCV).

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

Run the entire analysis end-to-end. All output is logged to `analysis_report.txt`:

```bash
bash run_all.sh \
    --mimic-hosp /oscar/data/shared/ursa/mimic-iv/hosp/3.1 \
    --mimic-ed /oscar/data/shared/ursa/mimic-iv/ed/2.2 \
    --local
```

For Slurm submission (GPU feature engineering, CPU ML pipeline):

```bash
bash run_all.sh \
    --mimic-hosp /oscar/data/shared/ursa/mimic-iv/hosp/3.1 \
    --mimic-ed /oscar/data/shared/ursa/mimic-iv/ed/2.2 \
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
```

### `run_all.sh` options

| Option | Default | Description |
|--------|---------|-------------|
| `--mimic-hosp` | *(required)* | Path to MIMIC-IV hosp directory |
| `--mimic-ed` | *(required)* | Path to MIMIC-IV ED directory |
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

## License

[MIT](LICENSE)
