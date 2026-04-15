# IE Triage

Predicting infective endocarditis (IE) from emergency department triage data using machine learning. This project builds a validated cohort from MIMIC-IV, engineers features (including BioClinical-ModernBERT chief complaint embeddings), and evaluates multiple classifiers for early IE detection.

## Description

Infective endocarditis is a life-threatening cardiac infection that is difficult to diagnose at ED triage. This project implements an end-to-end pipeline that:

1. Extracts and validates an IE cohort from MIMIC-IV ED and hospital records
2. Engineers feature representations using clinical NLP embeddings, vitals, and demographics
3. Trains and evaluates Logistic Regression, Random Forest, and XGBoost models with stratified cross-validation

The goal is to assess whether ED triage data alone can flag patients at risk for IE, supporting earlier diagnostic workup.

## Programs

| File | Description |
|------|-------------|
| `cohort_creation.py` | Builds the IE triage cohort from MIMIC-IV: identifies IE-positive cases by ICD code, samples hard-negative controls across three clinical strata, validates vital signs, and outputs the labeled dataset. |
| `proj_feasibility.py` | Preliminary feasibility analysis that estimates IE prevalence in the ED and assesses whether the MIMIC-IV data supports the study design. |
| `feature_engineering.py` | Transforms the cohort into a single feature matrix with BioClinical-ModernBERT `[CLS]` embeddings of chief complaints, ordinal ESI acuity, one-hot demographics, and float vitals. |
| `feature_engineering_3options.py` | Extended feature engineering that produces three alternative representations: (1) categorical chief complaints, (2) PCA-reduced BERT embeddings, and (3) a hybrid of both. |
| `ml_pipeline.py` | Runs Logistic Regression (L1), Random Forest, and XGBoost over all feature sets with 5-fold stratified CV; reports AUROC, AUPRC, sensitivity, specificity, and generates ROC, PR, and decision curve plots. |
| `table1.py` | Generates Table 1 descriptive statistics (means, medians, counts, percentages) stratified by IE label, following Hayes-Larson et al. (2019) guidelines. |
| `demographics.py` | Utility script for consolidating race categories and computing demographic summary statistics on a general IE cohort. |
| `demographics_proj1.py` | Project-specific demographic analysis on the IE triage cohort, producing stratified summaries by IE label. |
| `run_feature_engineering_3options.sh` | Slurm batch script that runs `feature_engineering_3options.py` on an Oscar GPU node (1 GPU, 4 CPUs, 32 GB RAM). |
| `run_ml_pipeline.sh` | Slurm batch script that runs `ml_pipeline.py` on an Oscar CPU node (16 CPUs, 32 GB RAM). |

## Installation

Requires Python 3.10+ and access to MIMIC-IV on Oscar (Brown CCV).

```bash
conda create -n ie-triage python=3.11
conda activate ie-triage
pip install pandas numpy torch transformers scikit-learn xgboost matplotlib
```

The scripts expect MIMIC-IV data at:
- `/oscar/data/shared/ursa/mimic-iv/hosp/3.1`
- `/oscar/data/shared/ursa/mimic-iv/ed/2.2`

Update `MIMIC_HOSP` and `MIMIC_ED` in `cohort_creation.py` if your paths differ.

## Usage

### 1. Create the cohort

```bash
python cohort_creation.py
```

### 2. Engineer features (GPU recommended)

```bash
sbatch run_feature_engineering_3options.sh
```

Or locally:

```bash
python feature_engineering_3options.py
```

### 3. Train and evaluate models

```bash
sbatch run_ml_pipeline.sh
```

Or locally:

```bash
python ml_pipeline.py
```

### 4. Generate summary tables

```bash
python table1.py
python demographics_proj1.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)
