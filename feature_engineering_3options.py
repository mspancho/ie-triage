"""
Feature engineering pipeline with 3 chief complaint representation options.

Reads proj1_ie_triage_cohort.csv and produces 3 separate feature matrices:
  1. Option 1: Categorical chief complaints (10-15 categories) + demographics + vitals
  2. Option 2: Full BERT embeddings + PCA reduction (100 dims) + other features
  3. Option 3: Hybrid (categorical + PCA embeddings + other features)

Output files:
  - features_option1_categorical.npz
  - features_option2_pca_embeddings.npz
  - features_option3_hybrid.npz
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from transformers import AutoTokenizer, AutoModel

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_DIR = "/users/mspancho/Downloads/proj1_ie_triage"
OUTPUT_DIR = INPUT_DIR
COHORT_CSV = os.path.join(INPUT_DIR, "proj1_ie_triage_cohort.csv")

CC_MODEL_PATH = "thomas-sounack/BioClinical-ModernBERT-base"

BATCH_SIZE = 64
MAX_SEQ_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PCA_N_COMPONENTS = 100

# ── 0. Load cohort ──────────────────────────────────────────────────────────
print("Loading cohort …")
cohort = pd.read_csv(COHORT_CSV)
print(f"  rows={len(cohort)}  cols={list(cohort.columns)}")

# ── 0.5. Impute missing values (statistically rigorous) ──────────────────────
print("Imputing missing values …")

# Define variable types
continuous_cols = ["anchor_age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
ordinal_cols = ["acuity"]
categorical_cols = ["gender", "race"]

# Log missing counts before imputation
missing_before = cohort[continuous_cols + ordinal_cols + categorical_cols].isna().sum()
if missing_before.sum() > 0:
    print(f"  Missing values before imputation:")
    for col in missing_before[missing_before > 0].index:
        print(f"    {col}: {missing_before[col]}")
else:
    print("  No missing values found.")

# Continuous variables: KNNImputer (k=5, uses nearest neighbors for robust imputation)
if cohort[continuous_cols].isna().sum().sum() > 0:
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    cohort[continuous_cols] = knn_imputer.fit_transform(cohort[continuous_cols])
    print(f"  Applied KNNImputer (k=5) to {len(continuous_cols)} continuous variables")

# Ordinal variable (acuity): SimpleImputer with median
if cohort[ordinal_cols].isna().sum().sum() > 0:
    ordinal_imputer = SimpleImputer(strategy='median')
    cohort[ordinal_cols] = ordinal_imputer.fit_transform(cohort[ordinal_cols])
    print(f"  Applied median imputation to {len(ordinal_cols)} ordinal variable(s)")

# Categorical variables: SimpleImputer with most_frequent
if cohort[categorical_cols].isna().sum().sum() > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cohort[categorical_cols] = cat_imputer.fit_transform(cohort[categorical_cols])
    print(f"  Applied most_frequent imputation to {len(categorical_cols)} categorical variable(s)")

# Verify no remaining NaN values
remaining_nan = cohort[continuous_cols + ordinal_cols + categorical_cols].isna().sum().sum()
print(f"  After imputation: {remaining_nan} remaining NaN values ✓")

# ── 1. Generate BERT embeddings (for Options 2 & 3) ──────────────────────────
print(f"Computing chief-complaint embeddings (model={CC_MODEL_PATH}) …")

tokenizer = AutoTokenizer.from_pretrained(CC_MODEL_PATH)
bert_model = AutoModel.from_pretrained(CC_MODEL_PATH)
bert_model.to(DEVICE)
bert_model.eval()

texts = cohort["chiefcomplaint"].fillna("").astype(str).tolist()
all_embeddings = []

for start in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[start : start + BATCH_SIZE]
    encoded = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = bert_model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    all_embeddings.append(cls_emb)

    if (start // BATCH_SIZE) % 20 == 0:
        print(f"  embedded {start + len(batch_texts)}/{len(texts)}")

cc_embeddings_768 = np.concatenate(all_embeddings, axis=0).astype(np.float32)
print(f"  embedding shape: {cc_embeddings_768.shape}")

# Free GPU memory
del bert_model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── 2. PCA Reduction (for Options 2 & 3) ────────────────────────────────────
print(f"Applying PCA to reduce {cc_embeddings_768.shape[1]} → {PCA_N_COMPONENTS} dims …")
pca = PCA(n_components=PCA_N_COMPONENTS)
cc_embeddings_pca = pca.fit_transform(cc_embeddings_768).astype(np.float32)
explained_var = pca.explained_variance_ratio_.sum()
print(f"  explained variance: {explained_var:.4f}")
cc_pca_feature_names = [f"cc_pca_{i}" for i in range(PCA_N_COMPONENTS)]

# ── 3. Categorical Chief Complaint Encoding (for Options 1 & 3) ──────────────
print("Categorical encoding chief complaints …")

# Pre-defined keyword patterns for ED triage categories
categories = {
    "fever": r"\b(fever|febrile|temperature|elevated temp)\b",
    "cough": r"\b(cough|coughing|productive)\b",
    "chest_pain": r"\b(chest pain|chest discomfort|cp|angina)\b",
    "abd_pain": r"\b(abdominal pain|abd pain|belly|stomach pain)\b",
    "dyspnea": r"\b(shortness of breath|dyspnea|sob|breathless|wheezing)\b",
    "weakness": r"\b(weakness|weak|fatigue|tired)\b",
    "altered_mental": r"\b(altered mental status|confusion|confused|altered|dementia|delirium)\b",
    "back_pain": r"\b(back pain|lower back|lumbar)\b",
    "nausea_vomiting": r"\b(nausea|vomiting|n/v|vomit|nauseous)\b",
    "headache": r"\b(headache|head pain|migraine)\b",
    "syncope": r"\b(syncope|fainting|passed out|loss of consciousness)\b",
    "infection_signs": r"\b(infection|infected|sepsis|chills|rigors)\b",
}

def classify_chief_complaint(cc_text):
    """Map chief complaint text to categories."""
    if pd.isna(cc_text) or cc_text == "":
        return set()
    
    text = str(cc_text).lower()
    matched_cats = set()
    for cat, pattern in categories.items():
        if re.search(pattern, text):
            matched_cats.add(cat)
    return matched_cats

cohort["cc_categories"] = cohort["chiefcomplaint"].apply(classify_chief_complaint)

# Create dummy variables for each category
cc_cat_df = pd.DataFrame(
    {cat: cohort["cc_categories"].apply(lambda x: 1.0 if cat in x else 0.0) 
     for cat in categories.keys()},
    dtype=np.float32
)
cc_cat_feature_names = cc_cat_df.columns.tolist()
cc_cat_arr = cc_cat_df.values

print(f"  {len(categories)} categories created")
print(f"  category coverage: {(cohort['cc_categories'].apply(len) > 0).sum()} / {len(cohort)}")

# ── 4. Common Features (all options) ─────────────────────────────────────────
print("Preparing common features (demographics, vitals, ESI) …")

# ESI ordinal
cohort["esi_ordinal"] = (5 - cohort["acuity"].fillna(3)).astype(int).clip(0, 4)
esi_arr = cohort[["esi_ordinal"]].values.astype(np.float32)
esi_feature_names = ["esi_ordinal"]

# Continuous vitals + age
continuous_cols = ["anchor_age", "temperature", "heartrate", "resprate",
                   "o2sat", "sbp", "dbp"]
cont_arr = cohort[continuous_cols].values.astype(np.float32)
cont_feature_names = continuous_cols

# Dummy-encoded demographics (gender, race; drop-first)
gender_dummies = pd.get_dummies(cohort["gender"], prefix="gender", drop_first=True, dtype=np.float32)
race_dummies = pd.get_dummies(cohort["race"], prefix="race", drop_first=True, dtype=np.float32)
cat_df = pd.concat([gender_dummies, race_dummies], axis=1)
cat_arr = cat_df.values.astype(np.float32)
cat_feature_names = cat_df.columns.tolist()

# Labels and IDs (same for all)
y = cohort["ie_label"].values.astype(np.int32)
stay_ids = cohort["stay_id"].values

# ── 5. Build 3 Feature Matrices ─────────────────────────────────────────────
print("\nAssembling feature matrices …")

# Option 1: Categorical only
X_opt1 = np.concatenate([cont_arr, esi_arr, cat_arr, cc_cat_arr], axis=1)
feature_names_opt1 = cont_feature_names + esi_feature_names + cat_feature_names + cc_cat_feature_names

print(f"Option 1 (Categorical):")
print(f"  X shape: {X_opt1.shape}  ({len(feature_names_opt1)} features)")
print(f"  y distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# Option 2: PCA embeddings only
X_opt2 = np.concatenate([cont_arr, esi_arr, cat_arr, cc_embeddings_pca], axis=1)
feature_names_opt2 = cont_feature_names + esi_feature_names + cat_feature_names + cc_pca_feature_names

print(f"Option 2 (PCA Embeddings):")
print(f"  X shape: {X_opt2.shape}  ({len(feature_names_opt2)} features)")
print(f"  y distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# Option 3: Hybrid (categorical + PCA embeddings)
X_opt3 = np.concatenate([cont_arr, esi_arr, cat_arr, cc_cat_arr, cc_embeddings_pca], axis=1)
feature_names_opt3 = (cont_feature_names + esi_feature_names + cat_feature_names + 
                      cc_cat_feature_names + cc_pca_feature_names)

print(f"Option 3 (Hybrid):")
print(f"  X shape: {X_opt3.shape}  ({len(feature_names_opt3)} features)")
print(f"  y distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# Option 4: No chief complaint features at all
X_opt4 = np.concatenate([cont_arr, esi_arr, cat_arr], axis=1)
feature_names_opt4 = cont_feature_names + esi_feature_names + cat_feature_names

print(f"Option 4 (No Chief Complaint):")
print(f"  X shape: {X_opt4.shape}  ({len(feature_names_opt4)} features)")
print(f"  y distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# ── 6. Save 4 NPZ Files ──────────────────────────────────────────────────────
option_labels = ["categorical", "pca_embeddings", "hybrid", "no_cc"]
option_data = [
    (X_opt1, feature_names_opt1),
    (X_opt2, feature_names_opt2),
    (X_opt3, feature_names_opt3),
    (X_opt4, feature_names_opt4),
]
for opt_num, (X, feature_names) in enumerate(option_data, start=1):
    out_path = os.path.join(OUTPUT_DIR, f"features_option{opt_num}_{option_labels[opt_num - 1]}.npz")
    np.savez(
        out_path,
        X=X,
        y=y,
        stay_ids=stay_ids,
        feature_names=np.array(feature_names, dtype=object),
    )
    print(f"Saved: {os.path.basename(out_path)}")

print("\nDone! 4 feature matrices ready for modeling.")
