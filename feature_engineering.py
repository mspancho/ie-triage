"""
Feature engineering pipeline for the IE triage cohort.

Reads proj1_ie_triage_cohort.csv (output of cohort_creation.py) and produces
a single feature matrix (numpy .npz) plus metadata ready for modelling.

Steps:
  1. Chief-complaint embeddings  – BERT [CLS] pooled output via
     the model from https://github.com/dchang56/chief_complaints
  2. Ordinal ESI (acuity) encoding  – 1-5 mapped to 0-4 integer ordinal
  3. One-hot categorical encoding   – gender, race
  4. Float32 continuous vitals       – temperature, heartrate, resprate,
                                       o2sat, sbp, dbp, anchor_age

Output files (in OUTPUT_DIR):
  features.npz   – keys: X, y, stay_ids, feature_names
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.impute import KNNImputer, SimpleImputer

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_DIR = "/users/mspancho/Downloads/proj1_ie_triage"
OUTPUT_DIR = INPUT_DIR
COHORT_CSV = os.path.join(INPUT_DIR, "proj1_ie_triage_cohort.csv")

CC_MODEL_PATH = "thomas-sounack/BioClinical-ModernBERT-base"

BATCH_SIZE = 64
MAX_SEQ_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ── 1. Chief-complaint BERT embeddings ──────────────────────────────────────
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
        # ModernBERT has no pooler; take the [CLS] token from last_hidden_state
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    all_embeddings.append(cls_emb)

    if (start // BATCH_SIZE) % 20 == 0:
        print(f"  embedded {start + len(batch_texts)}/{len(texts)}")

cc_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
emb_dim = cc_embeddings.shape[1]
cc_feature_names = [f"cc_emb_{i}" for i in range(emb_dim)]
print(f"  embedding shape: {cc_embeddings.shape}")

# Free GPU memory
del bert_model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── 2. Ordinal ESI (acuity) encoding ────────────────────────────────────────
print("Encoding ESI acuity (ordinal 0-4) …")
# ESI 1 (most acute) → 4, ESI 5 (least acute) → 0  (higher = more acute)
cohort["esi_ordinal"] = (5 - cohort["acuity"].fillna(3)).astype(int).clip(0, 4)
esi_arr = cohort[["esi_ordinal"]].values.astype(np.float32)
esi_feature_names = ["esi_ordinal"]

# ── 3. One-hot categorical encoding ─────────────────────────────────────────
print("Dummy encoding gender and race (drop-first to avoid collinearity) …")

gender_dummies = pd.get_dummies(cohort["gender"], prefix="gender", drop_first=True, dtype=np.float32)
race_dummies = pd.get_dummies(cohort["race"], prefix="race", drop_first=True, dtype=np.float32)

cat_df = pd.concat([gender_dummies, race_dummies], axis=1)
cat_arr = cat_df.values.astype(np.float32)
cat_feature_names = cat_df.columns.tolist()

# ── 4. Float32 continuous vitals + age ──────────────────────────────────────
print("Casting continuous features to float32 …")
continuous_cols = ["anchor_age", "temperature", "heartrate", "resprate",
                   "o2sat", "sbp", "dbp"]
cont_arr = cohort[continuous_cols].values.astype(np.float32)
cont_feature_names = continuous_cols

# ── 5. Assemble final matrix ────────────────────────────────────────────────
print("Assembling feature matrix …")
X = np.concatenate([cont_arr, esi_arr, cat_arr, cc_embeddings], axis=1)
feature_names = cont_feature_names + esi_feature_names + cat_feature_names + cc_feature_names
y = cohort["ie_label"].values.astype(np.int32)
stay_ids = cohort["stay_id"].values

print(f"  X shape: {X.shape}  ({len(feature_names)} features)")
print(f"  y distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# ── 6. Save ─────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "features.npz")
np.savez(
    out_path,
    X=X,
    y=y,
    stay_ids=stay_ids,
    feature_names=np.array(feature_names, dtype=object),
)
print(f"Saved feature matrix to {out_path}")
