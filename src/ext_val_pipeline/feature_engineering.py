"""
External-validation feature engineering — categorical chief-complaint encoding only.

Builds a single feature matrix that is schema-compatible with the internal
Option 1 (categorical) features, so that an RF trained on the internal
features can predict directly on the MC-MED matrix without any column drift.

Notes:
  * Imputation here is fit ON THE EXTERNAL COHORT (train statistics from
    the external set are used to fill external missingness). This matches
    the internal pipeline, which fits its imputers on the cohort being
    encoded — there is no leakage of internal-cohort statistics into the
    external set, and the ML script itself standardizes against the
    training set's mean/std.
  * Gender/race dummy columns are aligned to the canonical names from
    the internal npz: missing dummy columns are filled with zeros, extra
    levels not seen internally are dropped.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


def parse_args():
    parser = argparse.ArgumentParser(
        description="External-validation feature engineering (categorical CC only).",
    )
    parser.add_argument("--cohort-csv", type=str, required=True,
                        help="Path to MC-MED external-validation cohort CSV")
    parser.add_argument("--internal-npz", type=str, required=True,
                        help="Path to internal Option 1 npz, used as the canonical "
                             "feature_names schema for column alignment.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output .npz file")
    return parser.parse_args()


CATEGORIES = {
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
    if pd.isna(cc_text) or cc_text == "":
        return set()
    text = str(cc_text).lower()
    return {cat for cat, pattern in CATEGORIES.items() if re.search(pattern, text)}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading external cohort and internal feature schema...")
    cohort = pd.read_csv(args.cohort_csv)
    print(f"  external cohort rows: {len(cohort)}")

    internal = np.load(args.internal_npz, allow_pickle=True)
    canonical_names = [str(n) for n in internal["feature_names"]]
    print(f"  canonical (internal) feature count: {len(canonical_names)}")

    # ── Imputation — same strategy as internal pipeline ─────────────────────
    print("Imputing missing values...")
    continuous_cols = ["anchor_age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
    ordinal_cols = ["acuity"]
    categorical_cols = ["gender", "race"]

    if cohort[continuous_cols].isna().sum().sum() > 0:
        knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
        cohort[continuous_cols] = knn_imputer.fit_transform(cohort[continuous_cols])
    if cohort[ordinal_cols].isna().sum().sum() > 0:
        cohort[ordinal_cols] = SimpleImputer(strategy="median").fit_transform(cohort[ordinal_cols])
    if cohort[categorical_cols].isna().sum().sum() > 0:
        cohort[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(cohort[categorical_cols])

    # ── Categorical chief-complaint encoding ────────────────────────────────
    print("Categorical encoding chief complaints...")
    cohort["cc_categories"] = cohort["chiefcomplaint"].apply(classify_chief_complaint)
    cc_cat_df = pd.DataFrame(
        {cat: cohort["cc_categories"].apply(lambda x, c=cat: 1.0 if c in x else 0.0)
         for cat in CATEGORIES.keys()},
        dtype=np.float32,
    )
    cc_cat_feature_names = cc_cat_df.columns.tolist()
    print(f"  category coverage: {(cohort['cc_categories'].apply(len) > 0).sum()} / {len(cohort)}")

    # ── Common features (must mirror internal feature_engineering.py) ──────
    print("Preparing common features (demographics, vitals, ESI)...")
    cohort["esi_ordinal"] = (5 - cohort["acuity"].fillna(3)).astype(int).clip(0, 4)
    esi_arr = cohort[["esi_ordinal"]].values.astype(np.float32)
    esi_feature_names = ["esi_ordinal"]

    cont_arr = cohort[continuous_cols].values.astype(np.float32)
    cont_feature_names = continuous_cols

    gender_dummies = pd.get_dummies(cohort["gender"], prefix="gender", drop_first=True, dtype=np.float32)
    race_dummies = pd.get_dummies(cohort["race"], prefix="race", drop_first=True, dtype=np.float32)
    cat_df_raw = pd.concat([gender_dummies, race_dummies], axis=1)

    # ── Schema alignment: match canonical (internal) column order ──────────
    # Categorical CC columns are deterministic (literal CATEGORIES keys), so they
    # already align. Vitals and ESI columns are also fixed. The only place schema
    # drift can happen is the gender/race dummies — fill missing levels with 0
    # and drop levels that were not present in the internal cohort.
    print("Aligning gender/race dummies to internal schema...")
    canonical_dummy_names = [n for n in canonical_names
                             if n.startswith("gender_") or n.startswith("race_")]
    cat_df = pd.DataFrame(0.0, index=cat_df_raw.index, columns=canonical_dummy_names, dtype=np.float32)
    for col in canonical_dummy_names:
        if col in cat_df_raw.columns:
            cat_df[col] = cat_df_raw[col].astype(np.float32).values
    extra = [c for c in cat_df_raw.columns if c not in canonical_dummy_names]
    if extra:
        print(f"  warning — dropping {len(extra)} dummy column(s) not in internal schema: {extra}")
    missing = [c for c in canonical_dummy_names if c not in cat_df_raw.columns]
    if missing:
        print(f"  filled with zeros for {len(missing)} internal-only dummy(ies): {missing}")

    # ── Assemble in canonical Option 1 order ────────────────────────────────
    expected_order = (cont_feature_names + esi_feature_names +
                      canonical_dummy_names + cc_cat_feature_names)
    if expected_order != canonical_names:
        raise ValueError(
            "Internal feature_names schema does not match the expected Option 1 layout.\n"
            f"  expected: {expected_order}\n"
            f"  internal: {canonical_names}"
        )

    X = np.concatenate([cont_arr, esi_arr, cat_df.values, cc_cat_df.values], axis=1)
    print(f"External feature matrix shape: {X.shape}  (canonical: {len(canonical_names)} features)")

    y = cohort["ie_label"].values.astype(np.int32)
    stay_ids = cohort["stay_id"].values

    out_path = os.path.join(args.output_dir, "extval_features_option1_categorical.npz")
    np.savez(out_path, X=X, y=y, stay_ids=stay_ids,
             feature_names=np.array(canonical_names, dtype=object))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
