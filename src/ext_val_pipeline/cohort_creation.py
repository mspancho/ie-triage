"""
External-validation cohort creation for IE triage, built from the MC-MED dataset.

Mirrors src/pipeline/cohort_creation.py exactly in spirit and procedure:
  * Same IE ICD code set for positive labeling.
  * Same 10:1 control-to-case target ratio.
  * Same A/B/C stratified hard-negative sampling (40 / 35 / 25 % targets).
  * Same vital-sign hard-range validation, same race normalization, same
    final column schema.

Differences from MIMIC version are MC-MED schema only:
  * MC-MED visits.csv stores ONE primary diagnosis per ED visit
    (Dx_ICD9 / Dx_ICD10), not the full hospital diagnosis list. The
    high-mimicry stratum-A flag is therefore derived from the visit's
    primary code AND from pmh.csv (the patient's prior diagnoses).
  * Triage_Temp is in degrees Celsius and is converted to Fahrenheit
    so the validation ranges from the internal pipeline apply unchanged.
  * Triage_acuity is encoded as "3-Urgent" etc. and is parsed back to
    the integer 1-5.
"""

import argparse
import os
import re
import pandas as pd


IE_CODES = {"I33.0", "I330", "4210", "421.0", "I33.9", "I339", "4211", "421.1", "I38", "I39"}

MIMIC_CODES = (
    "A40", "A41", "038",
    "J18", "J15", "4801", "481", "482", "484", "486", "487", "5168",
    "G00", "G03", "3209", "322",
    "R50",
    "M00", "M86", "0389", "730",
    "I63", "434",
    "I50", "428",
    "I30", "I40", "420", "42292",
)

RISK_DIAGNOSIS_CODES = ("Z87891", "V1582", "3051", "Z95", "Z8739", "Z992", "V4511")
RISK_KEYWORD_PATTERN = r"fever|chills|rigors|back pain|weakness|shortness of breath"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the IE triage external-validation cohort from MC-MED.",
    )
    parser.add_argument("--mc-med-dir", type=str, required=True,
                        help="Path to MC-MED data directory (e.g. /oscar/data/shared/ursa/mc-med/1.0.1/data)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output CSV files (data/csv)")
    return parser.parse_args()


def normalize_race_value(race_value):
    if pd.isna(race_value):
        return "Missing"
    race_text = str(race_value).strip().upper()
    if race_text == "":
        return "Missing"
    if race_text in {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER", "DECLINES TO STATE"}:
        return "Unknown"
    if race_text.startswith("WHITE"):
        return "White"
    if "BLACK" in race_text or "AFRICAN" in race_text:
        return "Black / African American"
    if race_text.startswith("ASIAN"):
        return "Asian"
    if "HISPANIC" in race_text or "LATINO" in race_text:
        return "Hispanic / Latino"
    return "Other"


def parse_acuity(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    match = re.match(r"\s*(\d+)", text)
    return int(match.group(1)) if match else None


def normalize_code(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def code_starts_with_any(code, prefixes):
    return code.startswith(prefixes) if code else False


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading MC-MED tables...")
    visits = pd.read_csv(
        os.path.join(args.mc_med_dir, "visits.csv"),
        parse_dates=["Arrival_time", "Departure_time", "Roomed_time",
                     "Dispo_time", "Admit_time"],
        dtype={"Dx_ICD9": str, "Dx_ICD10": str, "CC": str},
    )
    pmh = pd.read_csv(
        os.path.join(args.mc_med_dir, "pmh.csv"),
        usecols=["MRN", "Code"],
        dtype={"Code": str},
    )

    # ── Rename / convert into the same column space as the internal cohort ──
    print("Mapping MC-MED schema to internal cohort schema...")
    df = visits.rename(columns={
        "MRN": "subject_id",
        "CSN": "stay_id",
        "Arrival_time": "intime",
        "Departure_time": "outtime",
        "Gender": "gender",
        "Age": "anchor_age",
        "Race": "race",
        "Triage_HR": "heartrate",
        "Triage_RR": "resprate",
        "Triage_SpO2": "o2sat",
        "Triage_SBP": "sbp",
        "Triage_DBP": "dbp",
        "CC": "chiefcomplaint",
    }).copy()

    # MC-MED has no hadm_id; one ED visit per row. Use stay_id to anchor PMH lookup.
    df["hadm_id"] = df["stay_id"]

    # Celsius → Fahrenheit so the internal vital_ranges (95–105 °F) apply directly.
    df["temperature"] = pd.to_numeric(visits["Triage_Temp"], errors="coerce") * 9.0 / 5.0 + 32.0
    df["acuity"] = visits["Triage_acuity"].apply(parse_acuity)

    df["Dx_ICD9_norm"] = visits["Dx_ICD9"].apply(normalize_code)
    df["Dx_ICD10_norm"] = visits["Dx_ICD10"].apply(normalize_code)

    # ── IE label: primary visit Dx in IE code set ──────────────────────────
    df["ie_label"] = (
        df["Dx_ICD10_norm"].isin({c.upper() for c in IE_CODES})
        | df["Dx_ICD9_norm"].isin({c.upper() for c in IE_CODES})
    ).astype(int)

    print(f"Total ED visits: {len(df)}")
    print(f"IE-positive ED visits: {df['ie_label'].sum()}")

    positive = df[df["ie_label"] == 1].copy()
    negatives = df[df["ie_label"] == 0].copy()
    print(f"Non-IE ED visits: {len(negatives)}")

    # ── Stratum flags for negative controls ─────────────────────────────────
    # Visit-primary-diagnosis flags
    neg_visit_codes = pd.concat([negatives["Dx_ICD9_norm"], negatives["Dx_ICD10_norm"]], axis=1)
    visit_mimic = (
        neg_visit_codes["Dx_ICD9_norm"].apply(lambda c: code_starts_with_any(c, MIMIC_CODES))
        | neg_visit_codes["Dx_ICD10_norm"].apply(lambda c: code_starts_with_any(c, MIMIC_CODES))
    )
    visit_risk_history = (
        neg_visit_codes["Dx_ICD9_norm"].apply(lambda c: code_starts_with_any(c, RISK_DIAGNOSIS_CODES))
        | neg_visit_codes["Dx_ICD10_norm"].apply(lambda c: code_starts_with_any(c, RISK_DIAGNOSIS_CODES))
    )

    # Per-patient PMH flags (analog of MIMIC's diagnoses_icd lookup)
    pmh = pmh.copy()
    pmh["Code_norm"] = pmh["Code"].apply(normalize_code)
    pmh["is_mimic"] = pmh["Code_norm"].apply(lambda c: code_starts_with_any(c, MIMIC_CODES))
    pmh["is_risk"] = pmh["Code_norm"].apply(lambda c: code_starts_with_any(c, RISK_DIAGNOSIS_CODES))
    pmh_flags = pmh.groupby("MRN").agg(
        pmh_mimic=("is_mimic", "any"),
        pmh_risk=("is_risk", "any"),
    ).reset_index().rename(columns={"MRN": "subject_id"})

    negatives = negatives.merge(pmh_flags, on="subject_id", how="left")
    negatives[["pmh_mimic", "pmh_risk"]] = negatives[["pmh_mimic", "pmh_risk"]].fillna(False)
    negatives["mimic"] = visit_mimic.values | negatives["pmh_mimic"].values
    negatives["risk_history"] = visit_risk_history.values | negatives["pmh_risk"].values

    negatives["chiefcomplaint_text"] = negatives["chiefcomplaint"].fillna("").str.lower()
    negatives["has_risk_keyword"] = negatives["chiefcomplaint_text"].str.contains(
        RISK_KEYWORD_PATTERN, regex=True,
    )
    negatives["moderate_mimic"] = negatives["risk_history"] | negatives["has_risk_keyword"]

    stratum_a = negatives[negatives["mimic"]].copy()
    stratum_b = negatives[~negatives["mimic"] & negatives["moderate_mimic"]].copy()
    stratum_c = negatives[~negatives["mimic"] & ~negatives["moderate_mimic"]].copy()

    positive_count = len(positive)
    target_controls = positive_count * 10

    n_a = min(int(round(target_controls * 0.40)), len(stratum_a))
    n_b = min(int(round(target_controls * 0.35)), len(stratum_b))
    n_c = min(target_controls - n_a - n_b, len(stratum_c))

    selected = []
    if n_a > 0:
        selected.append(stratum_a.sample(n=n_a, random_state=42))
    if n_b > 0:
        selected.append(stratum_b.sample(n=n_b, random_state=42))
    if n_c > 0:
        selected.append(stratum_c.sample(n=n_c, random_state=42))

    sampled_negatives = pd.concat(selected, axis=0) if selected else negatives.iloc[0:0]

    if len(sampled_negatives) < target_controls:
        remaining = negatives.drop(sampled_negatives.index, errors="ignore")
        fill = min(target_controls - len(sampled_negatives), len(remaining))
        if fill > 0:
            sampled_negatives = pd.concat(
                [sampled_negatives, remaining.sample(n=fill, random_state=42)], axis=0,
            )

    final_controls = len(sampled_negatives)
    final_ratio = final_controls / positive_count if positive_count else 0.0
    print(f"Selected negatives: {final_controls} / {target_controls}")
    print(f"Stratum pool sizes: A={len(stratum_a)}, B={len(stratum_b)}, C={len(stratum_c)}")
    print(f"Sampled counts: A={n_a}, B={n_b}, C={n_c}, fill={final_controls - (n_a + n_b + n_c)}")
    print(f"Final negative:positive ratio: {final_ratio:.2f}:1")

    # Drop helper columns before concatenating positives (which lack them)
    helper_cols = ["chiefcomplaint_text", "has_risk_keyword", "mimic", "risk_history",
                   "moderate_mimic", "pmh_mimic", "pmh_risk"]
    sampled_negatives = sampled_negatives.drop(columns=helper_cols, errors="ignore")

    cohort = pd.concat([positive, sampled_negatives], axis=0).sample(frac=1, random_state=42)

    feature_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
                    "acuity", "chiefcomplaint"]
    demographic_cols = ["gender", "anchor_age", "race"]
    cohort_out = cohort[
        ["subject_id", "stay_id", "hadm_id", "intime", "outtime", "ie_label",
         *demographic_cols, *feature_cols]
    ].copy()

    cohort_out["race"] = cohort_out["race"].apply(normalize_race_value)
    cohort_out["chiefcomplaint"] = cohort_out["chiefcomplaint"].fillna("").astype(str)

    # Validate and clean data — identical ranges to internal pipeline.
    print("Validating and cleaning cohort data...")
    cohort_out["subject_id"] = pd.to_numeric(cohort_out["subject_id"], errors="coerce").astype("Int64")
    cohort_out["stay_id"] = pd.to_numeric(cohort_out["stay_id"], errors="coerce").astype("Int64")
    cohort_out["hadm_id"] = pd.to_numeric(cohort_out["hadm_id"], errors="coerce").astype("Int64")
    cohort_out = cohort_out[cohort_out["ie_label"].isin([0, 1])]
    cohort_out = cohort_out[cohort_out["gender"].isin(["M", "F"])]

    cohort_out["anchor_age"] = pd.to_numeric(cohort_out["anchor_age"], errors="coerce")
    cohort_out = cohort_out[(cohort_out["anchor_age"] >= 0) & (cohort_out["anchor_age"] <= 120)]

    vital_ranges = {
        "temperature": (95, 105),
        "heartrate": (40, 200),
        "resprate": (8, 40),
        "o2sat": (70, 100),
        "sbp": (70, 250),
        "dbp": (40, 150),
    }
    for col, (min_val, max_val) in vital_ranges.items():
        cohort_out[col] = pd.to_numeric(cohort_out[col], errors="coerce")
        cohort_out = cohort_out[
            (cohort_out[col].isna()) | ((cohort_out[col] >= min_val) & (cohort_out[col] <= max_val))
        ]

    cohort_out["acuity"] = pd.to_numeric(cohort_out["acuity"], errors="coerce")
    cohort_out = cohort_out[(cohort_out["acuity"].isna()) | (cohort_out["acuity"].between(1, 5))]

    cohort_out["intime"] = pd.to_datetime(cohort_out["intime"], errors="coerce")
    cohort_out["outtime"] = pd.to_datetime(cohort_out["outtime"], errors="coerce")
    cohort_out = cohort_out[cohort_out["intime"] < cohort_out["outtime"]]
    cohort_out["chiefcomplaint"] = cohort_out["chiefcomplaint"].astype(str).str.strip()

    print(f"After validation: {len(cohort_out)} rows remaining")

    out_path = os.path.join(args.output_dir, "proj1_ie_triage_extval_cohort.csv")
    cohort_out.to_csv(out_path, index=False)

    summary = pd.DataFrame({
        "metric": [
            "ie_positive_count", "target_negative_controls", "selected_negative_controls",
            "final_negative_positive_ratio",
            "stratum_a_pool", "stratum_b_pool", "stratum_c_pool",
            "stratum_a_sampled", "stratum_b_sampled", "stratum_c_sampled",
        ],
        "value": [
            positive_count, target_controls, final_controls, f"{final_ratio:.2f}:1",
            len(stratum_a), len(stratum_b), len(stratum_c), n_a, n_b, n_c,
        ],
    })
    summary.to_csv(os.path.join(args.output_dir, "proj1_ie_extval_cohort_summary.csv"), index=False)

    print(f"Wrote external-validation cohort to {out_path}")


if __name__ == "__main__":
    main()
