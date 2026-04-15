"""
Cohort creation pipeline for IE triage.

Builds an IE-positive cohort from MIMIC-IV, samples hard-negative controls
across three clinical strata, validates vital signs, and outputs a labeled dataset.
"""

import argparse
import os
import pandas as pd


IE_CODES = {"I33.0", "I330", "4210", "421.0", "I33.9", "I339", "4211", "421.1", "I38", "I39"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create IE triage cohort from MIMIC-IV.")
    parser.add_argument("--mimic-hosp", type=str, required=True,
                        help="Path to MIMIC-IV hosp directory (e.g. /oscar/data/shared/ursa/mimic-iv/hosp/3.1)")
    parser.add_argument("--mimic-ed", type=str, required=True,
                        help="Path to MIMIC-IV ED directory (e.g. /oscar/data/shared/ursa/mimic-iv/ed/2.2)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output CSV files (data/csv)")
    return parser.parse_args()


def normalize_race_value(race_value):
    if pd.isna(race_value):
        return "Missing"
    race_text = str(race_value).strip().upper()
    if race_text == "":
        return "Missing"
    if race_text in {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}:
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tables...")
    diagnoses = pd.read_csv(
        os.path.join(args.mimic_hosp, "diagnoses_icd.csv"),
        dtype={"icd_code": str, "hadm_id": float},
    )
    admissions = pd.read_csv(
        os.path.join(args.mimic_hosp, "admissions.csv"),
        parse_dates=["admittime", "dischtime", "deathtime"],
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "insurance", "language", "marital_status", "race"],
    )
    patients = pd.read_csv(
        os.path.join(args.mimic_hosp, "patients.csv"),
        usecols=["subject_id", "gender", "anchor_age"],
    )
    edstays = pd.read_csv(
        os.path.join(args.mimic_ed, "edstays.csv"),
        parse_dates=["intime", "outtime"],
    )
    triage = pd.read_csv(os.path.join(args.mimic_ed, "triage.csv"))

    # Label IE-positive ED stays by hospital admission mapping
    ie_dx = diagnoses[diagnoses["icd_code"].str.strip().isin(IE_CODES)].copy()
    ije_hadm = ie_dx["hadm_id"].dropna().unique()
    print(f"IE diagnosis events (hospital hadm): {len(ie_dx)}")
    print(f"Unique IE admissions (hadm_id): {len(ije_hadm)}")

    ed_triage = edstays.merge(triage, on=["subject_id", "stay_id"], how="left")
    ed_triage = ed_triage.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id", how="left", suffixes=("", "_patients"),
    )
    ad_demo = admissions[["hadm_id", "race"]].drop_duplicates(subset=["hadm_id"]).reset_index(drop=True)
    ed_triage = ed_triage.merge(ad_demo, on="hadm_id", how="left", suffixes=("", "_admissions"))

    ed_triage["gender"] = ed_triage["gender_patients"].fillna(ed_triage.get("gender", None))
    ed_triage["race"] = ed_triage.get("race_admissions", ed_triage.get("race", None))
    ed_triage = ed_triage.drop(columns=["gender_patients", "race_admissions"], errors="ignore")

    ed_triage["ie_label"] = ed_triage["hadm_id"].isin(ije_hadm).astype(int)

    print(f"Columns in ed_triage: {ed_triage.columns.tolist()}")
    print(f"Gender present: {ed_triage['gender'].notna().sum()}, Race present: {ed_triage['race'].notna().sum()}")

    positive = ed_triage[ed_triage["ie_label"] == 1].copy()
    negatives = ed_triage[ed_triage["ie_label"] == 0].copy()

    print(f"Total ED stays: {len(ed_triage)}")
    print(f"IE-positive ED stays: {len(positive)}")
    print(f"Non-IE ED stays: {len(negatives)}")

    # Stratum A: high mimicry conditions
    mimic_codes = [
        "A40", "A41", "038",
        "J18", "J15", "4801", "481", "482", "484", "486", "487", "5168",
        "G00", "G03", "3209", "322",
        "R50",
        "M00", "M86", "0389", "730",
        "I63", "434",
        "I50", "428",
        "I30", "I40", "420", "42292",
    ]

    risk_diagnosis_codes = ["Z87891", "V1582", "3051", "Z95", "Z8739", "Z992", "V4511"]
    risk_keyword_pattern = r"fever|chills|rigors|back pain|weakness|shortness of breath"

    negatives["chiefcomplaint_text"] = negatives["chiefcomplaint"].fillna("").str.lower()
    negatives["has_risk_keyword"] = negatives["chiefcomplaint_text"].str.contains(
        risk_keyword_pattern, regex=True,
    )

    negative_hadm = negatives["hadm_id"].dropna().unique()
    negative_diag = diagnoses[diagnoses["hadm_id"].isin(negative_hadm)].copy()
    negative_diag["icd_code"] = negative_diag["icd_code"].astype(str).str.strip().str.upper()
    negative_diag["is_mimic_code"] = negative_diag["icd_code"].str.startswith(tuple(mimic_codes))
    negative_diag["is_risk_code"] = negative_diag["icd_code"].str.startswith(tuple(risk_diagnosis_codes))

    hadm_flags = negative_diag.groupby("hadm_id").agg(
        mimic=("is_mimic_code", "any"),
        risk_history=("is_risk_code", "any"),
    ).reset_index()
    negatives = negatives.merge(hadm_flags, on="hadm_id", how="left")
    negatives[["mimic", "risk_history"]] = negatives[["mimic", "risk_history"]].fillna(False)
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

    # Validate and clean data
    print("Validating and cleaning cohort data...")
    cohort_out['subject_id'] = pd.to_numeric(cohort_out['subject_id'], errors='coerce').astype('Int64')
    cohort_out['stay_id'] = pd.to_numeric(cohort_out['stay_id'], errors='coerce').astype('Int64')
    cohort_out['hadm_id'] = pd.to_numeric(cohort_out['hadm_id'], errors='coerce').astype('Int64')
    cohort_out = cohort_out[cohort_out['ie_label'].isin([0, 1])]
    cohort_out = cohort_out[cohort_out['gender'].isin(['M', 'F'])]

    cohort_out['anchor_age'] = pd.to_numeric(cohort_out['anchor_age'], errors='coerce')
    cohort_out = cohort_out[(cohort_out['anchor_age'] >= 0) & (cohort_out['anchor_age'] <= 120)]

    vital_ranges = {
        'temperature': (95, 105),
        'heartrate': (40, 200),
        'resprate': (8, 40),
        'o2sat': (70, 100),
        'sbp': (70, 250),
        'dbp': (40, 150),
    }
    for col, (min_val, max_val) in vital_ranges.items():
        cohort_out[col] = pd.to_numeric(cohort_out[col], errors='coerce')
        cohort_out = cohort_out[
            (cohort_out[col].isna()) | ((cohort_out[col] >= min_val) & (cohort_out[col] <= max_val))
        ]

    cohort_out['acuity'] = pd.to_numeric(cohort_out['acuity'], errors='coerce')
    cohort_out = cohort_out[(cohort_out['acuity'].isna()) | (cohort_out['acuity'].between(1, 5))]

    cohort_out['intime'] = pd.to_datetime(cohort_out['intime'], errors='coerce')
    cohort_out['outtime'] = pd.to_datetime(cohort_out['outtime'], errors='coerce')
    cohort_out = cohort_out[cohort_out['intime'] < cohort_out['outtime']]
    cohort_out['chiefcomplaint'] = cohort_out['chiefcomplaint'].astype(str).str.strip()

    print(f"After validation: {len(cohort_out)} rows remaining")

    cohort_out.to_csv(os.path.join(args.output_dir, "proj1_ie_triage_cohort.csv"), index=False)

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
    summary.to_csv(os.path.join(args.output_dir, "proj1_ie_cohort_summary.csv"), index=False)

    print(f"Wrote cohort to {os.path.join(args.output_dir, 'proj1_ie_triage_cohort.csv')}")
    print(f"Wrote cohort summary to {os.path.join(args.output_dir, 'proj1_ie_cohort_summary.csv')}")


if __name__ == "__main__":
    main()
