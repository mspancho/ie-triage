"""
Preliminary feasibility analysis for the IE triage study.

Estimates IE prevalence in the ED and assesses whether MIMIC-IV data
supports the study design.
"""

import argparse
import os
import pandas as pd


IE_CODES = {"I33.0", "I330", "4210", "421.0", "I33.9", "I339", "4211", "421.1", "I38", "I39"}


def parse_args():
    parser = argparse.ArgumentParser(description="IE triage feasibility analysis.")
    parser.add_argument("--mimic-hosp", type=str, required=True,
                        help="Path to MIMIC-IV hosp directory")
    parser.add_argument("--mimic-ed", type=str, required=True,
                        help="Path to MIMIC-IV ED directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output CSV files (data/csv)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tables...")
    diagnoses = pd.read_csv(os.path.join(args.mimic_hosp, "diagnoses_icd.csv"), dtype={"icd_code": str})
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
    edstays = pd.read_csv(os.path.join(args.mimic_ed, "edstays.csv"), parse_dates=["intime", "outtime"])
    triage = pd.read_csv(os.path.join(args.mimic_ed, "triage.csv"))

    ie_dx = diagnoses[diagnoses["icd_code"].str.strip().isin(IE_CODES)]
    ije_hadm = ie_dx["hadm_id"].dropna().unique()
    print(f"IE diagnosis events (hospital hadm): {len(ie_dx)}")
    print(f"Unique IE admissions (hadm_id): {len(ije_hadm)}")

    ed_iebymap = edstays[edstays["hadm_id"].isin(ije_hadm)].copy()
    print(f"IE ED stays via hadm match: {len(ed_iebymap)}")

    ed_triage = edstays.merge(triage, on=["subject_id", "stay_id"], how="left")
    ed_triage["ie_label"] = ed_triage["hadm_id"].isin(ije_hadm).astype(int)

    feature_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
                    "acuity", "chiefcomplaint"]
    feats = ed_triage[feature_cols + ["ie_label"]]

    n_total = len(ed_triage)
    positive = int(feats["ie_label"].sum())
    negative = n_total - positive
    pos_rate = positive / n_total * 100 if n_total > 0 else 0.0
    print(f"Total ED stays: {n_total}")
    print(f"IE-labeled ED stays: {positive} ({pos_rate:.2f}%)")
    print(f"Non-IE ED stays: {negative}")

    availability = feats.isna().mean().rename("missing_pct").to_frame()
    availability["present_pct"] = 100 - availability["missing_pct"] * 100
    print("\nFeature availability (all ED stays):")
    print(availability)

    availability_pos = feats[feats["ie_label"] == 1].isna().mean().rename("missing_pct").to_frame()
    availability_pos["present_pct"] = 100 - availability_pos["missing_pct"] * 100
    print("\nFeature availability (IE-positive ED stays):")
    print(availability_pos)

    target_controls = positive * 10
    print(f"Target non-IE controls for 10:1 class balance: {target_controls}")
    print(f"Available non-IE ED stays: {negative}")

    summary = pd.DataFrame({
        "metric": [
            "total_ed_stays", "ie_ed_stays", "non_ie_ed_stays", "ie_rate_pct",
            "esi_collected_pct", "chief_complaint_present_pct", "temp_present_pct",
        ],
        "value": [
            n_total, positive, negative, f"{pos_rate:.2f}",
            f"{availability.loc['acuity', 'present_pct']:.2f}",
            f"{availability.loc['chiefcomplaint', 'present_pct']:.2f}",
            f"{availability.loc['temperature', 'present_pct']:.2f}",
        ],
    })
    summary.to_csv(os.path.join(args.output_dir, "proj1_ie_triage_feasibility_summary.csv"), index=False)
    print(f"Wrote feasibility summary to {args.output_dir}")


if __name__ == "__main__":
    main()
