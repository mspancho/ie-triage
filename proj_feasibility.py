import os
import pandas as pd

# Paths
MIMIC_HOSP = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1"
MIMIC_ED = "/oscar/data/shared/ursa/mimic-iv/ed/2.2"
OUTPUT_DIR = "/users/mspancho/Downloads/proj1_ie_triage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IE_CODES = {"I33.0", "I330", "4210", "421.0", "I33.9", "I339", "4211", "421.1", "I38", "I39"}

print("Loading tables...")
# Core tables
diagnoses = pd.read_csv(os.path.join(MIMIC_HOSP, "diagnoses_icd.csv"), dtype={"icd_code": str})
admissions = pd.read_csv(
    os.path.join(MIMIC_HOSP, "admissions.csv"),
    parse_dates=["admittime", "dischtime", "deathtime"],
    usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "insurance", "language", "marital_status", "race"],
)
patients = pd.read_csv(
    os.path.join(MIMIC_HOSP, "patients.csv"),
    usecols=["subject_id", "gender", "anchor_age"],
)
edstays = pd.read_csv(os.path.join(MIMIC_ED, "edstays.csv"), parse_dates=["intime", "outtime"])
triage = pd.read_csv(os.path.join(MIMIC_ED, "triage.csv"))

# IE cohort from hospital diagnoses
ie_dx = diagnoses[diagnoses["icd_code"].str.strip().isin(IE_CODES)]
ije_hadm = ie_dx["hadm_id"].dropna().unique()
print(f"IE diagnosis events (hospital hadm): {len(ie_dx)}")
print(f"Unique IE admissions (hadm_id): {len(ije_hadm)}")

# ED stays that are IE-positive by same admission mapping
ed_iebymap = edstays[edstays["hadm_id"].isin(ije_hadm)].copy()
print(f"IE ED stays via hadm match: {len(ed_iebymap)}")

# Merge ED stays with triage features
ed_triage = edstays.merge(triage, on=["subject_id", "stay_id"], how="left")

# Label ED stays: IE-positive if matched by hadm. Otherwise non-IE.
ed_triage["ie_label"] = ed_triage["hadm_id"].isin(ije_hadm).astype(int)

# Identify available features
feature_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity", "chiefcomplaint"]
feats = ed_triage[feature_cols + ["ie_label"]]

# Summary counts
n_total = len(ed_triage)
positive = int(feats["ie_label"].sum())
negative = n_total - positive
pos_rate = positive / n_total * 100 if n_total > 0 else 0.0
print(f"Total ED stays: {n_total}")
print(f"IE-labeled ED stays: {positive} ({pos_rate:.2f}%)")
print(f"Non-IE ED stays: {negative}")

# Feature availability
availability = feats.isna().mean().rename("missing_pct").to_frame()
availability["present_pct"] = 100 - availability["missing_pct"] * 100
print("\nFeature availability (all ED stays):")
print(availability)

# For IE positives only
availability_pos = feats[feats["ie_label"] == 1].isna().mean().rename("missing_pct").to_frame()
availability_pos["present_pct"] = 100 - availability_pos["missing_pct"] * 100
print("\nFeature availability (IE-positive ED stays):")
print(availability_pos)

# 10:1 control feasibility
target_controls = positive * 10
print(f"Target non-IE controls for 10:1 class balance: {target_controls}")
print(f"Available non-IE ED stays: {negative}")

# Small EDA table for output
summary = pd.DataFrame({
    "metric": [
        "total_ed_stays",
        "ie_ed_stays",
        "non_ie_ed_stays",
        "ie_rate_pct",
        "esi_collected_pct",
        "chief_complaint_present_pct",
        "temp_present_pct",
    ],
    "value": [
        n_total,
        positive,
        negative,
        f"{pos_rate:.2f}",
        f"{availability.loc['acuity', 'present_pct']:.2f}",
        f"{availability.loc['chiefcomplaint', 'present_pct']:.2f}",
        f"{availability.loc['temperature', 'present_pct']:.2f}",
    ],
})

summary.to_csv(os.path.join(OUTPUT_DIR, "proj1_ie_triage_feasibility_summary.csv"), index=False)

# Save candidate table
ed_triage["is_ie"] = ed_triage["ie_label"]
ed_triage_out = ed_triage[["subject_id", "stay_id", "hadm_id", "intime", "outtime", "ie_label", *feature_cols]]
ed_triage_out.to_csv(os.path.join(OUTPUT_DIR, "proj1_ie_triage_cohort.csv"), index=False)

print(f"Wrote feasibility summary and cohort file to {OUTPUT_DIR}")
