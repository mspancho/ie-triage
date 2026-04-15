import pandas as pd
import numpy as np

COHORT_PATH = "/users/mspancho/Downloads/proj1_ie_triage/proj1_ie_triage_cohort.csv"
RESULTS_DIR = "/users/mspancho/Downloads/proj1_ie_triage"

# Important: this cohort already contains only ED stay-level data for IE and non-IE,
# so summary is for encounters, not unique patients.

df = pd.read_csv(COHORT_PATH)

# --- Consolidate race ---
def consolidate_race(r):
    r = str(r).upper()
    if r.startswith("WHITE"):
        return "White"
    if "BLACK" in r or "AFRICAN" in r:
        return "Black / African American"
    if "HISPANIC" in r or "LATINO" in r:
        return "Hispanic / Latino"
    if "ASIAN" in r:
        return "Asian"
    if "UNKNOWN" in r or "UNABLE" in r:
        return "Unknown / Not Obtained"
    return "Other"

if "race" in df.columns:
    df["race_group"] = df["race"].apply(consolidate_race)
else:
    df["race_group"] = "Unknown / Not Obtained"

if "anchor_age" not in df.columns:
    df["anchor_age"] = np.nan

# --- Summary stats table ---
summary = pd.DataFrame({
    "N": [len(df)],
    "IE prevalence (%)": [f"{df['ie_label'].mean() * 100:.3f}"],
    "Male (%)": [f"{(df['gender']=='M').mean() * 100:.1f}" if 'gender' in df.columns else 'NA'],
    "Median age": [df["anchor_age"].median()],
    "IQR age": [f"{df['anchor_age'].quantile(0.25):.0f}–{df['anchor_age'].quantile(0.75):.0f}"],
})
summary.to_csv(f"{RESULTS_DIR}/demographics_proj1_summary.csv", index=False)

N = len(df)

# ── Raw Numbers: Age Bins ─────────────────────────────────────────────────────
age_bins = np.histogram_bin_edges(df["anchor_age"].dropna(), bins=6)
age_counts, _ = np.histogram(df["anchor_age"].dropna(), bins=age_bins)
age_bin_labels = [f"{int(age_bins[i])}-{int(age_bins[i+1])}" for i in range(len(age_bins)-1)]

print("\n=== AGE DISTRIBUTION (6 BINS) ===")
print(f"{'Bin Range':<15} {'Count':<10} {'Percentage':<10}")
print("-" * 35)
for label, count in zip(age_bin_labels, age_counts):
    pct = count / max(1, len(df["anchor_age"].dropna())) * 100
    print(f"{label:<15} {int(count):<10} {pct:.1f}%")

# ── Raw Numbers: IE label ──────────────────────────────────────────────────────
print("\n=== IE LABEL DISTRIBUTION ===")
ie_counts = df["ie_label"].value_counts()
print(f"{'Label':<10} {'Count':<10} {'Percentage':<10}")
print("-" * 30)
for label, count in ie_counts.items():
    pct = count / N * 100
    print(f"{label:<10} {count:<10} {pct:.1f}%")

# ── Raw Numbers: Insurance Status ─────────────────────────────────────────────
if 'insurance' in df.columns:
    print("\n=== INSURANCE STATUS ===")
    ins_counts = df["insurance"].value_counts()
    print(f"{'Category':<30} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    for label, count in ins_counts.items():
        pct = count / N * 100
        print(f"{label:<30} {count:<10} {pct:.1f}%")

# ── Raw Numbers: Race / Ethnicity ─────────────────────────────────────────────
print("\n=== RACE / ETHNICITY ===")
race_counts = df["race_group"].value_counts()
print(f"{'Category':<30} {'Count':<10} {'Percentage':<10}")
print("-" * 50)
for label, count in race_counts.items():
    pct = count / N * 100
    print(f"{label:<30} {count:<10} {pct:.1f}%")

print("\nDone!")
