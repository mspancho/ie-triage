"""
Table 1 summary statistics following Hayes-Larson et al. (2019) guidelines.

Columns:  Total | ie_label==0 | ie_label==1
Rows:     All analytic variables except chiefcomplaint
Cells:
  - Categorical : n (%) with missingness reported
  - Continuous  : mean (SD) and median [Q1–Q3] with missingness reported
No p-values per guideline recommendation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load data  ← update path as needed
# ------------------------------------------------------------------
df = pd.read_csv("proj1_ie_triage_cohort.csv")  # replace with your actual file path

# ------------------------------------------------------------------
# 2. Variable classification  (chiefcomplaint excluded per note 2)
# ------------------------------------------------------------------
EXPOSURE = "ie_label"

CATEGORICAL = [
    "gender",
    "race",
]

CONTINUOUS = [
    "anchor_age",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "acuity",
]

# ------------------------------------------------------------------
# 3. Build a global category registry so every group uses the
#    same set of levels (categories present in ANY group), avoiding
#    row-length mismatches when a level is absent in a subgroup.
# ------------------------------------------------------------------

CAT_LEVELS = {}  # var -> sorted list of all levels seen across full df
for var in CATEGORICAL + [EXPOSURE]:
    CAT_LEVELS[var] = sorted(df[var].dropna().unique().tolist())

# ------------------------------------------------------------------
# 4. Helper functions
# ------------------------------------------------------------------

def cat_stats(series, total_n, all_levels):
    """
    Return value-count rows for a categorical variable.
    Uses `all_levels` so every group produces the same rows,
    even if a level is absent in that subgroup (count = 0).
    """
    n_missing = series.isna().sum()
    pct_missing = 100 * n_missing / total_n
    counts = series.value_counts(dropna=True)
    rows = []
    for val in all_levels:
        cnt = counts.get(val, 0)
        pct = 100 * cnt / total_n
        rows.append({"label": f"  {val}", "stat": f"{cnt} ({pct:.0f}%)"})
    if "Missing" not in all_levels:
        rows.append({"label": "  Missing", "stat": f"{n_missing} ({pct_missing:.0f}%)"})
    return rows


def cont_stats(series, total_n):
    """Return mean (SD), median [Q1–Q3], and missingness for a continuous variable."""
    n_missing = series.isna().sum()
    pct_missing = 100 * n_missing / total_n
    valid = series.dropna()
    mean = valid.mean()
    sd   = valid.std()
    med  = valid.median()
    q1   = valid.quantile(0.25)
    q3   = valid.quantile(0.75)
    rows = [
        {"label": "  Mean (SD)",      "stat": f"{mean:.1f} ({sd:.1f})"},
        {"label": "  Median [Q1–Q3]", "stat": f"{med:.1f} [{q1:.1f}–{q3:.1f}]"},
        {"label": "  Missing",        "stat": f"{n_missing} ({pct_missing:.0f}%)"},
    ]
    return rows


def summarise(subdf, total_n):
    """Build a list of (label, stat) rows for one column group."""
    records = []

    # --- N for this group ---
    records.append({"label": "N", "stat": str(total_n)})

    # --- Exposure variable itself (categorical) ---
    records.append({"label": EXPOSURE, "stat": ""})
    records.extend(cat_stats(subdf[EXPOSURE], total_n, CAT_LEVELS[EXPOSURE]))

    # --- Categorical variables ---
    for var in CATEGORICAL:
        records.append({"label": var, "stat": ""})
        records.extend(cat_stats(subdf[var], total_n, CAT_LEVELS[var]))

    # --- Continuous variables ---
    for var in CONTINUOUS:
        records.append({"label": var, "stat": ""})
        records.extend(cont_stats(subdf[var], total_n))

    return records


# ------------------------------------------------------------------
# 5. Build columns
# ------------------------------------------------------------------
groups = {
    "Total":        df,
    "ie_label = 0": df[df[EXPOSURE] == 0],
    "ie_label = 1": df[df[EXPOSURE] == 1],
}

columns = {}
for col_name, subdf in groups.items():
    columns[col_name] = summarise(subdf, len(subdf))

# Sanity-check: all groups must produce the same number of rows
row_counts = {k: len(v) for k, v in columns.items()}
assert len(set(row_counts.values())) == 1, \
    f"Row count mismatch across groups: {row_counts}"

# ------------------------------------------------------------------
# 6. Assemble into a single DataFrame
# ------------------------------------------------------------------
labels = [r["label"] for r in columns["Total"]]
result = pd.DataFrame({"Variable": labels})

for col_name, rows in columns.items():
    result[col_name] = [r["stat"] for r in rows]

# ------------------------------------------------------------------
# 7. Output
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLE 1 — Sample characteristics by ie_label")
print("Cells: n (%) for categorical; mean (SD) and median [Q1–Q3] for continuous")
print("Denominator for % includes subjects with missing values")
print("No inferential statistics per Hayes-Larson et al. (2019)")
print("=" * 70 + "\n")
print(result.to_string(index=False))

# Also save to CSV for easy copy-paste into Excel / Word
result.to_csv("table1_output.csv", index=False)
print("\n→ Results saved to table1_output.csv")

# ------------------------------------------------------------------
# 8. Save table as PNG
# ------------------------------------------------------------------
fig_height = max(8, len(result) * 0.22)
fig, ax = plt.subplots(figsize=(14, fig_height))
ax.axis("off")

# Render the table with row labels and data
mpl_table = ax.table(
    cellText=result.values,
    colLabels=result.columns,
    cellLoc="left",
    loc="center",
)

mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(8)
mpl_table.scale(1.1, 1.1)

for (row, col), cell in mpl_table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#f2f2f2")
    if col == 0:
        cell.set_text_props(weight="bold")

plt.tight_layout()
plt.savefig("table1_output.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("→ Results saved to table1_output.png")