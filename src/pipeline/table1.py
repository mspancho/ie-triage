"""
Table 1 summary statistics following Hayes-Larson et al. (2019) guidelines.

Columns:  Total | ie_label==0 | ie_label==1
Cells:
  - Categorical: n (%) with missingness reported
  - Continuous:  mean (SD) and median [Q1-Q3] with missingness reported
No p-values per guideline recommendation.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


EXPOSURE = "ie_label"
CATEGORICAL = ["gender", "race"]
CONTINUOUS = ["anchor_age", "temperature", "heartrate", "resprate",
              "o2sat", "sbp", "dbp", "acuity"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Table 1 descriptive statistics.")
    parser.add_argument("--cohort-csv", type=str, required=True,
                        help="Path to cohort CSV")
    parser.add_argument("--csv-dir", type=str, required=True,
                        help="Directory for output CSV (data/csv)")
    parser.add_argument("--plot-dir", type=str, required=True,
                        help="Directory for output PNG (results/eda)")
    return parser.parse_args()


def cat_stats(series, total_n, all_levels):
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
    n_missing = series.isna().sum()
    pct_missing = 100 * n_missing / total_n
    valid = series.dropna()
    mean, sd = valid.mean(), valid.std()
    med = valid.median()
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    return [
        {"label": "  Mean (SD)", "stat": f"{mean:.1f} ({sd:.1f})"},
        {"label": "  Median [Q1-Q3]", "stat": f"{med:.1f} [{q1:.1f}-{q3:.1f}]"},
        {"label": "  Missing", "stat": f"{n_missing} ({pct_missing:.0f}%)"},
    ]


def summarise(subdf, total_n, cat_levels):
    records = [{"label": "N", "stat": str(total_n)}]
    records.append({"label": EXPOSURE, "stat": ""})
    records.extend(cat_stats(subdf[EXPOSURE], total_n, cat_levels[EXPOSURE]))
    for var in CATEGORICAL:
        records.append({"label": var, "stat": ""})
        records.extend(cat_stats(subdf[var], total_n, cat_levels[var]))
    for var in CONTINUOUS:
        records.append({"label": var, "stat": ""})
        records.extend(cont_stats(subdf[var], total_n))
    return records


def main():
    args = parse_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    df = pd.read_csv(args.cohort_csv)

    cat_levels = {}
    for var in CATEGORICAL + [EXPOSURE]:
        cat_levels[var] = sorted(df[var].dropna().unique().tolist())

    groups = {
        "Total": df,
        "ie_label = 0": df[df[EXPOSURE] == 0],
        "ie_label = 1": df[df[EXPOSURE] == 1],
    }

    columns = {}
    for col_name, subdf in groups.items():
        columns[col_name] = summarise(subdf, len(subdf), cat_levels)

    row_counts = {k: len(v) for k, v in columns.items()}
    assert len(set(row_counts.values())) == 1, f"Row count mismatch: {row_counts}"

    labels = [r["label"] for r in columns["Total"]]
    result = pd.DataFrame({"Variable": labels})
    for col_name, rows in columns.items():
        result[col_name] = [r["stat"] for r in rows]

    print("\n" + "=" * 70)
    print("TABLE 1 — Sample characteristics by ie_label")
    print("=" * 70 + "\n")
    print(result.to_string(index=False))

    csv_path = os.path.join(args.csv_dir, "table1_output.csv")
    result.to_csv(csv_path, index=False)
    print(f"\n→ Results saved to {csv_path}")

    # Save as PNG
    fig_height = max(8, len(result) * 0.22)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    mpl_table = ax.table(cellText=result.values, colLabels=result.columns,
                         cellLoc="left", loc="center")
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
    png_path = os.path.join(args.plot_dir, "table1_output.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Results saved to {png_path}")


if __name__ == "__main__":
    main()
