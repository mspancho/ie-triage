"""
Table 2 — sample characteristics of the external-validation (MC-MED) cohort,
stratified by IE label. Reuses the same statistic functions as the internal
Table 1 so format and methodology match exactly (Hayes-Larson et al., 2019).
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from table1 import EXPOSURE, CATEGORICAL, summarise  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Table 2 (external-validation cohort characteristics).",
    )
    parser.add_argument("--cohort-csv", type=str, required=True,
                        help="Path to MC-MED external-validation cohort CSV")
    parser.add_argument("--csv-dir", type=str, required=True,
                        help="Directory for output CSV (data/csv)")
    parser.add_argument("--plot-dir", type=str, required=True,
                        help="Directory for output PNG (results/ext_val)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    df = pd.read_csv(args.cohort_csv)

    cat_levels = {
        var: sorted(df[var].dropna().unique().tolist())
        for var in CATEGORICAL + [EXPOSURE]
    }

    groups = {
        "Total": df,
        "ie_label = 0": df[df[EXPOSURE] == 0],
        "ie_label = 1": df[df[EXPOSURE] == 1],
    }

    columns = {name: summarise(sub, len(sub), cat_levels) for name, sub in groups.items()}

    row_counts = {k: len(v) for k, v in columns.items()}
    assert len(set(row_counts.values())) == 1, f"Row count mismatch: {row_counts}"

    labels = [r["label"] for r in columns["Total"]]
    result = pd.DataFrame({"Variable": labels})
    for col_name, rows in columns.items():
        result[col_name] = [r["stat"] for r in rows]

    print("\n" + "=" * 70)
    print("TABLE 2 — External-validation (MC-MED) cohort characteristics by ie_label")
    print("=" * 70 + "\n")
    print(result.to_string(index=False))

    csv_path = os.path.join(args.csv_dir, "table2_output.csv")
    result.to_csv(csv_path, index=False)
    print(f"\n→ Results saved to {csv_path}")

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
    png_path = os.path.join(args.plot_dir, "table2_output.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Results saved to {png_path}")


if __name__ == "__main__":
    main()
