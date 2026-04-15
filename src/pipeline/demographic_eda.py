"""
Demographic EDA: case vs. control comparison with seaborn visualizations.

Produces side-by-side plots comparing IE-positive (case) and IE-negative
(control) groups across demographics and vitals, plus a summary CSV.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
VITAL_LABELS = {
    "temperature": "Temperature (°F)",
    "heartrate": "Heart Rate (bpm)",
    "resprate": "Respiratory Rate (breaths/min)",
    "o2sat": "O₂ Saturation (%)",
    "sbp": "Systolic BP (mmHg)",
    "dbp": "Diastolic BP (mmHg)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demographic EDA comparing IE cases vs. controls.",
    )
    parser.add_argument("--cohort-csv", type=str, required=True,
                        help="Path to cohort CSV")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for EDA output (results/eda)")
    return parser.parse_args()


def consolidate_race(r):
    r = str(r).upper()
    if r.startswith("WHITE"):
        return "White"
    if "BLACK" in r or "AFRICAN" in r:
        return "Black"
    if "HISPANIC" in r or "LATINO" in r:
        return "Hispanic"
    if "ASIAN" in r:
        return "Asian"
    if r in {"MISSING", ""}:
        return "Missing"
    if "UNKNOWN" in r or "UNABLE" in r:
        return "Unknown"
    return "Other"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.cohort_csv)
    df["group"] = df["ie_label"].map({1: "IE Case", 0: "Control"})
    df["race_group"] = df["race"].apply(consolidate_race)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {"IE Case": "#d62728", "Control": "#1f77b4"}

    # ── 1. Age distribution ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="anchor_age", hue="group", palette=palette,
                 kde=True, stat="density", common_norm=False, bins=30, alpha=0.5, ax=ax)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Density")
    ax.set_title("Age Distribution: Cases vs. Controls")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "age_distribution.png"), dpi=300)
    plt.close(fig)
    print("  Saved age_distribution.png")

    # ── 2. Gender comparison ────────────────────────────────────────────────
    gender_counts = (
        df.groupby(["group", "gender"]).size()
        .reset_index(name="count")
    )
    gender_counts["pct"] = gender_counts.groupby("group")["count"].transform(
        lambda x: x / x.sum() * 100
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=gender_counts, x="gender", y="pct", hue="group",
                palette=palette, ax=ax)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Gender Distribution: Cases vs. Controls")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "gender_comparison.png"), dpi=300)
    plt.close(fig)
    print("  Saved gender_comparison.png")

    # ── 3. Race comparison ──────────────────────────────────────────────────
    race_order = (
        df["race_group"].value_counts().index.tolist()
    )
    race_counts = (
        df.groupby(["group", "race_group"]).size()
        .reset_index(name="count")
    )
    race_counts["pct"] = race_counts.groupby("group")["count"].transform(
        lambda x: x / x.sum() * 100
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=race_counts, x="race_group", y="pct", hue="group",
                palette=palette, order=race_order, ax=ax)
    ax.set_xlabel("Race / Ethnicity")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Race Distribution: Cases vs. Controls")
    ax.legend(title="")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "race_comparison.png"), dpi=300)
    plt.close(fig)
    print("  Saved race_comparison.png")

    # ── 4. Acuity (ESI) comparison ──────────────────────────────────────────
    acuity_df = df.dropna(subset=["acuity"]).copy()
    acuity_df["acuity"] = acuity_df["acuity"].astype(int)
    acuity_counts = (
        acuity_df.groupby(["group", "acuity"]).size()
        .reset_index(name="count")
    )
    acuity_counts["pct"] = acuity_counts.groupby("group")["count"].transform(
        lambda x: x / x.sum() * 100
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=acuity_counts, x="acuity", y="pct", hue="group",
                palette=palette, ax=ax)
    ax.set_xlabel("ESI Acuity Level")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Acuity Distribution: Cases vs. Controls")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "acuity_comparison.png"), dpi=300)
    plt.close(fig)
    print("  Saved acuity_comparison.png")

    # ── 5. Vitals box plots (case vs. control) ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, vital in zip(axes.flatten(), VITALS):
        sns.boxplot(data=df, x="group", y=vital, palette=palette,
                    order=["Control", "IE Case"], ax=ax, fliersize=2)
        ax.set_xlabel("")
        ax.set_ylabel(VITAL_LABELS.get(vital, vital))
        ax.set_title(VITAL_LABELS.get(vital, vital))
    fig.suptitle("Vital Signs: Cases vs. Controls", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "vitals_boxplots.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved vitals_boxplots.png")

    # ── 6. Vitals violin plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, vital in zip(axes.flatten(), VITALS):
        sns.violinplot(data=df, x="group", y=vital, palette=palette,
                       order=["Control", "IE Case"], ax=ax, cut=0, inner="quartile")
        ax.set_xlabel("")
        ax.set_ylabel(VITAL_LABELS.get(vital, vital))
        ax.set_title(VITAL_LABELS.get(vital, vital))
    fig.suptitle("Vital Signs Distributions: Cases vs. Controls", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "vitals_violins.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved vitals_violins.png")

    # ── 7. Correlation heatmap by group ─────────────────────────────────────
    numeric_cols = ["anchor_age"] + VITALS + ["acuity"]
    for grp_name, grp_label in [("IE Case", "cases"), ("Control", "controls")]:
        sub = df[df["group"] == grp_name][numeric_cols].dropna()
        corr = sub.corr()
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title(f"Feature Correlation — {grp_name}s")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"correlation_{grp_label}.png"), dpi=300)
        plt.close(fig)
        print(f"  Saved correlation_{grp_label}.png")

    # ── 8. Summary CSV ──────────────────────────────────────────────────────
    rows = []
    for grp in ["Control", "IE Case"]:
        sub = df[df["group"] == grp]
        n = len(sub)
        row = {"Group": grp, "N": n}
        row["Male (%)"] = f"{(sub['gender'] == 'M').mean() * 100:.1f}"
        row["Age mean (SD)"] = f"{sub['anchor_age'].mean():.1f} ({sub['anchor_age'].std():.1f})"
        row["Age median [IQR]"] = (
            f"{sub['anchor_age'].median():.0f} "
            f"[{sub['anchor_age'].quantile(0.25):.0f}-{sub['anchor_age'].quantile(0.75):.0f}]"
        )
        for vital in VITALS:
            v = sub[vital].dropna()
            row[f"{vital} mean (SD)"] = f"{v.mean():.1f} ({v.std():.1f})"
            row[f"{vital} missing (%)"] = f"{sub[vital].isna().mean() * 100:.1f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "demographic_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved demographic_summary.csv")

    print("\nDone! EDA plots and summary saved.")


if __name__ == "__main__":
    main()
