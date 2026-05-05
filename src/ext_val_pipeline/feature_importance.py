"""
Feature importance for the external-validation Random Forest.

Trains a single RF (same hyperparameters as the internal pipeline) on the
full internal Option 1 (categorical) cohort and saves a top-20 feature
importance bar chart. There is no CV here — a single training fit on the
full internal cohort is what produces the model used for external
validation, so error bars (which require multiple folds) do not apply.
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TOP_N = 20
OPTION_NAME = "Option1_Categorical"
MODEL_NAME = "RandomForest"
MODEL_COLOR = "#ff7f0e"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature importance for the external-validation RF."
    )
    parser.add_argument("--internal-npz", type=str, required=True,
                        help="Path to internal Option 1 npz used for training")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the importance plot")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.internal_npz, allow_pickle=True)
    X = np.nan_to_num(data["X"], nan=0.0)
    y = data["y"]
    feature_names = [str(n) for n in data["feature_names"]]
    print(f"  X: {X.shape}  pos={int(np.sum(y == 1))}  neg={int(np.sum(y == 0))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=500, max_depth=None, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_scaled, y)
    importances = model.feature_importances_

    top_n = min(TOP_N, len(feature_names))
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    plot_idx = sorted_idx[::-1]
    values = importances[plot_idx]
    names = [feature_names[i] for i in plot_idx]
    y_pos = np.arange(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.42)))
    ax.barh(y_pos, values, align="center", color=MODEL_COLOR, alpha=0.80)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.set_title(
        f"Feature Importance — External Validation — {OPTION_NAME} — {MODEL_NAME}\n"
        f"Top {top_n} features  |  trained on full internal Option 1 cohort",
        fontsize=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = os.path.join(
        args.output_dir, f"feat_importance_{OPTION_NAME}_{MODEL_NAME}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
