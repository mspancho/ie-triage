"""
Feature importance visualization for IE triage ML models.

For each (feature set, model) combination, computes mean ± SD feature
importances across 5-fold CV and saves horizontal bar charts of the top
20 features. Importance is |coefficient| for logistic regression and
tree-based feature_importances_ for Random Forest / XGBoost.
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

N_SPLITS = 5
RANDOM_STATE = 42
TOP_N = 20

FEATURE_FILES = {
    "Option1_Categorical": "features_option1_categorical.npz",
    "Option2_PCA_Embeddings": "features_option2_pca_embeddings.npz",
    "Option3_Hybrid": "features_option3_hybrid.npz",
    "Option4_No_CC": "features_option4_no_cc.npz",
}

MODEL_COLORS = {
    "LogisticRegression_L1": "#1f77b4",
    "RandomForest": "#ff7f0e",
    "XGBoost": "#2ca02c",
}

IMPORTANCE_XLABEL = {
    "LogisticRegression_L1": "Mean |Coefficient| (± SD)",
    "RandomForest": "Mean Gini Importance (± SD)",
    "XGBoost": "Mean Feature Importance (± SD)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature importance plots for IE triage ML pipeline."
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing feature .npz files (data/npz)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for feature importance plots")
    return parser.parse_args()


def get_models(n_pos, n_neg):
    scale_pos_weight = n_neg / n_pos
    return {
        "LogisticRegression_L1": LogisticRegression(
            penalty="l1", solver="saga", class_weight="balanced",
            max_iter=5000, random_state=RANDOM_STATE, C=1.0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def extract_importance(model, model_name):
    if model_name == "LogisticRegression_L1":
        return np.abs(model.coef_[0])
    return model.feature_importances_


def compute_cv_importances(X, y, model_template, model_name):
    """Returns (mean, std) of feature importances averaged across CV training folds."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_importances = []

    for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
        X_train = X[train_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = type(model_template)(**model_template.get_params())
        model.fit(X_train_scaled, y_train)
        fold_importances.append(extract_importance(model, model_name))
        print(f"      Fold {fold_idx + 1}/{N_SPLITS} done", end="\r", flush=True)

    print()
    fold_importances = np.array(fold_importances)  # (N_SPLITS, n_features)
    return fold_importances.mean(axis=0), fold_importances.std(axis=0)


def save_plot(mean_imp, std_imp, feature_names, option_name, model_name, output_dir):
    top_n = min(TOP_N, len(feature_names))
    # Top features sorted descending, then reversed so highest maps to top of chart
    sorted_idx = np.argsort(mean_imp)[::-1][:top_n]
    plot_idx = sorted_idx[::-1]

    means = mean_imp[plot_idx]
    stds = std_imp[plot_idx]
    names = [str(feature_names[i]) for i in plot_idx]
    y_pos = np.arange(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.42)))
    ax.barh(
        y_pos, means, xerr=stds, align="center",
        color=MODEL_COLORS.get(model_name, "#7f7f7f"),
        alpha=0.80, capsize=3,
        ecolor="dimgray", error_kw={"linewidth": 0.8},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(IMPORTANCE_XLABEL.get(model_name, "Mean Importance (± SD)"), fontsize=11)
    ax.set_title(
        f"Feature Importance — {option_name.replace('_', ' ')} — {model_name}\n"
        f"Top {top_n} features  |  mean ± SD across {N_SPLITS}-fold CV",
        fontsize=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"feat_importance_{option_name}_{model_name}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for option_name, feature_file in FEATURE_FILES.items():
        fpath = os.path.join(args.data_dir, feature_file)
        if not os.path.isfile(fpath):
            print(f"  Skipping {option_name}: {fpath} not found")
            continue

        print(f"\n{'='*70}\n  {option_name}\n{'='*70}")
        data = np.load(fpath, allow_pickle=True)
        X = np.nan_to_num(data["X"], nan=0.0)
        y = data["y"]
        feature_names = data["feature_names"]

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        print(f"  X: {X.shape}  pos={n_pos}  neg={n_neg}  features={len(feature_names)}")

        models = get_models(n_pos, n_neg)

        for model_name, model_template in models.items():
            print(f"  → {model_name}")
            mean_imp, std_imp = compute_cv_importances(X, y, model_template, model_name)
            fname = save_plot(mean_imp, std_imp, feature_names,
                              option_name, model_name, args.output_dir)
            print(f"    Saved: {fname}")

    print(f"\nFeature importance plots saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
