"""
ML pipeline for IE triage prediction.

Models: Logistic Regression (L1), Random Forest, XGBoost
Evaluation: 5-fold stratified CV
Metrics: Precision, Recall, AUROC, AUPRC, NPV, PPV, Sensitivity, Specificity
Plots: ROC curves, PR curves, Decision Curve Analysis (per model)
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

N_SPLITS = 5
RANDOM_STATE = 42

FEATURE_FILES = {
    "Option1_Categorical": "features_option1_categorical.npz",
    "Option2_PCA_Embeddings": "features_option2_pca_embeddings.npz",
    "Option3_Hybrid": "features_option3_hybrid.npz",
    "Option4_No_CC": "features_option4_no_cc.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="ML pipeline for IE triage prediction.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing feature .npz files (data/npz)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for results CSV and plots (results/ml)")
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


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    return {
        "Sensitivity": sensitivity, "Specificity": specificity,
        "PPV": ppv, "NPV": npv, "Precision": ppv, "Recall": sensitivity,
        "AUROC": auroc, "AUPRC": auprc,
    }


def plot_roc_curves(results, option_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"LogisticRegression_L1": "#1f77b4", "RandomForest": "#ff7f0e", "XGBoost": "#2ca02c"}
    for model_name, folds in results.items():
        all_y = np.concatenate([f["y_true"] for f in folds])
        all_prob = np.concatenate([f["y_prob"] for f in folds])
        fpr, tpr, _ = roc_curve(all_y, all_prob)
        auc = roc_auc_score(all_y, all_prob)
        ax.plot(fpr, tpr, color=colors.get(model_name, "gray"),
                label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {option_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"roc_{option_name}.png"), dpi=300)
    plt.close(fig)


def plot_pr_curves(results, option_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"LogisticRegression_L1": "#1f77b4", "RandomForest": "#ff7f0e", "XGBoost": "#2ca02c"}
    for model_name, folds in results.items():
        all_y = np.concatenate([f["y_true"] for f in folds])
        all_prob = np.concatenate([f["y_prob"] for f in folds])
        precision, recall, _ = precision_recall_curve(all_y, all_prob)
        ap = average_precision_score(all_y, all_prob)
        ax.plot(recall, precision, color=colors.get(model_name, "gray"),
                label=f"{model_name} (AP={ap:.3f})", linewidth=2)
    prevalence = all_y.mean()
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {option_name}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pr_{option_name}.png"), dpi=300)
    plt.close(fig)


def net_benefit(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return (tp / n) - (fp / n) * (threshold / (1 - threshold)) if threshold < 1 else 0.0


def plot_decision_curves(results, option_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"LogisticRegression_L1": "#1f77b4", "RandomForest": "#ff7f0e", "XGBoost": "#2ca02c"}
    thresholds = np.linspace(0.01, 0.99, 200)
    all_y_any = np.concatenate([f["y_true"] for folds in results.values() for f in folds])
    prevalence = all_y_any.mean()
    treat_all_nb = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, treat_all_nb, color="gray", linestyle="--", alpha=0.6, label="Treat All")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.4, label="Treat None")
    for model_name, folds in results.items():
        all_y = np.concatenate([f["y_true"] for f in folds])
        all_prob = np.concatenate([f["y_prob"] for f in folds])
        nb = [net_benefit(all_y, all_prob, t) for t in thresholds]
        ax.plot(thresholds, nb, color=colors.get(model_name, "gray"),
                label=model_name, linewidth=2)
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(f"Decision Curve Analysis — {option_name}", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 0.5]); ax.set_ylim([-0.05, max(0.15, prevalence + 0.05)])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dca_{option_name}.png"), dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_summary = []

    for option_name, feature_file in FEATURE_FILES.items():
        fpath = os.path.join(args.data_dir, feature_file)
        if not os.path.isfile(fpath):
            print(f"Skipping {option_name}: {fpath} not found")
            continue

        print(f"\n{'='*70}\n  {option_name}\n{'='*70}")
        data = np.load(fpath, allow_pickle=True)
        X = np.nan_to_num(data["X"], nan=0.0)
        y = data["y"]

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        print(f"  X: {X.shape}, y: pos={n_pos}, neg={n_neg}")

        models = get_models(n_pos, n_neg)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        option_results = {name: [] for name in models}

        for model_name, model in models.items():
            print(f"\n  → {model_name}")
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_train_scaled, y_train)
                y_prob = model_clone.predict_proba(X_test_scaled)[:, 1]

                metrics = compute_metrics(y_test, y_prob)
                metrics["Fold"] = fold_idx + 1
                fold_metrics.append(metrics)
                option_results[model_name].append({"y_true": y_test, "y_prob": y_prob})

                print(f"    Fold {fold_idx+1}: AUROC={metrics['AUROC']:.3f}  "
                      f"AUPRC={metrics['AUPRC']:.3f}  "
                      f"Sens={metrics['Sensitivity']:.3f}  "
                      f"Spec={metrics['Specificity']:.3f}")

            fold_df = pd.DataFrame(fold_metrics)
            mean_metrics = fold_df.drop(columns=["Fold"]).mean()
            std_metrics = fold_df.drop(columns=["Fold"]).std()
            print(f"    Mean AUROC: {mean_metrics['AUROC']:.3f} ± {std_metrics['AUROC']:.3f}")
            print(f"    Mean AUPRC: {mean_metrics['AUPRC']:.3f} ± {std_metrics['AUPRC']:.3f}")

            for metric_name in mean_metrics.index:
                all_summary.append({
                    "Option": option_name, "Model": model_name,
                    "Metric": metric_name,
                    "Mean": mean_metrics[metric_name],
                    "Std": std_metrics[metric_name],
                })

        print(f"\n  Generating plots for {option_name} …")
        plot_roc_curves(option_results, option_name, args.output_dir)
        plot_pr_curves(option_results, option_name, args.output_dir)
        plot_decision_curves(option_results, option_name, args.output_dir)

    summary_df = pd.DataFrame(all_summary)
    summary_path = os.path.join(args.output_dir, "cv_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved CV results to {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
