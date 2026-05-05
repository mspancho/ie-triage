"""
External-validation ML pipeline.

Trains a Random Forest on the FULL internal Option 1 (categorical chief
complaint) cohort with the exact same hyperparameters used in the internal
5-fold CV pipeline, then evaluates it on the MC-MED external-validation
cohort. Produces:
  * Held-out external metrics (Sensitivity, Specificity, PPV, NPV,
    Precision, Recall, AUROC, AUPRC) at threshold 0.5.
  * ROC, Precision-Recall, and Decision Curve Analysis plots.
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
OPTION_NAME = "Option1_Categorical"
MODEL_NAME = "RandomForest"


def parse_args():
    parser = argparse.ArgumentParser(
        description="External validation ML pipeline (RF, categorical CC, MC-MED).",
    )
    parser.add_argument("--internal-npz", type=str, required=True,
                        help="Path to internal Option 1 npz used for training")
    parser.add_argument("--external-npz", type=str, required=True,
                        help="Path to MC-MED external Option 1 npz")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for results CSV and plots (results/ext_val)")
    return parser.parse_args()


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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


def plot_roc(y_true, y_prob, option_name, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="#ff7f0e",
            label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — External Validation — {option_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"roc_{option_name}.png"), dpi=300)
    plt.close(fig)


def plot_pr(y_true, y_prob, option_name, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, color="#ff7f0e",
            label=f"{model_name} (AP={ap:.3f})", linewidth=2)
    prevalence = y_true.mean()
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — External Validation — {option_name}", fontsize=14)
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


def plot_dca(y_true, y_prob, option_name, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    thresholds = np.linspace(0.01, 0.99, 200)
    prevalence = y_true.mean()
    treat_all_nb = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, treat_all_nb, color="gray", linestyle="--", alpha=0.6, label="Treat All")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.4, label="Treat None")
    nb = [net_benefit(y_true, y_prob, t) for t in thresholds]
    ax.plot(thresholds, nb, color="#ff7f0e", label=model_name, linewidth=2)
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(f"Decision Curve Analysis — External Validation — {option_name}", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 0.5]); ax.set_ylim([-0.05, max(0.15, prevalence + 0.05)])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dca_{option_name}.png"), dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading internal training set (Option 1, categorical)...")
    train = np.load(args.internal_npz, allow_pickle=True)
    X_train = np.nan_to_num(train["X"], nan=0.0)
    y_train = train["y"]
    train_feature_names = [str(n) for n in train["feature_names"]]
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    print(f"  internal X: {X_train.shape}  pos={n_pos}  neg={n_neg}")

    print("Loading external validation set (MC-MED)...")
    ext = np.load(args.external_npz, allow_pickle=True)
    X_ext = np.nan_to_num(ext["X"], nan=0.0)
    y_ext = ext["y"]
    ext_feature_names = [str(n) for n in ext["feature_names"]]
    print(f"  external X: {X_ext.shape}  pos={int(np.sum(y_ext == 1))}  neg={int(np.sum(y_ext == 0))}")

    if train_feature_names != ext_feature_names:
        raise ValueError(
            "Internal and external feature_names do not match. "
            "Re-run feature engineering — schema must be identical for external validation."
        )

    # ── Standard scaler is fit on training only and applied to external ────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_ext_scaled = scaler.transform(X_ext)

    # ── Random Forest with the EXACT hyperparameters from the internal pipeline ─
    print(f"Training {MODEL_NAME} on full internal Option 1 cohort...")
    model = RandomForestClassifier(
        n_estimators=500, max_depth=None, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    print("Predicting on external cohort...")
    y_prob = model.predict_proba(X_ext_scaled)[:, 1]
    metrics = compute_metrics(y_ext, y_prob)

    print(f"  AUROC = {metrics['AUROC']:.3f}")
    print(f"  AUPRC = {metrics['AUPRC']:.3f}")
    print(f"  Sensitivity = {metrics['Sensitivity']:.3f}")
    print(f"  Specificity = {metrics['Specificity']:.3f}")
    print(f"  PPV / NPV   = {metrics['PPV']:.3f} / {metrics['NPV']:.3f}")

    # ── Save metrics in the same shape as internal cv_results_summary.csv ──
    summary = pd.DataFrame([
        {"Option": OPTION_NAME, "Model": MODEL_NAME, "Metric": k, "Value": v}
        for k, v in metrics.items()
    ])
    summary_path = os.path.join(args.output_dir, "extval_results_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved external metrics to {summary_path}")

    print("Generating ROC, PR, DCA plots...")
    plot_roc(y_ext, y_prob, OPTION_NAME, MODEL_NAME, args.output_dir)
    plot_pr(y_ext, y_prob, OPTION_NAME, MODEL_NAME, args.output_dir)
    plot_dca(y_ext, y_prob, OPTION_NAME, MODEL_NAME, args.output_dir)

    # ── Persist the held-out predictions for reproducibility / later analysis ─
    preds_path = os.path.join(args.output_dir, "extval_predictions.csv")
    pd.DataFrame({
        "stay_id": ext["stay_ids"],
        "y_true": y_ext,
        "y_prob": y_prob,
    }).to_csv(preds_path, index=False)
    print(f"Saved per-visit predictions to {preds_path}")

    print("Done!")


if __name__ == "__main__":
    main()
