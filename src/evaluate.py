"""
Model Evaluation Module.
Evaluates the saved PyTorch model checkpoint on the unseen Test dataset (15% split).
Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC),
generates annotated confusion matrices, ROC curves, and Precision-Recall curves,
and translates technical metrics into business interpretation.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def load_test_data_and_model() -> tuple:
    """Loads processed test CSV and winning PyTorch model checkpoint."""
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)
    X_test = df_test.drop(columns=[config.TARGET_COLUMN]).values.astype(np.float32)
    y_test = df_test[config.TARGET_COLUMN].values.astype(np.int32)

    input_dim = X_test.shape[1]
    
    # Read architecture parameters from experiment tracking or default to winning architecture [128, 64]
    exp_file = config.EXPERIMENT_LOG_FILE
    hidden_1, hidden_2 = 128, 64  # Default winning Exp2 architecture
    if exp_file.exists():
        exp_df = pd.read_csv(exp_file)
        best_row = exp_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]
        arch_str = best_row["Architecture"]  # e.g. "[128-64]"
        dims = [int(d) for d in arch_str.strip("[]").split("-")]
        hidden_1, hidden_2 = dims[0], dims[1]

    model = TabularChurnNN(
        input_dim=input_dim,
        hidden_dim_1=hidden_1,
        hidden_dim_2=hidden_2,
    ).to(config.DEVICE)

    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=config.DEVICE))
    model.eval()

    return X_test, y_test, model, df_test.drop(columns=[config.TARGET_COLUMN]).columns.tolist()


def evaluate_model_on_test_set():
    """
    Executes full evaluation on 1,057 test samples.
    Calculates metrics, plots figures, saves json report.
    """
    print("\n" + "=" * 65)
    print("               TEST SET MODEL EVALUATION REPORT")
    print("=" * 65)

    X_test, y_test, model, feature_names = load_test_data_and_model()

    # Predict Probabilities
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, device=config.DEVICE)
        probs_2d = model.predict_proba(X_tensor).cpu().numpy()
        y_probs = probs_2d[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

    # Classification Metrics
    acc = accuracy_score(y_test, y_preds)
    prec = precision_score(y_test, y_preds, zero_division=0)
    rec = recall_score(y_test, y_preds, zero_division=0)
    f1 = f1_score(y_test, y_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_preds)

    tn, fp, fn, tp = cm.ravel()

    metrics_dict = {
        "num_test_samples": len(y_test),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    print(f"\n[1] TEST PERFORMANCE METRICS (N = {len(y_test):,} samples):")
    print(f"    - Accuracy:              {acc * 100:.2f}%")
    print(f"    - Precision:             {prec * 100:.2f}%")
    print(f"    - Recall (Sensitivity):  {rec * 100:.2f}%  <-- KEY RETENTION METRIC")
    print(f"    - F1-Score:              {f1:.4f}")
    print(f"    - ROC-AUC:               {roc_auc:.4f}")
    print(f"    - PR-AUC:                {pr_auc:.4f}")

    print(f"\n[2] CONFUSION MATRIX BREAKDOWN:")
    print(f"    - True Negatives  (Correct Non-Churners): {tn:>4}")
    print(f"    - False Positives (False Churn Alarm):    {fp:>4}")
    print(f"    - False Negatives (Missed Churners):      {fn:>4}  <-- BUSINESS COST")
    print(f"    - True Positives  (Caught Churners):      {tp:>4}")

    # Business Metrics Interpretation
    print(f"\n[3] BUSINESS INTERPRETATION & ACTIONABILITY:")
    print(f"    * Out of {y_test.sum()} actual churners in test set, the model correctly identified {tp} ({rec*100:.1f}% Recall).")
    print(f"    * When the model flags a customer for retention intervention, it is correct {prec*100:.1f}% of the time.")
    print(f"    * False Negative Rate is {(fn/y_test.sum())*100:.1f}%, representing uncaptured churn risk.")

    # Save JSON report
    metrics_file = config.REPORTS_DIR / "test_evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\n[SAVED] Test Evaluation Metrics -> {metrics_file}")

    # Plot Visualizations
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title("Test Set Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("Actual Class", fontsize=10)
    plt.tight_layout()
    cm_path = figures_dir / "07_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 7 -> {cm_path}")

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2b5c8f", linewidth=2.5, label=f"PyTorch Neural Net (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = figures_dir / "08_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 8 -> {roc_path}")

    # Plot 3: Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color="#d9534f", linewidth=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.4f})")
    baseline_pr = y_test.sum() / len(y_test)
    ax.axhline(y=baseline_pr, color="gray", linestyle="--", linewidth=1, label=f"Baseline Churn Rate ({baseline_pr:.2f})")
    ax.set_title("Precision-Recall (PR) Curve", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=9)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    pr_path = figures_dir / "09_precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 9 -> {pr_path}\n")

    print("=" * 65 + "\n")
    return metrics_dict


if __name__ == "__main__":
    evaluate_model_on_test_set()
