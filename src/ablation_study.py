"""
Ablation Study Module.
Evaluates model retrainability and predictive performance degradation under 4 distinct feature subset conditions:
Experiment A (All 47 Features), Experiment B (Remove Top 5 Features),
Experiment C (Use ONLY Top 10 Features), Experiment D (Remove Bottom 20 Features).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN
from src.train import train_single_experiment


def run_ablation_study():
    """
    Executes 4-experiment ablation study:
    - Exp A: Full Feature Set (47 features)
    - Exp B: Remove Top 5 Features (Mask core drivers)
    - Exp C: Use ONLY Top 10 Features (Sparse high-impact model)
    - Exp D: Remove Bottom 20 Features (Pruned low-impact features)
    """
    print("\n" + "=" * 65)
    print("                ABLATION EXPERIMENTATION STUDY")
    print("=" * 65)

    comp_path = config.REPORTS_DIR / "feature_attribution_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError("feature_attribution_comparison.csv not found! Run Phase 13 first.")

    df_comp = pd.read_csv(comp_path).sort_values(by="Consensus_Rank", ascending=True)

    all_features = df_comp["Feature"].tolist()
    top5_features = df_comp.iloc[:5]["Feature"].tolist()
    top10_features = df_comp.iloc[:10]["Feature"].tolist()
    bottom20_features = df_comp.iloc[-20:]["Feature"].tolist()

    df_train = pd.read_csv(config.PROCESSED_TRAIN_FILE)
    df_val = pd.read_csv(config.PROCESSED_VAL_FILE)
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)

    experiments = [
        {
            "Exp_ID": "Exp A: Full Features",
            "Description": "All 47 Transformed Features",
            "Features": all_features,
        },
        {
            "Exp_ID": "Exp B: Remove Top 5",
            "Description": "Remove tenure, Contract, TotalCharges, MonthlyCharges",
            "Features": [f for f in all_features if f not in top5_features],
        },
        {
            "Exp_ID": "Exp C: Top 10 Only",
            "Description": "Use ONLY Top 10 Consensus Features",
            "Features": top10_features,
        },
        {
            "Exp_ID": "Exp D: Remove Bottom 20",
            "Description": "Prune 20 Low-Impact Features",
            "Features": [f for f in all_features if f not in bottom20_features],
        },
    ]

    ablation_results = []

    print(f"\n[1] EXECUTING 4 ABLATION TRAINING EXPERIMENTS...")

    for exp in experiments:
        exp_id = exp["Exp_ID"]
        feats = exp["Features"]

        X_tr = df_train[feats].values.astype(np.float32)
        y_tr = df_train[config.TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

        X_v = df_val[feats].values.astype(np.float32)
        y_v = df_val[config.TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

        X_te = df_test[feats].values.astype(np.float32)
        y_te = df_test[config.TARGET_COLUMN].values.astype(np.int32)

        input_dim = X_tr.shape[1]

        # Calculate pos_weight
        pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

        # Train model for ablation condition
        model = TabularChurnNN(input_dim=input_dim, hidden_dim_1=128, hidden_dim_2=64).to(config.DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=config.DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Quick train loop for ablation
        for epoch in range(15):
            model.train()
            optimizer.zero_grad()
            logits = model(torch.tensor(X_tr, device=config.DEVICE))
            loss = criterion(logits, torch.tensor(y_tr, device=config.DEVICE))
            loss.backward()
            optimizer.step()

        # Evaluate Test Performance
        model.eval()
        with torch.no_grad():
            t_test = torch.tensor(X_te, device=config.DEVICE)
            probs = model.predict_proba(t_test).cpu().numpy()[:, 1]
            preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)
        auc = roc_auc_score(y_te, probs)

        print(f"\n  * [{exp_id}] ({input_dim} features):")
        print(f"    - Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

        ablation_results.append(
            {
                "Experiment": exp_id,
                "Num_Features": input_dim,
                "Description": exp["Description"],
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1_Score": round(float(f1), 4),
                "ROC_AUC": round(float(auc), 4),
            }
        )

    # Save CSV
    df_ablation = pd.DataFrame(ablation_results)
    ablation_csv_path = config.REPORTS_DIR / "ablation_study_results.csv"
    df_ablation.to_csv(ablation_csv_path, index=False)
    print(f"\n[SAVED] Ablation Study Table -> {ablation_csv_path}")

    # Save Plot
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    x = np.arange(len(df_ablation))
    width = 0.25

    plt.bar(x - width, df_ablation["ROC_AUC"], width=width, label="ROC-AUC", color="#2b5c8f")
    plt.bar(x, df_ablation["F1_Score"], width=width, label="F1-Score", color="#e67e22")
    plt.bar(x + width, df_ablation["Recall"], width=width, label="Recall", color="#27ae60")

    plt.title("Ablation Study: Performance Across Feature Subset Experiments", fontsize=11, fontweight="bold", pad=10)
    plt.ylabel("Performance Metric Score", fontsize=9)
    plt.xticks(x, df_ablation["Experiment"], fontsize=9)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=9, loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    ablation_plot_path = figures_dir / "20_ablation_study_comparison.png"
    plt.savefig(ablation_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 20 -> {ablation_plot_path}\n")

    print("=" * 65 + "\n")
    return df_ablation


if __name__ == "__main__":
    run_ablation_study()
