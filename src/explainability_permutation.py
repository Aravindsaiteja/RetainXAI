"""
Permutation Feature Importance Module.
Computes model-agnostic baseline feature importance by measuring the drop in validation/test ROC-AUC
when individual features are randomly shuffled across test instances.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def run_permutation_importance_analysis(n_repeats: int = 10):
    """
    Executes Permutation Feature Importance workflow:
    1. Loads processed test dataset and PyTorch neural network checkpoint.
    2. Calculates baseline Test ROC-AUC score.
    3. Sequentially shuffles each feature n_repeats times and computes metric degradation.
    4. Saves ranking table and visualization plot.
    """
    print("\n" + "=" * 65)
    print("         PERMUTATION FEATURE IMPORTANCE ANALYSIS")
    print("=" * 65)

    # 1. Load Data
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)
    X_test_df = df_test.drop(columns=[config.TARGET_COLUMN])
    y_test = df_test[config.TARGET_COLUMN].values

    feature_names = X_test_df.columns.tolist()
    input_dim = len(feature_names)
    X_test_np = X_test_df.values

    # 2. Load PyTorch Model
    exp_file = config.EXPERIMENT_LOG_FILE
    hidden_1, hidden_2 = 128, 64
    if exp_file.exists():
        exp_df = pd.read_csv(exp_file)
        best_row = exp_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]
        arch_str = best_row["Architecture"]
        dims = [int(d) for d in arch_str.strip("[]").split("-")]
        hidden_1, hidden_2 = dims[0], dims[1]

    model = TabularChurnNN(input_dim=input_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=config.DEVICE))
    model.eval()

    def get_test_roc_auc(X_matrix):
        with torch.no_grad():
            tensor_input = torch.tensor(X_matrix, dtype=torch.float32, device=config.DEVICE)
            probs_2d = model.predict_proba(tensor_input).cpu().numpy()
            probs_pos = probs_2d[:, 1]
        return roc_auc_score(y_test, probs_pos)

    # 3. Calculate Baseline Score
    baseline_auc = get_test_roc_auc(X_test_np)
    print(f"\n[1] BASELINE TEST PERFORMANCE:")
    print(f"    - Baseline Test ROC-AUC: {baseline_auc:.4f}")
    print(f"    - Number of Permutation Repeats: {n_repeats}")

    # 4. Permutation Loop
    print(f"\n[2] COMPUTING PERMUTATION DROPS FOR {len(feature_names)} FEATURES...")
    np.random.seed(config.RANDOM_SEED)

    mean_drops = []
    std_drops = []

    for j, col_name in enumerate(feature_names):
        repeat_drops = []
        for rep in range(n_repeats):
            X_permuted = X_test_np.copy()
            # Randomly shuffle column j
            shuffled_col = np.random.permutation(X_permuted[:, j])
            X_permuted[:, j] = shuffled_col

            perm_auc = get_test_roc_auc(X_permuted)
            auc_drop = baseline_auc - perm_auc
            repeat_drops.append(auc_drop)

        mean_drop = float(np.mean(repeat_drops))
        std_drop = float(np.std(repeat_drops))
        mean_drops.append(mean_drop)
        std_drops.append(std_drop)

    perm_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "ROC_AUC_Drop_Mean": mean_drops,
            "ROC_AUC_Drop_Std": std_drops,
        }
    ).sort_values(by="ROC_AUC_Drop_Mean", ascending=False).reset_index(drop=True)

    print(f"\n[3] TOP 10 PERMUTATION FEATURE IMPORTANCE (ROC-AUC DROP):")
    print(perm_df.head(10).to_string(index=False))

    # Save CSV
    perm_csv_path = config.REPORTS_DIR / "permutation_importance.csv"
    perm_df.to_csv(perm_csv_path, index=False)
    print(f"\n[SAVED] Permutation Importance Table -> {perm_csv_path}")

    # 5. Save Visualization
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    top15_perm = perm_df.head(15).sort_values(by="ROC_AUC_Drop_Mean", ascending=True)

    plt.barh(
        top15_perm["Feature"],
        top15_perm["ROC_AUC_Drop_Mean"],
        xerr=top15_perm["ROC_AUC_Drop_Std"],
        color="#2b5c8f",
        capsize=3,
        ecolor="#555555",
    )
    plt.title(
        f"Permutation Feature Importance (ROC-AUC Performance Degradation)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel(f"Mean ROC-AUC Score Drop (Baseline = {baseline_auc:.4f})", fontsize=9)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()

    perm_plot_path = figures_dir / "15_permutation_importance.png"
    plt.savefig(perm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 15 -> {perm_plot_path}\n")

    print("=" * 65 + "\n")
    return perm_df


if __name__ == "__main__":
    run_permutation_importance_analysis()
