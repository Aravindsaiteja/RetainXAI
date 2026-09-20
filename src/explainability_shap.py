"""
SHAP (Shapley Additive exPlanations) Feature Attribution Module.
Computes game-theoretic Shapley values for the trained PyTorch neural network model.
Generates global feature importance rankings, beeswarm plots, bar plots, and local instance waterfall explanations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def run_shap_analysis(num_background_samples: int = 100, num_test_samples: int = 200):
    """
    Executes SHAP feature attribution workflow:
    1. Loads test data and trained PyTorch model.
    2. Builds model probability wrapper for SHAP.
    3. Computes SHAP values using KernelExplainer.
    4. Calculates global feature importance ranking.
    5. Saves visualizations and CSV outputs.
    """
    print("\n" + "=" * 65)
    print("         SHAP (SHAPLEY ADDITIVE EXPLANATIONS) ANALYSIS")
    print("=" * 65)

    # 1. Load Data
    df_train = pd.read_csv(config.PROCESSED_TRAIN_FILE)
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)

    X_train = df_train.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test.drop(columns=[config.TARGET_COLUMN])
    y_test = df_test[config.TARGET_COLUMN].values

    feature_names = X_test.columns.tolist()
    input_dim = len(feature_names)

    # 2. Load Model
    exp_file = config.EXPERIMENT_LOG_FILE
    hidden_1, hidden_2 = 128, 64
    if exp_file.exists():
        exp_df = pd.read_csv(exp_file)
        best_row = exp_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]
        arch_str = best_row["Architecture"]
        dims = [int(d) for d in arch_str.strip("[]").split("-")]
        hidden_1, hidden_2 = dims[0], dims[1]

    model = TabularChurnNN(input_dim=input_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2)
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=torch.device("cpu")))
    model.eval()

    # Model probability prediction wrapper returning float numpy array P(Churn)
    def model_predict_proba(x_numpy):
        if x_numpy.ndim == 1:
            x_numpy = x_numpy.reshape(1, -1)
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            probs = model.predict_proba(x_tensor).numpy()
        return probs[:, 1]  # Return P(Churn=1)

    # 3. Prepare Background and Test Subsets
    background_data = shap.kmeans(X_train.values, 50) if len(X_train) > 100 else X_train.values[:num_background_samples]
    eval_data = X_test.values[:num_test_samples]

    print(f"\n[1] INITIALIZING SHAP KERNEL EXPLAINER:")
    print(f"    - Background K-Means Clusters: 50")
    print(f"    - Evaluation Test Subsets:     {len(eval_data)} instances")

    explainer = shap.KernelExplainer(model_predict_proba, background_data)
    shap_values = explainer.shap_values(eval_data, l1_reg="num_features(10)")

    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]  # Positive class if list
    else:
        shap_values_pos = shap_values  # Shape: (num_test_samples, num_features)

    # 4. Calculate Global Feature Importance (Mean |SHAP|)
    mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)
    shap_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean_Abs_SHAP": mean_abs_shap,
        }
    ).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print(f"\n[2] TOP 10 GLOBAL SHAP FEATURE IMPORTANCE:")
    print(shap_importance_df.head(10).to_string(index=False))

    # Save Importance CSV
    shap_csv_path = config.REPORTS_DIR / "shap_global_importance.csv"
    shap_importance_df.to_csv(shap_csv_path, index=False)
    print(f"\n[SAVED] SHAP Global Importance Table -> {shap_csv_path}")

    # 5. Generate Visualizations
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot 10: SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_pos,
        eval_data,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Feature Attribution Summary (Beeswarm Plot)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = figures_dir / "10_shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 10 -> {beeswarm_path}")

    # Plot 11: SHAP Global Bar Plot
    plt.figure(figsize=(8, 6))
    top15_shap = shap_importance_df.head(15).sort_values(by="Mean_Abs_SHAP", ascending=True)
    plt.barh(top15_shap["Feature"], top15_shap["Mean_Abs_SHAP"], color="#2b5c8f")
    plt.title("Top 15 Global Feature Importance (Mean |SHAP Value|)", fontsize=11, fontweight="bold")
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output Magnitude)", fontsize=9)
    plt.tight_layout()
    bar_path = figures_dir / "11_shap_global_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 11 -> {bar_path}")

    # Plot 12: SHAP Local Explanation (Waterfall Plot for Test Instance #0)
    instance_idx = 0
    sample_feat_values = eval_data[instance_idx]
    sample_shap_vals = shap_values_pos[instance_idx]
    base_val = explainer.expected_value
    if isinstance(base_val, list) or isinstance(base_val, np.ndarray):
        base_val = float(base_val[0])

    explanation_obj = shap.Explanation(
        values=sample_shap_vals,
        base_values=base_val,
        data=sample_feat_values,
        feature_names=feature_names,
    )

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation_obj, max_display=10, show=False)
    plt.title(f"Local SHAP Explanation (Waterfall Plot for Test Sample #{instance_idx})", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall_path = figures_dir / "12_shap_local_waterfall.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 12 -> {waterfall_path}\n")

    print("=" * 65 + "\n")
    return shap_importance_df, shap_values_pos, explainer


if __name__ == "__main__":
    run_shap_analysis()
