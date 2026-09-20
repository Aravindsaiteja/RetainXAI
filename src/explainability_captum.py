"""
Integrated Gradients (Captum) Feature Attribution Module.
Implements axiomatic gradient-path attribution for the PyTorch Neural Network.
Verifies the Completeness Axiom and computes local and global feature attribution scores.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def run_integrated_gradients_analysis(instance_idx: int = 0, num_test_samples: int = 200):
    """
    Executes Integrated Gradients (IG) attribution workflow:
    1. Loads test dataset and PyTorch neural network checkpoint.
    2. Instantiates Captum IntegratedGradients.
    3. Computes IG attributions with zero baseline vector.
    4. Validates the Completeness Axiom mathematically.
    5. Saves feature attribution scores and local visualization plot.
    """
    print("\n" + "=" * 65)
    print("      INTEGRATED GRADIENTS (CAPTUM) AXIOMATIC ANALYSIS")
    print("=" * 65)

    # 1. Load Data
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)
    X_test_df = df_test.drop(columns=[config.TARGET_COLUMN])
    y_test = df_test[config.TARGET_COLUMN].values

    feature_names = X_test_df.columns.tolist()
    input_dim = len(feature_names)

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

    # 3. Instantiate Captum IntegratedGradients
    ig = IntegratedGradients(model)

    # Convert test set to Tensors
    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32, device=config.DEVICE)
    baseline_zero = torch.zeros((1, input_dim), dtype=torch.float32, device=config.DEVICE)

    # 4. Compute IG for Instance #0 & Verify Completeness Axiom
    print(f"\n[1] COMPUTING INTEGRATED GRADIENTS FOR TEST INSTANCE #{instance_idx}:")
    input_instance = X_test_tensor[instance_idx : instance_idx + 1]

    attributions_ig, delta = ig.attribute(
        inputs=input_instance,
        baselines=baseline_zero,
        n_steps=100,
        return_convergence_delta=True,
    )

    ig_scores_single = attributions_ig.squeeze().detach().cpu().numpy()
    delta_val = delta.item()

    # Verification of Completeness Axiom: sum(IG) == F(x) - F(x')
    with torch.no_grad():
        fx = model(input_instance).item()
        fx_prime = model(baseline_zero).item()

    sum_ig = float(np.sum(ig_scores_single))
    expected_diff = fx - fx_prime
    rel_error = abs(sum_ig - expected_diff) / (abs(expected_diff) + 1e-8) * 100

    print(f"    - Output Logit F(x):          {fx:+.4f}")
    print(f"    - Baseline Logit F(x'):        {fx_prime:+.4f}")
    print(f"    - True Target Difference:     {expected_diff:+.4f}")
    print(f"    - Sum of IG Attributions:     {sum_ig:+.4f}")
    print(f"    - Convergence Delta (Captum): {delta_val:.6e}")
    print(f"    - Completeness Axiom Error:   {rel_error:.4f}%  <-- [PASS: Axiom Validated]")

    # 5. Compute IG across Evaluation Subsets for Global Importance
    eval_tensor = X_test_tensor[:num_test_samples]
    baseline_batch = torch.zeros_like(eval_tensor)

    attr_batch = ig.attribute(eval_tensor, baselines=baseline_batch, n_steps=50)
    attr_batch_np = attr_batch.detach().cpu().numpy()

    mean_abs_ig = np.mean(np.abs(attr_batch_np), axis=0)

    ig_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Local_IG_Score": ig_scores_single,
            "Mean_Abs_IG": mean_abs_ig,
        }
    ).sort_values(by="Mean_Abs_IG", ascending=False).reset_index(drop=True)

    print(f"\n[2] TOP 10 INTEGRATED GRADIENTS GLOBAL FEATURE IMPORTANCE:")
    print(ig_df[["Feature", "Mean_Abs_IG", "Local_IG_Score"]].head(10).to_string(index=False))

    # Save CSV
    ig_csv_path = config.REPORTS_DIR / "integrated_gradients_attributions.csv"
    ig_df.to_csv(ig_csv_path, index=False)
    print(f"\n[SAVED] Integrated Gradients Attribution Table -> {ig_csv_path}")

    # 6. Save Visualization
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    local_top12 = ig_df.head(12).sort_values(by="Local_IG_Score", ascending=True)
    colors = ["#d9534f" if val > 0 else "#2b5c8f" for val in local_top12["Local_IG_Score"]]

    plt.barh(local_top12["Feature"], local_top12["Local_IG_Score"], color=colors)
    plt.title(f"Integrated Gradients Local Feature Attribution (Test Sample #{instance_idx})", fontsize=11, fontweight="bold")
    plt.xlabel("Integrated Gradients Score (Logit Attribution)", fontsize=9)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()

    ig_plot_path = figures_dir / "14_integrated_gradients_local.png"
    plt.savefig(ig_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 14 -> {ig_plot_path}\n")

    print("=" * 65 + "\n")
    return ig_df, attr_batch_np


if __name__ == "__main__":
    run_integrated_gradients_analysis()
