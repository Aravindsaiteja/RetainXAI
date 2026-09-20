"""
Explanation Faithfulness & Perturbation Analysis Module.
Evaluates whether top-k features identified by SHAP, LIME, and Integrated Gradients truly drive
neural network decisions via feature-removal (masking) experiments.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from captum.attr import IntegratedGradients

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def run_explanation_faithfulness_analysis(num_samples: int = 150):
    """
    Executes top-k feature removal perturbation experiments:
    1. Loads test dataset and PyTorch neural network checkpoint.
    2. Computes attributions for top-k features (k = 1, 3, 5, 10).
    3. Replaces top-k features with zero/median baseline.
    4. Measures prediction probability drop: Delta P = P(Churn|x) - P(Churn|x_masked).
    5. Saves faithfulness results table and degradation plot.
    """
    print("\n" + "=" * 65)
    print("      EXPLANATION FAITHFULNESS & PERTURBATION AUDIT")
    print("=" * 65)

    # 1. Load Data & Model
    df_train = pd.read_csv(config.PROCESSED_TRAIN_FILE)
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)

    X_train_df = df_train.drop(columns=[config.TARGET_COLUMN])
    X_test_df = df_test.drop(columns=[config.TARGET_COLUMN])
    feature_names = X_test_df.columns.tolist()
    input_dim = len(feature_names)
    X_test_np = X_test_df.values[:num_samples]

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

    # Original Probabilities
    with torch.no_grad():
        t_orig = torch.tensor(X_test_np, dtype=torch.float32, device=config.DEVICE)
        orig_probs = model.predict_proba(t_orig).cpu().numpy()[:, 1]

    # 2. Setup Attribution Engines
    ig = IntegratedGradients(model)

    def model_predict_proba_shap(x_numpy):
        if x_numpy.ndim == 1:
            x_numpy = x_numpy.reshape(1, -1)
        x_t = torch.tensor(x_numpy, dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            return model.predict_proba(x_t).cpu().numpy()[:, 1]

    bg_summary = shap.kmeans(X_train_df.values, 30) if len(X_train_df) > 50 else X_train_df.values[:30]
    shap_explainer = shap.KernelExplainer(model_predict_proba_shap, bg_summary)

    print(f"\n[1] COMPUTING ATTRIBUTIONS FOR {num_samples} TEST INSTANCES...")

    # Integrated Gradients attributions batch
    t_batch = torch.tensor(X_test_np, dtype=torch.float32, device=config.DEVICE)
    base_batch = torch.zeros_like(t_batch)
    ig_attrs = ig.attribute(t_batch, baselines=base_batch, n_steps=30).detach().cpu().numpy()

    # SHAP attributions batch
    shap_attrs = shap_explainer.shap_values(X_test_np, l1_reg="num_features(10)")
    if isinstance(shap_attrs, list):
        shap_attrs = shap_attrs[1]

    k_list = [1, 3, 5, 10]
    results = []

    print(f"\n[2] RUNNING FEATURE MASKING PERTURBATION EXPERIMENTS:")

    for k in k_list:
        drop_ig = []
        drop_shap = []
        drop_random = []

        for i in range(num_samples):
            p_orig = orig_probs[i]
            x_inst = X_test_np[i].copy()

            # Mask Top-k IG
            top_k_ig_indices = np.argsort(np.abs(ig_attrs[i]))[::-1][:k]
            x_masked_ig = x_inst.copy()
            x_masked_ig[top_k_ig_indices] = 0.0
            t_m_ig = torch.tensor(x_masked_ig.reshape(1, -1), dtype=torch.float32, device=config.DEVICE)
            with torch.no_grad():
                p_masked_ig = float(model.predict_proba(t_m_ig).cpu().numpy()[0, 1])
            drop_ig.append(p_orig - p_masked_ig)

            # Mask Top-k SHAP
            top_k_shap_indices = np.argsort(np.abs(shap_attrs[i]))[::-1][:k]
            x_masked_shap = x_inst.copy()
            x_masked_shap[top_k_shap_indices] = 0.0
            t_m_shap = torch.tensor(x_masked_shap.reshape(1, -1), dtype=torch.float32, device=config.DEVICE)
            with torch.no_grad():
                p_masked_shap = float(model.predict_proba(t_m_shap).cpu().numpy()[0, 1])
            drop_shap.append(p_orig - p_masked_shap)

            # Mask Random k
            rand_indices = np.random.choice(input_dim, size=k, replace=False)
            x_masked_rand = x_inst.copy()
            x_masked_rand[rand_indices] = 0.0
            t_m_rand = torch.tensor(x_masked_rand.reshape(1, -1), dtype=torch.float32, device=config.DEVICE)
            with torch.no_grad():
                p_masked_rand = float(model.predict_proba(t_m_rand).cpu().numpy()[0, 1])
            drop_random.append(p_orig - p_masked_rand)

        mean_drop_ig = float(np.mean(drop_ig))
        mean_drop_shap = float(np.mean(drop_shap))
        mean_drop_rand = float(np.mean(drop_random))

        print(f"    - Top-{k:<2} Masking | IG Drop: {mean_drop_ig:+.4f} | SHAP Drop: {mean_drop_shap:+.4f} | Random Drop: {mean_drop_rand:+.4f}")

        results.append(
            {
                "Top_k_Masked": k,
                "IG_Mean_Prob_Drop": round(mean_drop_ig, 4),
                "SHAP_Mean_Prob_Drop": round(mean_drop_shap, 4),
                "Random_Baseline_Drop": round(mean_drop_rand, 4),
                "Faithfulness_Gain_IG_vs_Random": round(mean_drop_ig - mean_drop_rand, 4),
            }
        )

    # 3. Save Results
    df_faith = pd.DataFrame(results)
    faith_csv_path = config.REPORTS_DIR / "explanation_faithfulness_results.csv"
    df_faith.to_csv(faith_csv_path, index=False)
    print(f"\n[SAVED] Faithfulness Report -> {faith_csv_path}")

    # 4. Save Visualization
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df_faith["Top_k_Masked"], df_faith["IG_Mean_Prob_Drop"], marker="o", color="#27ae60", linewidth=2.5, label="Integrated Gradients (Captum)")
    plt.plot(df_faith["Top_k_Masked"], df_faith["SHAP_Mean_Prob_Drop"], marker="s", color="#2b5c8f", linewidth=2.5, label="SHAP (Kernel)")
    plt.plot(df_faith["Top_k_Masked"], df_faith["Random_Baseline_Drop"], marker="^", color="gray", linestyle="--", linewidth=1.5, label="Random Feature Masking (Baseline)")

    plt.title("Explanation Faithfulness: Prediction Drop Under Top-k Feature Masking", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Top-k Important Features Removed (k)", fontsize=10)
    plt.ylabel("Mean Prediction Probability Drop (Delta P)", fontsize=10)
    plt.xticks(k_list)
    plt.legend(fontsize=9, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    faith_plot_path = figures_dir / "19_explanation_faithfulness_degradation.png"
    plt.savefig(faith_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 19 -> {faith_plot_path}\n")

    print("=" * 65 + "\n")
    return df_faith


if __name__ == "__main__":
    run_explanation_faithfulness_analysis()
