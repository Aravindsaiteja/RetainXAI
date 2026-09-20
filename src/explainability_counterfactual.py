"""
Counterfactual ("What-If") Analysis Module.
Explores minimal input feature perturbations that flip high-risk churn predictions
to low-risk retained predictions without claiming real-world causality.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def run_counterfactual_analysis(instance_idx: int = 10):
    """
    Executes counterfactual "what-if" scenario perturbations:
    1. Selects a high-churn prediction instance.
    2. Applies targeted counterfactual feature modifications.
    3. Re-evaluates model prediction probabilities.
    4. Distinguishes model feature attribution from causality.
    5. Saves counterfactual CSV/JSON reports and plot.
    """
    print("\n" + "=" * 65)
    print("      COUNTERFACTUAL ('WHAT-IF') PREDICTION ANALYSIS")
    print("=" * 65)

    # 1. Load Data & Model
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)
    X_test_df = df_test.drop(columns=[config.TARGET_COLUMN])
    y_test = df_test[config.TARGET_COLUMN].values

    feature_names = X_test_df.columns.tolist()
    input_dim = len(feature_names)
    X_test_np = X_test_df.values

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

    def get_prob(x_vector):
        x_t = torch.tensor(x_vector.reshape(1, -1), dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            return float(model.predict_proba(x_t).cpu().numpy()[0, 1])

    # 2. Select High-Risk Target Instance
    orig_sample = X_test_np[instance_idx].copy()
    orig_prob = get_prob(orig_sample)

    print(f"\n[1] BASELINE HIGH-RISK TARGET INSTANCE (Test Index #{instance_idx}):")
    print(f"    - Actual Class:     {'Churn' if y_test[instance_idx]==1 else 'Retained'}")
    print(f"    - Initial P(Churn): {orig_prob * 100:.2f}%  <-- HIGH CHURN RISK FLAG")

    # Locate column indices
    col_tenure = feature_names.index("tenure")
    col_m2m = feature_names.index("Contract_Month-to-month") if "Contract_Month-to-month" in feature_names else -1
    col_2yr = feature_names.index("Contract_Two year") if "Contract_Two year" in feature_names else -1
    col_sec_no = feature_names.index("OnlineSecurity_No") if "OnlineSecurity_No" in feature_names else -1
    col_sec_yes = feature_names.index("OnlineSecurity_Yes") if "OnlineSecurity_Yes" in feature_names else -1

    scenarios = []

    # Baseline Scenario 0
    scenarios.append(
        {
            "Scenario_ID": "S0: Original Instance",
            "Intervention_Description": "No changes (Baseline High Risk)",
            "Modified_P_Churn": round(orig_prob, 4),
            "Risk_Reduction_Delta": 0.0,
            "Prediction_Class": "Churn" if orig_prob >= 0.5 else "Retained",
        }
    )

    # Scenario 1: Increase Tenure by +12 Months (+0.5 std scaled)
    s1_sample = orig_sample.copy()
    s1_sample[col_tenure] += 0.5
    p1 = get_prob(s1_sample)
    scenarios.append(
        {
            "Scenario_ID": "S1: Increase Tenure (+12 Mos)",
            "Intervention_Description": "Increase customer tenure by 12 months",
            "Modified_P_Churn": round(p1, 4),
            "Risk_Reduction_Delta": round(orig_prob - p1, 4),
            "Prediction_Class": "Churn" if p1 >= 0.5 else "Retained",
        }
    )

    # Scenario 2: Switch Contract from Month-to-Month to 2-Year
    s2_sample = orig_sample.copy()
    if col_m2m != -1:
        s2_sample[col_m2m] = 0.0
    if col_2yr != -1:
        s2_sample[col_2yr] = 1.0
    p2 = get_prob(s2_sample)
    scenarios.append(
        {
            "Scenario_ID": "S2: Switch to 2-Year Contract",
            "Intervention_Description": "Convert subscriber contract to 2-Year fixed plan",
            "Modified_P_Churn": round(p2, 4),
            "Risk_Reduction_Delta": round(orig_prob - p2, 4),
            "Prediction_Class": "Churn" if p2 >= 0.5 else "Retained",
        }
    )

    # Scenario 3: Add Online Security Add-On
    s3_sample = orig_sample.copy()
    if col_sec_no != -1:
        s3_sample[col_sec_no] = 0.0
    if col_sec_yes != -1:
        s3_sample[col_sec_yes] = 1.0
    p3 = get_prob(s3_sample)
    scenarios.append(
        {
            "Scenario_ID": "S3: Add Online Security",
            "Intervention_Description": "Enable complementary online security protection",
            "Modified_P_Churn": round(p3, 4),
            "Risk_Reduction_Delta": round(orig_prob - p3, 4),
            "Prediction_Class": "Churn" if p3 >= 0.5 else "Retained",
        }
    )

    # Scenario 4: Combined Intervention (Tenure + 2-Year Contract + Security)
    s4_sample = orig_sample.copy()
    s4_sample[col_tenure] += 0.5
    if col_m2m != -1:
        s4_sample[col_m2m] = 0.0
    if col_2yr != -1:
        s4_sample[col_2yr] = 1.0
    if col_sec_no != -1:
        s4_sample[col_sec_no] = 0.0
    if col_sec_yes != -1:
        s4_sample[col_sec_yes] = 1.0
    p4 = get_prob(s4_sample)
    scenarios.append(
        {
            "Scenario_ID": "S4: Combined Intervention",
            "Intervention_Description": "Tenure + 2-Yr Contract + Online Security",
            "Modified_P_Churn": round(p4, 4),
            "Risk_Reduction_Delta": round(orig_prob - p4, 4),
            "Prediction_Class": "Churn" if p4 >= 0.5 else "Retained",
        }
    )

    print(f"\n[2] COUNTERFACTUAL WHAT-IF INTERVENTION RESULTS:")
    df_scenarios = pd.DataFrame(scenarios)
    print(df_scenarios.to_string(index=False))

    # 3. Methodological Warning
    print(f"\n[3] IMPORTANT METHODOLOGICAL DISTINCTION:")
    print("    * Feature Attribution != Causal Explanation.")
    print("    * Feature attributions describe model scoring behavior on training correlations.")
    print("    * Counterfactual analysis explores hypothetical decision boundary shifts.")
    print("    * Counterfactual interventions guide business policy experiments, but do not guarantee real-world causality.")

    # 4. Save Artifacts
    csv_path = config.REPORTS_DIR / "counterfactual_analysis.csv"
    df_scenarios.to_csv(csv_path, index=False)

    json_path = config.REPORTS_DIR / "counterfactual_scenarios.json"
    with open(json_path, "w") as f:
        json.dump(scenarios, f, indent=4)

    print(f"\n[SAVED] Counterfactual CSV Report  -> {csv_path}")
    print(f"[SAVED] Counterfactual JSON Report -> {json_path}")

    # 5. Save Visualization
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    plot_df = df_scenarios.iloc[::-1]
    colors = ["#d9534f" if p >= 0.5 else "#27ae60" for p in plot_df["Modified_P_Churn"]]

    plt.barh(plot_df["Scenario_ID"], plot_df["Modified_P_Churn"] * 100, color=colors)
    plt.axvline(50, color="black", linestyle="--", linewidth=1, label="50% Decision Threshold")
    plt.title(f"Counterfactual 'What-If' Interventions (Test Instance #{instance_idx})", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Predicted Churn Probability (%)", fontsize=9)
    plt.xlim(0, 100)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    cf_plot_path = figures_dir / "21_counterfactual_scenarios.png"
    plt.savefig(cf_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 21 -> {cf_plot_path}\n")

    print("=" * 65 + "\n")
    return df_scenarios


if __name__ == "__main__":
    run_counterfactual_analysis()
