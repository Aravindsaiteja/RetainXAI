"""
Local Explanation Analysis Module.
Selects 5 representative test samples (True Positive, True Negative, High Confidence, Low Confidence, Misclassified)
and performs deep multi-method local feature attribution analysis (SHAP, LIME, Integrated Gradients).
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import shap
import lime
import lime.lime_tabular
from captum.attr import IntegratedGradients

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def run_local_case_analysis():
    """
    Executes local multi-case explanation analysis:
    1. Loads test data and PyTorch neural network checkpoint.
    2. Filters 5 distinct representative cases.
    3. Calculates SHAP, LIME, and Integrated Gradients scores for each case.
    4. Saves case analysis report to CSV and JSON.
    """
    print("\n" + "=" * 65)
    print("        LOCAL EXPLANATION ANALYSIS (5 REPRESENTATIVE CASES)")
    print("=" * 65)

    # 1. Load Data & Model
    df_train = pd.read_csv(config.PROCESSED_TRAIN_FILE)
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)

    X_train_df = df_train.drop(columns=[config.TARGET_COLUMN])
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

    # Predict Probabilities
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32, device=config.DEVICE)
        probs_2d = model.predict_proba(X_test_tensor).cpu().numpy()
        probs_pos = probs_2d[:, 1]
        preds = (probs_pos >= 0.5).astype(int)

    # 2. Automatically Find 5 Representative Cases
    tp_indices = np.where((y_test == 1) & (preds == 1))[0]
    tn_indices = np.where((y_test == 0) & (preds == 0))[0]
    high_conf_indices = np.where(probs_pos >= 0.90)[0]
    low_conf_indices = np.where((probs_pos >= 0.45) & (probs_pos <= 0.55))[0]
    misclass_indices = np.where(y_test != preds)[0]

    case1_idx = int(tp_indices[0]) if len(tp_indices) > 0 else 0
    case2_idx = int(tn_indices[0]) if len(tn_indices) > 0 else 1
    case3_idx = int(high_conf_indices[0]) if len(high_conf_indices) > 0 else int(np.argmax(probs_pos))
    case4_idx = int(low_conf_indices[0]) if len(low_conf_indices) > 0 else int(np.argmin(np.abs(probs_pos - 0.5)))
    case5_idx = int(misclass_indices[0]) if len(misclass_indices) > 0 else 2

    cases = [
        {"Case": "Case 1: True Positive (Correct Churn)", "idx": case1_idx},
        {"Case": "Case 2: True Negative (Correct Retained)", "idx": case2_idx},
        {"Case": "Case 3: High Confidence Prediction", "idx": case3_idx},
        {"Case": "Case 4: Low Confidence (Uncertain Boundary)", "idx": case4_idx},
        {"Case": "Case 5: Misclassified Instance (Error)", "idx": case5_idx},
    ]

    # 3. Setup Attribution Engines
    # SHAP
    def model_predict_proba_shap(x_numpy):
        if x_numpy.ndim == 1:
            x_numpy = x_numpy.reshape(1, -1)
        x_t = torch.tensor(x_numpy, dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            return model.predict_proba(x_t).cpu().numpy()[:, 1]

    bg_summary = shap.kmeans(X_train_df.values, 50) if len(X_train_df) > 100 else X_train_df.values[:50]
    shap_explainer = shap.KernelExplainer(model_predict_proba_shap, bg_summary)

    # LIME
    def model_predict_proba_lime(x_numpy):
        if x_numpy.ndim == 1:
            x_numpy = x_numpy.reshape(1, -1)
        x_t = torch.tensor(x_numpy, dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            return model.predict_proba(x_t).cpu().numpy()

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=feature_names,
        class_names=["Retained", "Churn"],
        mode="classification",
        random_state=config.RANDOM_SEED,
    )

    # Captum IG
    ig = IntegratedGradients(model)

    # 4. Analyze Each Case
    case_results = []
    print("\n" + "-" * 65)

    for case_info in cases:
        c_title = case_info["Case"]
        idx = case_info["idx"]

        sample_np = X_test_np[idx : idx + 1]
        actual_cls = int(y_test[idx])
        prob_churn = float(probs_pos[idx])
        pred_cls = int(preds[idx])

        print(f"\nAnalyzing [{c_title}] (Test Index #{idx}):")
        print(f"  - Actual: {actual_cls} ({'Churn' if actual_cls==1 else 'Retained'}) | Pred: {pred_cls} ({'Churn' if pred_cls==1 else 'Retained'}) | P(Churn): {prob_churn*100:.2f}%")

        # SHAP
        shap_vals = shap_explainer.shap_values(sample_np, l1_reg="num_features(5)")[0]
        top_shap_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
        top_shap_str = ", ".join([f"{feature_names[i]}: {shap_vals[i]:+.4f}" for i in top_shap_idx])

        # LIME
        lime_exp = lime_explainer.explain_instance(sample_np[0], model_predict_proba_lime, num_features=3, labels=(1,))
        lime_list = lime_exp.as_list(label=1)
        top_lime_str = ", ".join([f"{cond}: {w:+.4f}" for cond, w in lime_list])

        # Captum IG
        sample_tensor = torch.tensor(sample_np, dtype=torch.float32, device=config.DEVICE)
        base_tensor = torch.zeros_like(sample_tensor)
        ig_attr = ig.attribute(sample_tensor, baselines=base_tensor, n_steps=50).squeeze().detach().cpu().numpy()
        top_ig_idx = np.argsort(np.abs(ig_attr))[::-1][:3]
        top_ig_str = ", ".join([f"{feature_names[i]}: {ig_attr[i]:+.4f}" for i in top_ig_idx])

        print(f"  - Top 3 SHAP Drivers: {top_shap_str}")
        print(f"  - Top 3 LIME Drivers: {top_lime_str}")
        print(f"  - Top 3 Captum IG Drivers: {top_ig_str}")

        case_results.append(
            {
                "Case_Title": c_title,
                "Test_Index": idx,
                "Actual_Class": actual_cls,
                "Predicted_Class": pred_cls,
                "Churn_Probability": round(prob_churn, 4),
                "Top_SHAP_Drivers": top_shap_str,
                "Top_LIME_Drivers": top_lime_str,
                "Top_IG_Drivers": top_ig_str,
            }
        )

    # 5. Save Artifacts
    df_results = pd.DataFrame(case_results)
    csv_path = config.REPORTS_DIR / "local_explanation_analysis.csv"
    df_results.to_csv(csv_path, index=False)

    json_path = config.REPORTS_DIR / "local_explanation_cases.json"
    with open(json_path, "w") as f:
        json.dump(case_results, f, indent=4)

    print("\n" + "=" * 65)
    print(f"[SAVED] Local Case CSV Report  -> {csv_path}")
    print(f"[SAVED] Local Case JSON Report -> {json_path}")
    print("=" * 65 + "\n")

    return df_results


if __name__ == "__main__":
    run_local_case_analysis()
