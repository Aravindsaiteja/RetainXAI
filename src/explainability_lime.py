"""
LIME (Local Interpretable Model-agnostic Explanations) Feature Attribution Module.
Fits local linear surrogate models around individual test samples to explain neural network predictions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import lime
import lime.lime_tabular

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def run_lime_analysis(instance_idx: int = 0):
    """
    Executes LIME local surrogate attribution analysis:
    1. Loads processed train/test data and PyTorch neural network checkpoint.
    2. Instantiates LimeTabularExplainer.
    3. Explains prediction for specified test sample.
    4. Computes feature contribution directions and magnitudes.
    5. Saves visualization and feature weights CSV.
    """
    print("\n" + "=" * 65)
    print("      LIME (LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS)")
    print("=" * 65)

    # 1. Load Data
    df_train = pd.read_csv(config.PROCESSED_TRAIN_FILE)
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)

    X_train = df_train.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test.drop(columns=[config.TARGET_COLUMN])
    y_test = df_test[config.TARGET_COLUMN].values

    feature_names = X_test.columns.tolist()
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

    model = TabularChurnNN(input_dim=input_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2)
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=torch.device("cpu")))
    model.eval()

    # Model probability prediction wrapper returning 2D numpy array [P(No Churn), P(Churn)]
    def predict_fn(x_numpy):
        if x_numpy.ndim == 1:
            x_numpy = x_numpy.reshape(1, -1)
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            probs = model.predict_proba(x_tensor).numpy()
        return probs

    # 3. Build LIME Tabular Explainer
    np.random.seed(config.RANDOM_SEED)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["Retained", "Churn"],
        mode="classification",
        random_state=config.RANDOM_SEED,
    )

    print(f"\n[1] GENERATING LIME LOCAL EXPLANATION FOR TEST INSTANCE #{instance_idx}:")
    test_sample = X_test.iloc[instance_idx].values
    actual_label = "Churn" if y_test[instance_idx] == 1 else "Retained"

    probs = predict_fn(test_sample)
    pred_prob_churn = probs[0, 1]
    pred_label = "Churn" if pred_prob_churn >= 0.5 else "Retained"

    print(f"    - Actual Target Class:     '{actual_label}'")
    print(f"    - Predicted Model Class:   '{pred_label}' (P(Churn) = {pred_prob_churn * 100:.2f}%)")

    # Generate Local Explanation
    exp = explainer.explain_instance(
        data_row=test_sample,
        predict_fn=predict_fn,
        num_features=10,
        labels=(1,),
    )

    # Extract LIME Feature Weights
    lime_list = exp.as_list(label=1)
    lime_df = pd.DataFrame(lime_list, columns=["Feature_Condition", "LIME_Weight"])
    lime_df["Impact_Direction"] = lime_df["LIME_Weight"].apply(
        lambda w: "Increases Churn" if w > 0 else "Decreases Churn"
    )
    lime_df["Abs_Weight"] = lime_df["LIME_Weight"].abs()
    lime_df = lime_df.sort_values(by="Abs_Weight", ascending=False).reset_index(drop=True)

    print(f"\n[2] LIME TOP 10 LOCAL FEATURE ATTRIBUTIONS:")
    print(lime_df[["Feature_Condition", "LIME_Weight", "Impact_Direction"]].to_string(index=False))

    # Save CSV
    lime_csv_path = config.REPORTS_DIR / "lime_feature_weights.csv"
    lime_df.to_csv(lime_csv_path, index=False)
    print(f"\n[SAVED] LIME Feature Weights Table -> {lime_csv_path}")

    # 4. Save Visualization
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    plot_df = lime_df.sort_values(by="LIME_Weight", ascending=True)
    colors = ["#d9534f" if w > 0 else "#2b5c8f" for w in plot_df["LIME_Weight"]]
    
    plt.barh(plot_df["Feature_Condition"], plot_df["LIME_Weight"], color=colors)
    plt.title(
        f"LIME Local Feature Attribution (Test Sample #{instance_idx} | P(Churn) = {pred_prob_churn:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("LIME Weight (Local Linear Coefficient)", fontsize=9)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()

    lime_plot_path = figures_dir / "13_lime_local_explanation.png"
    plt.savefig(lime_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 13 -> {lime_plot_path}\n")

    print("=" * 65 + "\n")
    return lime_df, exp


if __name__ == "__main__":
    run_lime_analysis()
