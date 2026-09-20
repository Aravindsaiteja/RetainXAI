"""
Global Dataset-Level Explainability Analysis Module.
Evaluates dataset-wide feature importance distributions, Pareto concentration ratios,
feature dominance, and potential model reliance bias.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def run_global_explainability_analysis():
    """
    Executes dataset-level global explainability analysis:
    1. Loads consensus feature importance rankings from Phase 13.
    2. Computes cumulative importance percentage (Pareto concentration).
    3. Calculates Gini coefficient of feature reliance.
    4. Evaluates feature dominance and bias.
    5. Saves global summary JSON and 4-panel Dashboard figure.
    """
    print("\n" + "=" * 65)
    print("      GLOBAL DATASET-LEVEL EXPLAINABILITY ANALYSIS")
    print("=" * 65)

    comp_path = config.REPORTS_DIR / "feature_attribution_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError("feature_attribution_comparison.csv not found! Run Phase 13 first.")

    df_comp = pd.read_csv(comp_path)

    # 1. Compute Cumulative Importance Percentage
    df_comp = df_comp.sort_values(by="Consensus_Score", ascending=False).reset_index(drop=True)
    total_score = df_comp["Consensus_Score"].sum()
    df_comp["Importance_Pct"] = (df_comp["Consensus_Score"] / total_score) * 100
    df_comp["Cumulative_Pct"] = df_comp["Importance_Pct"].cumsum()

    # 2. Pareto Concentration Metrics
    top3_pct = df_comp.iloc[:3]["Importance_Pct"].sum()
    top5_pct = df_comp.iloc[:5]["Importance_Pct"].sum()
    top10_pct = df_comp.iloc[:10]["Importance_Pct"].sum()
    bottom_half_pct = df_comp.iloc[len(df_comp) // 2 :]["Importance_Pct"].sum()

    # Compute Gini Coefficient of Feature Importance
    scores = np.sort(df_comp["Consensus_Score"].values)
    n = len(scores)
    index = np.arange(1, n + 1)
    gini = float((np.sum((2 * index - n - 1) * scores)) / (n * np.sum(scores)))

    global_summary = {
        "total_features": int(len(df_comp)),
        "top_3_features_concentration_pct": round(float(top3_pct), 2),
        "top_5_features_concentration_pct": round(float(top5_pct), 2),
        "top_10_features_concentration_pct": round(float(top10_pct), 2),
        "bottom_half_concentration_pct": round(float(bottom_half_pct), 2),
        "feature_attribution_gini_coefficient": round(float(gini), 4),
        "top_driver": df_comp.iloc[0]["Feature"],
        "top_5_drivers": df_comp.iloc[:5]["Feature"].tolist(),
        "low_impact_features": df_comp.iloc[-5:]["Feature"].tolist(),
    }

    print(f"\n[1] DATASET-LEVEL FEATURE ATTRIBUTION CONCENTRATION:")
    print(f"    - Total Input Features:             {len(df_comp)}")
    print(f"    - Top 3 Features Concentration:     {top3_pct:.2f}% of total model attribution")
    print(f"    - Top 5 Features Concentration:     {top5_pct:.2f}% of total model attribution")
    print(f"    - Top 10 Features Concentration:    {top10_pct:.2f}% of total model attribution")
    print(f"    - Bottom 50% Features Concentration: {bottom_half_pct:.2f}%")
    print(f"    - Feature Importance Gini Coeff:    {gini:.4f} (High Inequality / Strong Feature Concentration)")

    # Save JSON Summary
    summary_path = config.REPORTS_DIR / "global_explainability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(global_summary, f, indent=4)
    print(f"\n[SAVED] Global Summary JSON -> {summary_path}")

    # 3. Generate 4-Panel Global Dashboard Figure
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Top 10 Feature Consensus Importance
    top10_df = df_comp.head(10).sort_values(by="Consensus_Score", ascending=True)
    axes[0, 0].barh(top10_df["Feature"], top10_df["Consensus_Score"], color="#2b5c8f")
    axes[0, 0].set_title("A. Top 10 Consensus Feature Importance", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Consensus Normalized Score [0 - 1]", fontsize=9)
    axes[0, 0].grid(True, linestyle=":", alpha=0.5)

    # Panel B: Cumulative Attribution (Pareto Curve)
    x_range = np.arange(1, len(df_comp) + 1)
    axes[0, 1].plot(x_range, df_comp["Cumulative_Pct"], color="#d9534f", linewidth=2.5, marker="o", markersize=3)
    axes[0, 1].axhline(80, color="gray", linestyle="--", linewidth=1, label="80% Pareto Cutoff")
    axes[0, 1].axvline(10, color="blue", linestyle=":", linewidth=1, label="Top 10 Features")
    axes[0, 1].set_title("B. Cumulative Feature Attribution Pareto Curve", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Feature Rank", fontsize=9)
    axes[0, 1].set_ylabel("Cumulative Attribution (%)", fontsize=9)
    axes[0, 1].legend(fontsize=9, loc="lower right")
    axes[0, 1].grid(True, linestyle=":", alpha=0.5)

    # Panel C: Top Features Score Breakdown across 4 Methods
    top6_df = df_comp.head(6)
    methods_matrix = top6_df[["SHAP_Norm", "LIME_Norm", "IG_Norm", "Permutation_Norm"]].values
    sns.heatmap(
        methods_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=["SHAP", "LIME", "Captum IG", "Permutation"],
        yticklabels=top6_df["Feature"],
        ax=axes[1, 0],
        cbar=False,
    )
    axes[1, 0].set_title("C. Top 6 Features 4-Method Cross-Heatmap", fontsize=11, fontweight="bold")

    # Panel D: Domain Feature Category Distribution
    category_map = {}
    for feat in df_comp["Feature"]:
        if "tenure" in feat or "Charges" in feat or "spend" in feat:
            category_map[feat] = "Financial & Tenure"
        elif "Contract" in feat:
            category_map[feat] = "Contractual Terms"
        elif "Internet" in feat or "Service" in feat:
            category_map[feat] = "Core Services"
        elif "Security" in feat or "Backup" in feat or "Protection" in feat or "Support" in feat:
            category_map[feat] = "Add-On Features"
        else:
            category_map[feat] = "Demographics & Billing"

    df_comp["Feature_Group"] = df_comp["Feature"].map(category_map)
    group_scores = df_comp.groupby("Feature_Group")["Consensus_Score"].sum().sort_values(ascending=True)

    axes[1, 1].barh(group_scores.index, group_scores.values, color="#27ae60")
    axes[1, 1].set_title("D. Total Attribution by Business Feature Category", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Aggregated Consensus Attribution Weight", fontsize=9)
    axes[1, 1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    dashboard_path = figures_dir / "17_global_explainability_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot 17 -> {dashboard_path}\n")

    print("=" * 65 + "\n")
    return global_summary, df_comp


if __name__ == "__main__":
    run_global_explainability_analysis()
