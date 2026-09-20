"""
Exploratory Data Analysis (EDA) Module.
Performs quantitative statistical analysis and generates targeted publication-quality visualizations
saved in reports/figures/. Answers key analytical questions required for feature attribution contextualization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Visual style setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FIGURES_DIR = config.REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_clean_raw_df() -> pd.DataFrame:
    """Loads raw data, handles whitespace in TotalCharges, and adds numeric Churn_Binary."""
    df = pd.read_csv(config.RAW_DATA_FILE)
    # Replace empty spaces with NaN and convert to float
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True).astype(float)
    df["Churn_Binary"] = (df[config.TARGET_COLUMN] == "Yes").astype(int)
    return df


def generate_eda_report(df: pd.DataFrame) -> dict:
    """
    Executes EDA calculations and prints structured analytical findings.
    """
    print("\n" + "=" * 65)
    print("           EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 65)

    stats = {}

    # 1. Target Distribution Analysis
    target_counts = df[config.TARGET_COLUMN].value_counts()
    stats["target_counts"] = target_counts.to_dict()
    churn_rate = (target_counts.get("Yes", 0) / len(df)) * 100
    stats["churn_rate"] = churn_rate

    print(f"\n[1] TARGET DISTRIBUTION & BASELINE RISK:")
    print(f"    - Retained ('No'): {target_counts.get('No', 0)} ({100 - churn_rate:.2f}%)")
    print(f"    - Churned ('Yes'):  {target_counts.get('Yes', 0)} ({churn_rate:.2f}%)")
    print(f"    - Baseline Churn Rate: {churn_rate:.2f}%")

    # 2. Numerical Summary Statistics
    num_cols = config.NUMERICAL_FEATURES
    print(f"\n[2] NUMERICAL FEATURE SUMMARY STATISTICS:")
    num_summary = df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    num_summary.rename(columns={"50%": "median"}, inplace=True)
    print(num_summary.to_string())

    # Outlier Analysis (IQR method)
    print(f"\n[3] OUTLIER ANALYSIS (IQR METHOD):")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"    - '{col}': Range [{df[col].min():.2f}, {df[col].max():.2f}], IQR Range [{lower_bound:.2f}, {upper_bound:.2f}] -> Outliers: {len(outliers)}")

    # 3. Correlation with Target (Point-Biserial / Numeric Encoding)
    df_temp = df.copy()
    df_temp["Churn_Binary"] = (df_temp[config.TARGET_COLUMN] == "Yes").astype(int)
    correlations = df_temp[num_cols].corrwith(df_temp["Churn_Binary"])
    print(f"\n[4] NUMERICAL CORRELATION WITH CHURN (Point-Biserial):")
    for col, corr in correlations.items():
        print(f"    - {col:<16}: r = {corr:+.4f}")

    # 4. Categorical Feature Churn Risk Analysis
    print(f"\n[5] HIGH-IMPACT CATEGORICAL CHURN DRIVERS:")
    cat_insights = {}
    key_cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]
    for cat in key_cats:
        churn_by_cat = df.groupby(cat)["Churn_Binary"].agg(["count", "mean"])
        churn_by_cat["churn_pct"] = churn_by_cat["mean"] * 100
        churn_by_cat = churn_by_cat.sort_values(by="churn_pct", ascending=False)
        cat_insights[cat] = churn_by_cat
        print(f"\n    Feature: '{cat}'")
        for idx, row in churn_by_cat.iterrows():
            print(f"      * {idx:<22}: {int(row['count']):>5} total | {row['churn_pct']:>6.2f}% Churn")

    print("\n" + "=" * 65 + "\n")
    return stats


def generate_eda_plots(df: pd.DataFrame):
    """
    Generates 5 specific, high-impact analytical visualizations and saves them to reports/figures/.
    """
    df_plot = df.copy()
    df_plot["Churn_Binary"] = (df_plot[config.TARGET_COLUMN] == "Yes").astype(int)

    # Plot 1: Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#2b5c8f", "#d9534f"]
    sns.countplot(data=df_plot, x=config.TARGET_COLUMN, palette=colors, ax=ax)
    ax.set_title("Customer Churn Distribution (Baseline Target)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Churn Status", fontsize=10)
    ax.set_ylabel("Customer Count", fontsize=10)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / len(df_plot)) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    plt.tight_layout()
    plot1_path = FIGURES_DIR / "01_target_distribution.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 1 -> {plot1_path}")

    # Plot 2: Numerical Distributions by Churn Status
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    num_cols = config.NUMERICAL_FEATURES
    titles = ["Tenure (Months)", "Monthly Charges ($)", "Total Charges ($)"]
    
    for i, col in enumerate(num_cols):
        sns.kdeplot(data=df_plot, x=col, hue=config.TARGET_COLUMN, common_norm=False,
                     palette=colors, fill=True, alpha=0.3, ax=axes[i], linewidth=1.5)
        axes[i].set_title(titles[i], fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel("Density", fontsize=9)
    plt.tight_layout()
    plot2_path = FIGURES_DIR / "02_numerical_distributions.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 2 -> {plot2_path}")

    # Plot 3: Numerical Feature Correlation Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_matrix = df_plot[num_cols + ["Churn_Binary"]].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                cbar=True, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Numerical Feature & Target Correlation Heatmap", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plot3_path = FIGURES_DIR / "03_correlation_matrix.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 3 -> {plot3_path}")

    # Plot 4: Key Categorical Features Churn Rates
    key_cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, cat in enumerate(key_cats):
        cat_churn = df_plot.groupby(cat)["Churn_Binary"].mean().reset_index()
        cat_churn["Churn_Pct"] = cat_churn["Churn_Binary"] * 100
        cat_churn = cat_churn.sort_values(by="Churn_Pct", ascending=False)

        sns.barplot(data=cat_churn, x="Churn_Pct", y=cat, palette="Reds_r", ax=axes[i])
        axes[i].set_title(f"Churn Rate by {cat}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Churn Rate (%)", fontsize=9)
        axes[i].set_ylabel(cat, fontsize=9)
        axes[i].set_xlim(0, 60)
        for p in axes[i].patches:
            width = p.get_width()
            axes[i].annotate(f"{width:.1f}%", (width + 1.2, p.get_y() + p.get_height() / 2.),
                             ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot4_path = FIGURES_DIR / "04_categorical_churn_rates.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 4 -> {plot4_path}")

    # Plot 5: Bivariate Interaction (Tenure vs. Monthly Charges by Churn)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_plot, x="tenure", y="MonthlyCharges", hue=config.TARGET_COLUMN,
                    palette=colors, alpha=0.5, s=25, ax=ax)
    ax.set_title("Interaction: Tenure vs. Monthly Charges by Churn Status", fontsize=11, fontweight="bold")
    ax.set_xlabel("Tenure (Months)", fontsize=10)
    ax.set_ylabel("Monthly Charges ($)", fontsize=10)
    plt.tight_layout()
    plot5_path = FIGURES_DIR / "05_tenure_vs_monthlycharges_churn.png"
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 5 -> {plot5_path}")


def run_eda_pipeline():
    """Main execution function for EDA."""
    df = load_clean_raw_df()
    generate_eda_report(df)
    generate_eda_plots(df)


if __name__ == "__main__":
    run_eda_pipeline()
