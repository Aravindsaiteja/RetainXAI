"""
Feature Engineering Module.
Creates domain-specific interaction and aggregate features to enhance model expressiveness
without introducing data leakage.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the raw or cleaned DataFrame.
    
    New Features Added:
    1. avg_monthly_spend: TotalCharges / (tenure + 1)
    2. num_active_services: Count of active security, backup, protection, and streaming services.
    3. is_auto_payment: Binary flag indicating automatic payment method (Bank transfer or Credit card).
    """
    df = df.copy()

    # 1. Financial interaction feature: Average monthly spend per tenure month
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        # Fill temp missing TotalCharges for calculation if needed
        total_charges_clean = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["avg_monthly_spend"] = total_charges_clean / (df["tenure"] + 1.0)

    # 2. Service aggregation feature: Number of active add-on services
    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    existing_addon_cols = [col for col in addon_cols if col in df.columns]
    if existing_addon_cols:
        # Active service count (value == 'Yes')
        df["num_active_services"] = (df[existing_addon_cols] == "Yes").sum(axis=1)

    # 3. Behavioral payment feature: Automatic vs Manual payment method
    if "PaymentMethod" in df.columns:
        df["is_auto_payment"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)

    return df
