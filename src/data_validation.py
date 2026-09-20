"""
Data Ingestion and Validation Module.
Downloads the IBM Telco Customer Churn dataset if not present locally,
and performs strict automated data validation checks.
"""

import sys
import os
from pathlib import Path
import requests
import pandas as pd
import numpy as np

# Ensure project root is in Python path for config imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def download_dataset(url: str = config.DATASET_URL, output_path: Path = config.RAW_DATA_FILE) -> Path:
    """
    Downloads the raw Telco Customer Churn dataset from the raw source URL if not already present.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"[INFO] Raw dataset already exists at: {output_path}")
        return output_path

    print(f"[INFO] Downloading dataset from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"[SUCCESS] Raw dataset downloaded successfully to: {output_path}")
    return output_path


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Performs comprehensive automated validation checks on the raw dataset.
    
    Checks include:
    1. Schema validation (column presence and counts).
    2. Data shape verification.
    3. Target column presence and valid values.
    4. Missing value & whitespace detection.
    5. Duplicate row verification.
    """
    print("\n" + "=" * 60)
    print("               DATASET VALIDATION AUDIT REPORT")
    print("=" * 60)

    report = {
        "status": "PASSED",
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "issues": []
    }

    # Check 1: Shape verification
    print(f"\n1. Dataset Dimensions:")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    if df.shape[0] < 5000:
        report["issues"].append("Warning: Unexpected low row count.")

    # Check 2: Column schema check
    expected_columns = (
        [config.ID_COLUMN]
        + config.NUMERICAL_FEATURES
        + config.BINARY_NUMERICAL_FEATURES
        + config.CATEGORICAL_FEATURES
        + [config.TARGET_COLUMN]
    )
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        report["status"] = "FAILED"
        report["issues"].append(f"Missing required columns: {missing_cols}")
        print(f"   [FAIL] Missing required columns: {missing_cols}")
    else:
        print(f"   [PASS] All {len(expected_columns)} expected columns present.")

    # Check 3: Target variable verification
    if config.TARGET_COLUMN in df.columns:
        target_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
        print(f"\n2. Target Variable ('{config.TARGET_COLUMN}') Distribution:")
        for val, count in target_counts.items():
            pct = (count / len(df)) * 100
            print(f"   - '{val}': {count} ({pct:.2f}%)")
        
        valid_targets = {"Yes", "No", 1, 0}
        invalid_targets = set(df[config.TARGET_COLUMN].unique()) - valid_targets
        if invalid_targets:
            report["status"] = "FAILED"
            report["issues"].append(f"Invalid target values: {invalid_targets}")
            print(f"   [FAIL] Invalid target values found: {invalid_targets}")
        else:
            print(f"   [PASS] Target variable values are valid.")

    # Check 4: Missing values & Whitespace strings
    print(f"\n3. Missing Value Audit:")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"   - Standard Null Values: {total_nulls}")

    # Check whitespace strings in string columns (e.g. TotalCharges)
    whitespace_counts = {}
    for col in df.select_dtypes(include=["object"]).columns:
        ws_mask = df[col].astype(str).str.strip() == ""
        ws_cnt = ws_mask.sum()
        if ws_cnt > 0:
            whitespace_counts[col] = int(ws_cnt)

    if whitespace_counts:
        print(f"   [ALERT] Whitespace/Empty strings detected in columns:")
        for col, cnt in whitespace_counts.items():
            print(f"     * Feature '{col}': {cnt} empty whitespace records")
        report["whitespace_issues"] = whitespace_counts
    else:
        print(f"   [PASS] No hidden whitespace empty strings detected.")

    # Check 5: Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n4. Duplicate Record Check:")
    print(f"   - Duplicate Rows: {duplicates}")
    if duplicates > 0:
        report["issues"].append(f"Found {duplicates} duplicate rows.")

    # Final verdict
    print("\n" + "-" * 60)
    print(f"FINAL VALIDATION VERDICT: [{report['status']}]")
    print("-" * 60 + "\n")

    return report


def load_and_validate_raw_data() -> pd.DataFrame:
    """
    Main entry point for downloading, loading, and validating raw data.
    """
    raw_path = download_dataset()
    df = pd.read_csv(raw_path)
    audit_report = validate_dataset(df)
    
    if audit_report["status"] == "FAILED":
        raise ValueError(f"Data validation failed! Issues: {audit_report['issues']}")
        
    return df


if __name__ == "__main__":
    load_and_validate_raw_data()
