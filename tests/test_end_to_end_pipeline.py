"""
End-to-End Automated Regression Test Suite.
Validates data ingestion, preprocessing, PyTorch model forward pass, checkpoint integrity,
evaluation metrics, XAI attribution reports, generated figures, and Streamlit app compatibility.
"""

import sys
import os
from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np
import torch
import joblib

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_validation import load_and_validate_raw_data
from src.data_preprocessing import ChurnPreprocessor
from src.model import TabularChurnNN
from src.feature_engineering import add_engineered_features


def test_01_data_validation():
    """Verifies raw data ingestion and schema validation audit."""
    assert config.RAW_DATA_FILE.exists(), f"Raw data file missing at {config.RAW_DATA_FILE}"
    df_raw = pd.read_csv(config.RAW_DATA_FILE)
    assert len(df_raw) == 7043, f"Unexpected row count: {len(df_raw)}"
    assert len(df_raw.columns) == 21, f"Unexpected column count: {len(df_raw.columns)}"
    assert config.TARGET_COLUMN in df_raw.columns, "Target column missing"


def test_02_preprocessing_artifact():
    """Verifies preprocessor joblib object and transformed feature shapes."""
    import sys
    sys.modules["__main__"].ChurnPreprocessor = ChurnPreprocessor
    assert config.PREPROCESSOR_FILE.exists(), f"Preprocessor object missing at {config.PREPROCESSOR_FILE}"
    preprocessor = joblib.load(config.PREPROCESSOR_FILE)
    assert hasattr(preprocessor, "feature_names"), "Preprocessor missing feature_names attribute"
    assert len(preprocessor.feature_names) == 47, f"Expected 47 transformed features, got {len(preprocessor.feature_names)}"

    # Check processed train CSV
    assert config.PROCESSED_TRAIN_FILE.exists(), "Processed train CSV missing"
    df_train = pd.read_csv(config.PROCESSED_TRAIN_FILE)
    assert df_train.isnull().sum().sum() == 0, "Processed train CSV contains NaN values!"
    assert df_train.shape[1] == 48, f"Expected 48 columns (47 features + 1 target), got {df_train.shape[1]}"


def test_03_pytorch_model_architecture():
    """Verifies PyTorch model instantiation, forward pass, and predict_proba shape."""
    dummy_input = torch.randn(8, 47)
    model = TabularChurnNN(input_dim=47, hidden_dim_1=128, hidden_dim_2=64)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)

    assert logits.shape == (8, 1), f"Expected logits shape (8, 1), got {logits.shape}"
    assert probs.shape == (8, 2), f"Expected predict_proba shape (8, 2), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of [0, 1] range!"


def test_04_model_checkpoint_integrity():
    """Verifies saved PyTorch model checkpoint state dictionary."""
    assert config.MODEL_FILE.exists(), f"Trained model checkpoint missing at {config.MODEL_FILE}"
    model = TabularChurnNN(input_dim=47, hidden_dim_1=128, hidden_dim_2=64)
    state_dict = torch.load(config.MODEL_FILE, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Perform inference test on processed test instance
    df_test = pd.read_csv(config.PROCESSED_TEST_FILE)
    X_sample = torch.tensor(df_test.drop(columns=[config.TARGET_COLUMN]).values[:5], dtype=torch.float32)
    probs = model.predict_proba(X_sample)
    assert probs.shape == (5, 2), "Inference on test sample failed!"


def test_05_evaluation_metrics_integrity():
    """Verifies test evaluation metrics JSON report."""
    metrics_file = config.REPORTS_DIR / "test_evaluation_metrics.json"
    assert metrics_file.exists(), f"Evaluation metrics report missing at {metrics_file}"

    with open(metrics_file) as f:
        metrics = json.load(f)

    assert "roc_auc" in metrics, "ROC-AUC missing from evaluation report"
    assert metrics["roc_auc"] >= 0.80, f"ROC-AUC below operational threshold: {metrics['roc_auc']}"
    assert metrics["recall"] >= 0.80, f"Recall below operational threshold: {metrics['recall']}"


def test_06_xai_reports_and_figures_existence():
    """Verifies all required XAI CSV reports and figures exist."""
    required_reports = [
        config.REPORTS_DIR / "experiment_tracking.csv",
        config.REPORTS_DIR / "shap_global_importance.csv",
        config.REPORTS_DIR / "lime_feature_weights.csv",
        config.REPORTS_DIR / "integrated_gradients_attributions.csv",
        config.REPORTS_DIR / "permutation_importance.csv",
        config.REPORTS_DIR / "feature_attribution_comparison.csv",
        config.REPORTS_DIR / "local_explanation_analysis.csv",
        config.REPORTS_DIR / "global_explainability_summary.json",
        config.REPORTS_DIR / "explanation_stability_results.csv",
        config.REPORTS_DIR / "explanation_faithfulness_results.csv",
        config.REPORTS_DIR / "ablation_study_results.csv",
        config.REPORTS_DIR / "counterfactual_analysis.csv",
    ]

    for report_path in required_reports:
        assert report_path.exists(), f"Required XAI report missing: {report_path.name}"

    figures_dir = config.REPORTS_DIR / "figures"
    fig_count = len(list(figures_dir.glob("*.png")))
    assert fig_count >= 20, f"Expected at least 20 generated figures, found {fig_count}"


def test_07_streamlit_app_compatibility():
    """Verifies Streamlit app module imports without exceptions."""
    import app.app
    assert hasattr(app.app, "load_model_and_preprocessor"), "Streamlit app missing load_model_and_preprocessor"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
