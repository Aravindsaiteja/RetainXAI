"""
Configuration management for Neural Network Explainability Project.
Centralizes paths, hyperparameters, seeds, data schemas, and feature lists.
"""

import os
from pathlib import Path
import torch

# Base Directory (RetainXAI root)
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
APP_DIR = BASE_DIR / "app"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, NOTEBOOKS_DIR, SRC_DIR, MODELS_DIR, REPORTS_DIR, APP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Data Source
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
MODEL_FILE = MODELS_DIR / "nn_churn_model.pt"
EXPERIMENT_LOG_FILE = REPORTS_DIR / "experiment_tracking.csv"

# Global Experiment Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # 15% val, 15% test, 70% train

# Feature Definitions
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUMERICAL_FEATURES = ["SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Neural Network Architecture Defaults
NN_HYPERPARAMETERS = {
    "input_dim": None,  # Computed after one-hot encoding
    "hidden_dim_1": 64,
    "hidden_dim_2": 32,
    "output_dim": 1,
    "dropout_rate": 0.25,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10,
    "weight_decay": 1e-4,
}

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
