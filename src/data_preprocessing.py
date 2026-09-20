"""
Data Preprocessing Pipeline Module.
Implements leak-free data splitting, imputations, scaling, and encoding.
Saves fitted preprocessor artifact to models/preprocessor.joblib.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.feature_engineering import add_engineered_features


def load_and_clean_dataset() -> pd.DataFrame:
    """Loads raw data, strips Whitespace, handles numeric conversions, and adds engineered features."""
    df = pd.read_csv(config.RAW_DATA_FILE)

    # 1. Handle Whitespace in TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    # 2. Drop customerID (ID column) to prevent leakage
    if config.ID_COLUMN in df.columns:
        df = df.drop(columns=[config.ID_COLUMN])

    # 3. Add Feature Engineering
    df = add_engineered_features(df)

    # 4. Binary Encode Target Column
    if config.TARGET_COLUMN in df.columns:
        df[config.TARGET_COLUMN] = (df[config.TARGET_COLUMN] == "Yes").astype(int)

    return df


def build_preprocessing_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline:
    - Numerical: Median Imputer + StandardScaler
    - Categorical: OneHotEncoder (handle_unknown='ignore', dense output)
    """
    num_pipeline = ColumnTransformer(
        transformers=[
            ("imputer", SimpleImputer(strategy="median"), slice(None)),
        ]
    )

    # Scikit-Learn ColumnTransformer for mixed types
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline_Num := SimpleImputer(strategy="median"),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


class ChurnPreprocessor:
    """
    Wrapper class to manage feature scaling, encoding, feature name tracking,
    and artifact persistence.
    """

    def __init__(self, num_cols: list, cat_cols: list):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        """Fits imputer, scaler, and one-hot encoder ONLY on training data."""
        # Numerical pipeline
        X_num = X[self.num_cols].values
        X_num_imp = self.num_imputer.fit_transform(X_num)
        self.scaler.fit(X_num_imp)

        # Categorical pipeline
        X_cat = X[self.cat_cols].values
        self.encoder.fit(X_cat)

        # Feature names tracking
        cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + cat_feature_names
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame using fitted components."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform!")

        # Numerical transform
        X_num = X[self.num_cols].values
        X_num_imp = self.num_imputer.transform(X_num)
        X_num_scaled = self.scaler.transform(X_num_imp)

        # Categorical transform
        X_cat = X[self.cat_cols].values
        X_cat_encoded = self.encoder.transform(X_cat)

        # Concatenate transformed features
        X_trans = np.hstack([X_num_scaled, X_cat_encoded])
        return pd.DataFrame(X_trans, columns=self.feature_names, index=X.index)


def run_preprocessing_pipeline() -> tuple:
    """
    Full reproducible preprocessing execution pipeline:
    1. Loads cleaned raw dataset.
    2. Performs Stratified Train (70%) / Val (15%) / Test (15%) split.
    3. Fits ChurnPreprocessor on Train set.
    4. Transforms Train, Val, Test sets.
    5. Saves processed datasets and fitted preprocessor artifact.
    """
    print("\n" + "=" * 65)
    print("             DATA PREPROCESSING & SPLITTING PIPELINE")
    print("=" * 65)

    df = load_and_clean_dataset()

    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    # Identify feature columns
    num_cols = config.NUMERICAL_FEATURES + ["avg_monthly_spend", "num_active_services", "is_auto_payment"]
    cat_cols = config.CATEGORICAL_FEATURES

    # 1. First Split: Separate Test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    # 2. Second Split: Separate Train (70%) and Val (15%) out of 85%
    val_relative_size = config.VAL_SIZE / (1.0 - config.TEST_SIZE)  # 0.15 / 0.85 = 0.17647
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=config.RANDOM_SEED, stratify=y_train_val
    )

    print(f"\n[1] DATASET SPLIT SUMMARY (Stratified):")
    print(f"    - Training Set:   {len(X_train):>5} samples ({len(X_train)/len(df)*100:.1f}%) | Positive Churn: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"    - Validation Set: {len(X_val):>5} samples ({len(X_val)/len(df)*100:.1f}%) | Positive Churn: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
    print(f"    - Test Set:       {len(X_test):>5} samples ({len(X_test)/len(df)*100:.1f}%) | Positive Churn: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # 3. Fit Preprocessor ONLY on Training Set
    preprocessor = ChurnPreprocessor(num_cols=num_cols, cat_cols=cat_cols)
    preprocessor.fit(X_train)

    print(f"\n[2] PREPROCESSING & FEATURE ENCODING:")
    print(f"    - Raw Input Features: {len(X.columns)} columns")
    print(f"    - Transformed Features (One-Hot Encoded): {len(preprocessor.feature_names)} columns")

    # 4. Transform Datasets
    X_train_proc = preprocessor.transform(X_train).reset_index(drop=True)
    X_val_proc = preprocessor.transform(X_val).reset_index(drop=True)
    X_test_proc = preprocessor.transform(X_test).reset_index(drop=True)

    # Attach Target Column with aligned indices
    df_train_proc = pd.concat([X_train_proc, y_train.reset_index(drop=True)], axis=1)
    df_val_proc = pd.concat([X_val_proc, y_val.reset_index(drop=True)], axis=1)
    df_test_proc = pd.concat([X_test_proc, y_test.reset_index(drop=True)], axis=1)

    # 5. Save Artifacts
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train_proc.to_csv(config.PROCESSED_TRAIN_FILE, index=False)
    df_val_proc.to_csv(config.PROCESSED_VAL_FILE, index=False)
    df_test_proc.to_csv(config.PROCESSED_TEST_FILE, index=False)

    joblib.dump(preprocessor, config.PREPROCESSOR_FILE)
    print(f"\n[3] ARTIFACTS PERSISTENCE:")
    print(f"    [SAVED] Train Set        -> {config.PROCESSED_TRAIN_FILE}")
    print(f"    [SAVED] Validation Set   -> {config.PROCESSED_VAL_FILE}")
    print(f"    [SAVED] Test Set         -> {config.PROCESSED_TEST_FILE}")
    print(f"    [SAVED] Preprocessor Obj -> {config.PREPROCESSOR_FILE}")

    # Compute positive class weight for loss function
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = neg_count / pos_count
    print(f"\n[4] CLASS IMBALANCE RATIO:")
    print(f"    - Train Negatives: {neg_count}, Positives: {pos_count}")
    print(f"    - Calculated pos_weight for BCE Loss: {pos_weight:.4f}")

    print("\n" + "=" * 65 + "\n")
    return df_train_proc, df_val_proc, df_test_proc, preprocessor, pos_weight


if __name__ == "__main__":
    run_preprocessing_pipeline()
