# RetainXAI: Explainability Analysis of Neural Networks Using Feature Attribution Methods

An industry-grade, end-to-end Machine Learning and Explainable AI (XAI) project evaluating feature attribution methods (**SHAP**, **LIME**, **Integrated Gradients / Captum**, and **Permutation Importance**) on a PyTorch Neural Network trained for Customer Churn Prediction.

---

## 📁 Repository Structure

```
RetainXAI/
├── data/
│   ├── raw/                  # Original downloaded raw dataset
│   └── processed/            # Cleaned, encoded, and split CSV files (train/val/test)
├── notebooks/                # Jupyter / Colab research notebooks
│   ├── 01_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_explainability.ipynb
├── src/                      # Modular production source code
│   ├── __init__.py
│   ├── data_preprocessing.py # Automated cleaning, scaling, & encoding pipeline
│   ├── feature_engineering.py# Feature transformations & interaction features
│   ├── model.py              # PyTorch Neural Network architecture
│   ├── train.py              # Model training loop with early stopping
│   ├── evaluate.py           # Comprehensive classification metrics & ROC/PR curves
│   ├── explainability.py     # Attribution generators (SHAP, LIME, Captum, Permutation)
│   └── visualization.py      # Plotting utilities for metrics & attribution plots
├── models/                   # Saved model weights (.pt) & fitted preprocessor (.joblib)
├── reports/                  # Generated XAI reports & experiment tracking tables
├── app/
│   └── app.py                # Multi-page interactive Streamlit dashboard
├── config.py                 # Centralized configuration & hyperparameter defaults
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation & execution guide
```

---

## ⚙️ Environment Setup & Installation

### 1. Local Python Setup
Ensure Python 3.10+ is installed.

```bash
# Navigate to project directory
cd RetainXAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Next Steps
* Proceed through modular pipeline execution: Data Ingestion $\rightarrow$ EDA $\rightarrow$ Preprocessing $\rightarrow$ Model Training $\rightarrow$ Feature Attribution Analysis $\rightarrow$ Web App Deployment.
