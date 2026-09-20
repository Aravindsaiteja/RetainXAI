# Phase 1: Project Definition and Dataset Selection

**Project Title:** RetainXAI — Explainability Analysis of Neural Networks Using Feature Attribution Methods  
**Domain:** Telecommunications / Customer Retention  
**Selected Dataset:** IBM Telco Customer Churn Dataset  

---

## 1. Project Title & Overview
The objective of this project is to build an end-to-end Explainable AI (XAI) pipeline using a PyTorch Neural Network trained on tabular data. The system predicts customer churn and evaluates model decisions using multiple feature attribution techniques (**SHAP**, **LIME**, **Integrated Gradients**, and **Permutation Feature Importance**).

---

## 2. Dataset Selection & Source Details

* **Dataset Name:** Telco Customer Churn Dataset
* **Dataset Source:** IBM Community / Kaggle Public Repository
* **Direct Access URL:** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
* **Dataset Format:** CSV (Comma-Separated Values)

---

## 3. Business Problem Definition
Customer churn represents the percentage of subscribers who discontinue their service within a given time frame. Acquiring new customers costs 5x to 25x more than retaining existing ones. 

### Why Black-Box Models Fail in Business Operations:
While deep neural networks can capture non-linear relationships and interactions between contract types, tenure, and monthly pricing, business operators cannot act on raw prediction probabilities (e.g., "78% churn probability"). Marketing and retention teams need to know **WHY** a specific customer is predicted to churn:
* Is it due to high monthly charges on a month-to-month contract?
* Is it due to lack of tech support or frequent service disruptions on fiber optic lines?
* Is it due to payment method friction (e.g., unautomated electronic check)?

---

## 4. Dataset Specification & Dimensions

* **Total Samples (Rows):** 7,043
* **Total Columns:** 21 (1 ID column, 19 Feature columns, 1 Target column)
* **Target Variable:** `Churn` (Categorical: `Yes` / `No` $\rightarrow$ Encoded to `1` / `0`)

### Class Distribution:
* **No (Non-Churner / Retained):** 5,174 instances (73.46%)
* **Yes (Churner):** 1,869 instances (26.54%)
* **Class Imbalance Ratio:** ~2.77 : 1 (Moderate imbalance requiring balanced metrics like F1-Score, PR-AUC, and ROC-AUC)

---

## 5. Feature Inventory & Data Types

| # | Feature Name | Data Type | Description & Category |
|---|---|---|---|
| 1 | `customerID` | Categorical (ID) | Unique identifier for customer (dropped during training) |
| 2 | `gender` | Categorical | Male / Female |
| 3 | `SeniorCitizen` | Numerical / Binary | 1 = Yes, 0 = No |
| 4 | `Partner` | Categorical | Whether customer has a partner (Yes, No) |
| 5 | `Dependents` | Categorical | Whether customer has dependents (Yes, No) |
| 6 | `tenure` | Numerical (Integer) | Months the customer has stayed with company (0 - 72) |
| 7 | `PhoneService` | Categorical | Whether customer has phone service (Yes, No) |
| 8 | `MultipleLines` | Categorical | Yes, No, No phone service |
| 9 | `InternetService` | Categorical | DSL, Fiber optic, No |
| 10 | `OnlineSecurity` | Categorical | Yes, No, No internet service |
| 11 | `OnlineBackup` | Categorical | Yes, No, No internet service |
| 12 | `DeviceProtection` | Categorical | Yes, No, No internet service |
| 13 | `TechSupport` | Categorical | Yes, No, No internet service |
| 14 | `StreamingTV` | Categorical | Yes, No, No internet service |
| 15 | `StreamingMovies` | Categorical | Yes, No, No internet service |
| 16 | `Contract` | Categorical | Month-to-month, One year, Two year |
| 17 | `PaperlessBilling` | Categorical | Yes, No |
| 18 | `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer (auto), Credit card (auto) |
| 19 | `MonthlyCharges` | Numerical (Float) | Amount charged monthly ($18.25 - $118.75) |
| 20 | `TotalCharges` | Numerical (Float) | Total amount charged ($18.80 - $8684.80) |
| 21 | `Churn` | Categorical (Target) | Target variable (Yes / No) |

* **Numerical Features (3):** `tenure`, `MonthlyCharges`, `TotalCharges` (+ binary `SeniorCitizen`)
* **Categorical Features (16):** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

---

## 6. Data Quality & Missing Value Assessment

* **Missing Values:** `TotalCharges` contains 11 records with whitespace strings (`" "`) corresponding to new customers with `tenure == 0`.
* **Data Leakage Risk:** `customerID` must be removed before model input.
* **Feature Multi-Collinearity:** `TotalCharges` is correlated with `tenure` $\times$ `MonthlyCharges`. Feature attribution methods (especially SHAP and Integrated Gradients) will be tested for how they handle correlated inputs.
* **Categorical Sub-categories:** Features like `OnlineSecurity` have sub-categories ("No internet service") that correlate directly with `InternetService == 'No'`.

---

## 7. Why Explainability (XAI) Matters for This Problem

1. **Targeted Interventions:** Retention campaigns cost money. Explaining individual churn predictions enables offering tailored discounts (e.g. long-term contract discounts for month-to-month subscribers).
2. **Model Trust & Reliability:** Neural networks often learn spurious correlations. XAI ensures the model bases decisions on business-logical features rather than artifacts.
3. **Comparative Analysis of Attribution Methods:** Tabular neural networks pose specific challenges for attribution methods:
   * **SHAP (Kernel/Deep):** Grounded in cooperative game theory (Shapley values), provides fair credit distribution.
   * **LIME:** Fits local linear surrogate models around individual instances.
   * **Integrated Gradients:** Axiomatically sound (Completeness & Implementation Invariance), evaluates gradients along path from baseline.
   * **Permutation Importance:** Model-agnostic global baseline measuring performance loss under feature shuffling.
