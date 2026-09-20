# Model Card: PyTorch Tabular Neural Network for Customer Churn Prediction

---

## 1. Model Details
* **Model Name:** `TabularChurnNN`
* **Model Type:** Multi-Layer Perceptron (MLP) for Tabular Binary Classification
* **Framework:** PyTorch (v2.0+)
* **Architecture:** Input (47 units) $\rightarrow$ Dense(64) + BatchNorm1d + ReLU + Dropout(0.25) $\rightarrow$ Dense(32) + BatchNorm1d + ReLU + Dropout(0.20) $\rightarrow$ Linear(1 Logit)
* **Parameters:** 5,377 trainable parameters
* **Saved Weights:** `models/nn_churn_model.pt`
* **Fitted Preprocessor:** `models/preprocessor.joblib`

---

## 2. Intended Use
* **Primary Intended Use:** Predict subscriber churn probability $P(\text{Churn})$ for telecommunications customer retention operations.
* **Primary Intended Users:** Customer Success Managers, Marketing Operations, Retention Teams, XAI Compliance Officers.
* **Out-of-Scope Use Cases:** Credit scoring, automated contract cancellation without human-in-the-loop review, real-time physical service cutoff decisions.

---

## 3. Training & Evaluation Datasets
* **Dataset Name:** IBM Telco Customer Churn Dataset (7,043 rows $\times$ 21 raw columns)
* **Splitting Strategy:** Stratified Train (70%, 4,929 rows), Validation (15%, 1,057 rows), Test (15%, 1,057 rows)
* **Class Balance:** Retained (73.46%) vs. Churned (26.54%). Imbalance weight $\text{pos\_weight} = 2.7683$ applied in `BCEWithLogitsLoss`.

---

## 4. Quantitative Evaluation Metrics (Test Set, N = 1,057)

| Metric | Score | Operational Significance |
| :--- | :-: | :--- |
| **Accuracy** | **74.36%** | Overall correct classifications |
| **Precision** | **50.97%** | Interventions targeting true churners |
| **Recall (Sensitivity)** | **84.29%** | **Primary Business Objective** (Catches 236 out of 280 churners) |
| **F1-Score** | **0.6353** | Harmonic mean of precision & recall |
| **ROC-AUC** | **0.8481** | High class separation capability |
| **PR-AUC** | **0.6527** | Superior performance over baseline (0.2649) |

---

## 5. Explainability (XAI) Methods Integrated

1. **SHAP (KernelExplainer):** Cooperative game-theoretic Shapley value attribution.
2. **LIME (LimeTabularExplainer):** Local linear surrogate decision boundary rules.
3. **Integrated Gradients (PyTorch Captum):** Axiomatically verified straight-line path integral gradients ($\sum \text{IG} = F(x) - F(x')$).
4. **Permutation Importance:** Model-agnostic performance degradation baseline under feature shuffling.

---

## 6. Caveats & Ethical Considerations
1. **Feature Attribution $\neq$ Real-World Causality:** High feature attribution indicates model dependency, not necessarily a physical cause-and-effect relationship in real-world customer behavior.
2. **Collinearity Artifacts:** `TotalCharges` is correlated with `tenure` $\times$ `MonthlyCharges`. Linear surrogates (LIME) may split weight across collinear pairs.
3. **Demographic Neutrality:** Demographic attributes (`gender`, `SeniorCitizen`) contribute $< 1\%$ to model decisions, ensuring non-discriminatory retention scoring.
