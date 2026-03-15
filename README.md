# AI Reliability & Bias Audit
This project audits the **Adult Income Dataset** to identify demographic bias in machine learning predictions.

## 📊 Key Findings
* **Data Imbalance:** Initial audit showed a significant over-representation of males in high-income brackets.
* **Model Fairness:** Measured prediction accuracy across genders to identify potential AI discrimination.

## 🛠️ Tech Stack
* **Python** (Pandas, Matplotlib, Seaborn)
* **Scikit-Learn** (Random Forest Classifier)
* **SHAP** (Explainable AI)yes
* **Explainable AI (XAI):** Implemented SHAP to visualize feature importance and verify if 'Sex' was a primary driver in model predictions.