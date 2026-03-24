# Credit Card Fraud Detection Analysis 

This project provides a rigorous comparative study of machine learning algorithms for detecting fraudulent transactions in highly imbalanced datasets. Developed as part of the **BIL 476 - Data Mining* course, this repository explores the challenges of "needle-in-a-haystack" anomaly detection.

## Project Overview
Fraud detection is a critical application of AI in fintech. The primary challenge is the **extreme class imbalance**, where fraudulent transactions account for only **0.173%** of the total data (492 out of 284,807).

### Key Features
- **Data Preprocessing:** Handled outliers using `RobustScaler` and addressed class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
- **Comparative Modeling:** Evaluated three distinct philosophies:
  - **Random Forest:** Supervised ensemble learning (The Champion).
  - **Isolation Forest:** Tree-based unsupervised anomaly detection.
  - **Local Outlier Factor (LOF):** Density-based unsupervised outlier detection.
- **Robust Evaluation:** Prioritized **Precision-Recall** and **ROC-AUC** metrics over misleading accuracy scores.

##  Performance Summary
Our analysis demonstrates the power of SMOTE-enhanced supervised learning. While **Random Forest** achieved an impressive **AUC of 0.983**, unsupervised methods like LOF struggled to differentiate sophisticated fraud patterns from normal behavioral noise.

| Model | AUC Score | Primary Metric |
| :--- | :--- | :--- |
| **Random Forest (with SMOTE)** | **0.983** | High Recall (0.86) |
| Isolation Forest | 0.958 | Moderate Precision |
| Local Outlier Factor | 0.506 | Baseline Performance |

##  Tech Stack
- **Language:** Python 3.x
- **Environment:** Google Colab / Jupyter Notebook
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Imbalanced-learn

## 📜How to Use
1. Clone the repository.
2. Upload the `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. Run the notebook to reproduce the results and visualizations.
