# Credit Card Fraud Detection Analysis

**BIL 476 — Data Mining Course Project | Spring 2026**  
TOBB University of Economics and Technology

---

## Overview

This project presents a comparative study of machine learning
algorithms for credit card fraud detection under conditions of
extreme class imbalance. The dataset contains 284,807 transactions
of which only 492 (0.173%) are fraudulent — a classic
needle-in-a-haystack problem.

Four configurations are evaluated: two supervised Random Forest
variants (with and without SMOTE-based oversampling) and two
unsupervised anomaly detection methods (Isolation Forest and
Local Outlier Factor). The central finding is that the
**unbalanced Random Forest outperformed the SMOTE-augmented
model on F1-Score (0.880 vs. 0.558) and PR-AUC (0.863 vs. 0.801)**,
suggesting the PCA-transformed features alone provide sufficient
separability without explicit balancing.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions (September 2013, European cardholders)
- **Features:** 30 input features (V1–V28 via PCA, plus Time and Amount) + 1 target
- **Class distribution:** 99.827% Normal / 0.173% Fraud
- **Missing values:** None

---

## Methodology

### Preprocessing
- `RobustScaler` applied to `Time` and `Amount` features (resistant to outliers)
- Stratified 80/20 train-test split to preserve the original fraud ratio
- SMOTE applied **only to the training set** to avoid data leakage

### Models
| Model | Type | Balancing |
|:---|:---|:---|
| Random Forest (No SMOTE) | Supervised | None |
| Random Forest (SMOTE) | Supervised | SMOTE |
| Isolation Forest | Unsupervised | None |
| Local Outlier Factor | Unsupervised | None |

### Evaluation Metrics
Standard accuracy is meaningless here (a model predicting all
transactions as normal achieves 99.83%). The following metrics
are used instead: **Precision, Recall, F1-Score, PR-AUC, ROC-AUC**.

---

## Results

| Model | Precision | Recall | F1-Score | PR-AUC |
|:---|:---:|:---:|:---:|:---:|
| **RF (No SMOTE)** | **0.942** | 0.827 | **0.880** | **0.863** |
| RF (SMOTE) | 0.414 | **0.857** | 0.558 | 0.801 |
| Isolation Forest | 0.297 | 0.337 | 0.316 | 0.162 |
| Local Outlier Factor | 0.005 | 0.143 | 0.010 | 0.003 |

**Key finding:** SMOTE marginally improved Recall (+3.7%) but
caused a severe Precision drop (0.942 → 0.414), pushing the
decision boundary outward. Both supervised models substantially
outperformed the unsupervised methods.

---

## Key Observations

- **ROC-AUC can be misleading** on imbalanced data. Isolation
  Forest achieved ROC-AUC = 0.958 yet PR-AUC = 0.162 —
  a direct consequence of the massive true-negative count
  inflating the ROC curve.
- **SMOTE provides diminishing returns** on well-separated
  datasets. The PCA features (especially V14, V12, V4) already
  give Random Forest sufficient discriminative power.
- **Unsupervised methods failed** in this high-dimensional
  PCA-transformed space where fraudulent transactions are
  designed to mimic normal behavior.

---

## Tech Stack

- **Language:** Python 3.x
- **Environment:** Google Colab / Jupyter Notebook
- **Libraries:** `scikit-learn`, `imbalanced-learn`, `pandas`,
  `numpy`, `matplotlib`, `seaborn`

---

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/emiryucedag/Credit-Card-Fraud-Detection-Analysis
```
2. Download `creditcard.csv` from
   [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   and place it in your Google Drive under `MyDrive/BIL476/`.
3. Open `BIL476_Project.ipynb` in Google Colab.
4. Update the `file_path` variable in Cell 3 if your folder
   structure differs.
5. Run all cells sequentially. Training takes approximately
   2–3 minutes for Random Forest.

---

## Repository Structure
```
├── BIL476_Project.ipynb        # Main notebook (all code)
├── BIL476_Project_Report.pdf   # IEEE-format report
└── README.md
```

---

## Report

The full IEEE-formatted report is included in this repository.
It covers dataset analysis, preprocessing justification,
algorithm descriptions, results, and a critical discussion of
the findings including the unexpected SMOTE behavior on this
dataset.

---

## Academic Context

**Course:** BIL 476 — Data Mining, Spring 2026  
**Institution:** TOBB University of Economics and Technology  
**Student:** Emir Yücedağ (221101077)
