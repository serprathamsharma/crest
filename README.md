# 🔬 Breast Cancer Detection & Confusion Matrix Analysis

A comprehensive Machine Learning pipeline and diagnostic benchmarking suite based on the **Wisconsin Breast Cancer (Diagnostic)** dataset, **Randerson112358's tutorial & video**, and **Wikipedia's Confusion Matrix framework**.

---

## 📌 Background & The "Confusion Matrix Trap"

In binary medical classification, evaluating predictive models requires analyzing **True Positives (TP)**, **True Negatives (TN)**, **False Positives (FP)**, and **False Negatives (FN)**.

However, developers frequently encounter a **subtle axis convention discrepancy**:

### 1. The Discrepancy
- **Wikipedia / Traditional Medical Standards**:
  - The condition of interest (**Malignant / Positive**) is placed **first**.
  - Row 0 = Positive, Col 0 = Positive $\rightarrow$ `[0, 0] = True Positive (TP)`.
- **Scikit-Learn (`sklearn.metrics.confusion_matrix`)**:
  - Labels are ordered numerically/alphabetically: `0` (**Benign / Negative**) precedes `1` (**Malignant / Positive**).
  - Row 0 = Negative, Col 0 = Negative $\rightarrow$ `[0, 0] = True Negative (TN)` and `[1, 1] = True Positive (TP)`.

### 2. The Randerson112358 Tutorial Pitfall
In Randerson112358's widely cited tutorial *"Breast Cancer Detection Using Python & Machine Learning"* (YouTube: `NSSOyhJBmWY`), the manual extraction code originally assigned:
```python
# The tutorial's initial code:
TP = cm[0][0]  # Actually TN in scikit-learn!
TN = cm[1][1]  # Actually TP in scikit-learn!
```
While overall accuracy remained identical ($\frac{TP+TN}{\text{Total}}$ is commutative), **Sensitivity (Recall)** and **Specificity** were inadvertently flipped.

### 3. Safe Solutions in Python
```python
# Solution A: Idiomatic unpacking (labels=[0, 1])
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Solution B: Explicitly match Wikipedia orientation (Positive first)
cm_wiki = confusion_matrix(y_true, y_pred, labels=[1, 0])
tp, fn, fp, tn = cm_wiki.ravel()
```

---

## 📊 Benchmark Results

Evaluated on a 25% stratified test holdout ($N=143$: 90 Benign, 53 Malignant) from Kaggle's `data.csv`:

| Model | Accuracy | Sensitivity (Recall / TPR) | Specificity (TNR) | Precision (PPV) | Miss Rate (FNR) | F1-Score |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 96.50% | 92.45% | 98.89% | 98.00% | 7.55% | 0.9515 |
| **Decision Tree** | 94.41% | 88.68% | 97.78% | 95.92% | 11.32% | 0.9216 |
| **Random Forest** | **97.20%** | **92.45%** | **100.00%** | **100.00%** | **7.55%** | **0.9608** |

> **Clinical Takeaway**: In oncology screening, **False Negatives (Miss Rate)** are far more perilous than False Positives. Random Forest achieved **0 False Positives** (100% Specificity & Precision) and a strong 92.45% Sensitivity.

---

## 📂 Project Structure

```
├── breast_cancer_detection.py      # Standalone, end-to-end ML pipeline
├── breast_cancer_detection.ipynb   # Executed, educational Jupyter Notebook
├── confusion_matrix_deepdive.md    # Detailed mathematical reference guide
├── data.csv                        # Kaggle Wisconsin Breast Cancer dataset
├── plots/                          # Exported high-resolution visualization charts
│   ├── confusion_matrices_comparison.png
│   ├── confusion_matrices_all_models.png
│   ├── models_performance_comparison.png
│   ├── roc_curves.png
│   └── feature_importance.png
├── pyproject.toml                  # uv / project configuration
├── requirements.txt                # Pinned dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
# Dependencies install automatically on run:
uv run python breast_cancer_detection.py
```

Or using standard `pip`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Command-Line Pipeline
```bash
uv run python breast_cancer_detection.py
```
This loads `data.csv`, cleans and scales features, trains the 3 classifiers, outputs metrics to the terminal, and saves high-resolution charts to `./plots/`.

### 3. Launch the Interactive Jupyter Notebook
```bash
uv run jupyter notebook breast_cancer_detection.ipynb
```

---

## 🔗 Referenced Resources

- **Wikipedia**: [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- **Medium Article**: [Randerson112358's Breast Cancer Detection](https://randerson112358.medium.com/breast-cancer-detection-using-machine-learning-38820fe98982)
- **YouTube Video**: [Breast Cancer Detection Tutorial (`NSSOyhJBmWY`)](https://www.youtube.com/watch?v=NSSOyhJBmWY)
- **Kaggle**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
