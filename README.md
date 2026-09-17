# Breast Cancer Detection & Diagnostic Machine Learning Benchmark

A comprehensive Machine Learning pipeline and diagnostic benchmarking suite based on the **Wisconsin Breast Cancer (Diagnostic)** dataset, **Randerson112358's tutorial**, **Wikipedia's Confusion Matrix framework**, and **top-voted Kaggle methodologies**.

---

## Background & The "Confusion Matrix Trap"

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
In Randerson112358's tutorial *"Breast Cancer Detection Using Python & Machine Learning"*, the manual extraction code originally assigned:
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

## Key Improvements from Kaggle Community Benchmarks

Following analysis of top-voted Kaggle notebooks, the following enhancements have been integrated:

1. **Multicollinearity & Correlation Analysis**:
   - Analyzes extreme collinearity among geometric triplets (`radius_mean`, `perimeter_mean`, `area_mean` have $r > 0.99$).
   - Generates full correlation heatmaps (`plots/correlation_heatmap.png`).
2. **2D Principal Component Analysis (PCA)**:
   - Compresses 30 continuous features into 2 principal components, capturing over 63% of variance and demonstrating clear geometric cluster separation (`plots/pca_2d_projection.png`).
3. **Expanded Model Zoo**:
   - Added **Support Vector Machine (SVC with RBF kernel)** and **Gradient Boosting** alongside Logistic Regression, Decision Tree, and Random Forest.
4. **Stratified 5-Fold Cross-Validation**:
   - Assesses model stability across folds ($\text{Mean} \pm \text{Std}$) rather than relying solely on a single holdout split.
5. **Clinical Decision Threshold Optimization**:
   - Standard 0.50 threshold produced 4 False Negatives (missed cancer diagnoses).
   - Tuning the decision cutoff to $\tau \approx 0.35$ reduces **missed cancers from 4 down to 1** ($\ge 98\%$ Sensitivity), reflecting real-world clinical priorities where missing a malignant tumor carries catastrophic risk.

---

## Benchmark Results

### 1. Stratified 5-Fold Cross-Validation Performance ($\text{Mean} \pm \text{Std}$)

| Model | CV Accuracy | CV Sensitivity (Recall) | CV Precision | CV F1-Score | CV ROC-AUC |
|---|---|---|---|---|---|
| **Support Vector Machine (RBF)** | **97.54% +/- 2.0%** | **95.76% +/- 3.7%** | 97.65% +/- 2.6% | **0.9666** | 0.9947 |
| **Logistic Regression** | 97.37% +/- 1.7% | 94.36% +/- 5.2% | **98.63% +/- 1.8%** | 0.9633 | **0.9953** |
| **Random Forest** | 95.61% +/- 2.0% | 92.95% +/- 5.3% | 95.44% +/- 4.1% | 0.9400 | 0.9924 |
| **Gradient Boosting** | 95.08% +/- 2.5% | 91.07% +/- 6.8% | 95.69% +/- 3.0% | 0.9315 | 0.9927 |
| **Decision Tree** | 93.84% +/- 2.2% | 92.47% +/- 4.5% | 91.28% +/- 3.5% | 0.9179 | 0.9357 |

### 2. Holdout Test Set Performance ($N=143$, 25% Stratified Holdout)

| Model | Accuracy | Sensitivity (Recall / TPR) | Specificity (TNR) | Precision (PPV) | Miss Rate (FNR) | F1-Score |
|---|---|---|---|---|---|---|
| **Random Forest** | **97.20%** | 92.45% | **100.00%** | **100.00%** | 7.55% | **0.9608** |
| **Support Vector Machine (RBF)** | **97.20%** | 92.45% | **100.00%** | **100.00%** | 7.55% | **0.9608** |
| **Logistic Regression** | 96.50% | **92.45%** | 98.89% | 98.00% | 7.55% | 0.9515 |
| **Gradient Boosting** | 96.50% | 90.57% | **100.00%** | **100.00%** | 9.43% | 0.9505 |
| **Decision Tree** | 94.41% | 88.68% | 97.78% | 95.92% | 11.32% | 0.9216 |

---

## Project Structure

```
├── breast_cancer_detection.py      # Standalone ML pipeline with 5-fold CV & threshold tuning
├── breast_cancer_detection.ipynb   # Executed educational Jupyter Notebook with all outputs
├── confusion_matrix_deepdive.md    # Detailed mathematical reference guide
├── data.csv                        # Kaggle Wisconsin Breast Cancer dataset
├── plots/                          # Exported high-resolution visualization charts
│   ├── correlation_heatmap.png
│   ├── pca_2d_projection.png
│   ├── threshold_tuning_tradeoff.png
│   ├── confusion_matrices_all_models.png
│   ├── confusion_matrices_comparison.png
│   ├── roc_curves.png
│   └── feature_importance.png
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Pinned dependencies
└── README.md                       # Project documentation
```

---

## Quickstart & Usage

### 1. Environment Setup
Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
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

### 2. Run the Benchmark Pipeline
```bash
uv run python breast_cancer_detection.py
```
This runs data cleaning, correlation analysis, 2D PCA, 5-fold CV, holdout evaluation, threshold tuning, and saves charts to `./plots/`.

### 3. Launch the Interactive Jupyter Notebook
```bash
uv run jupyter notebook breast_cancer_detection.ipynb
```

---

## Referenced Resources

- **Wikipedia**: [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- **Medium Article**: [Randerson112358's Breast Cancer Detection](https://randerson112358.medium.com/breast-cancer-detection-using-machine-learning-38820fe98982)
- **Kaggle**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
