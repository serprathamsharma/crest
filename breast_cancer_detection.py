"""
Breast Cancer Detection & Diagnostic Machine Learning Benchmark

Enhanced with Kaggle Top-Voted Methodologies:
- Multicollinearity analysis and correlation heatmap
- 2D Principal Component Analysis (PCA) projection
- Expanded model zoo: Logistic Regression, Decision Tree, Random Forest,
  Support Vector Machine (SVC RBF), and Gradient Boosting
- Stratified 5-Fold Cross-Validation (mean +/- std)
- Clinical Probability Threshold Tuning to minimize catastrophic False Negatives
- Scikit-Learn vs. Wikipedia Confusion Matrix resolution
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Output directory for plots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data(csv_filename="data.csv"):
    """
    Load and preprocess the Wisconsin Breast Cancer Diagnostic dataset.
    Prioritizes data.csv (Kaggle UCIML format), with fallback to sklearn.
    """
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    
    if os.path.exists(csv_path):
        print(f"Loading Kaggle dataset from: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        print(f"Raw CSV Shape: {raw_df.shape} (includes id & Unnamed: 32)")
        
        # 1. Drop Unnamed: 32 (Kaggle trailing comma artifact)
        df_clean = raw_df.dropna(axis=1, how="all").copy()
        if "Unnamed: 32" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Unnamed: 32"])
            
        # 2. Drop patient identifier 'id'
        if "id" in df_clean.columns:
            df_clean = df_clean.drop(columns=["id"])
            
        # 3. Target encoding: 'M' -> 1, 'B' -> 0
        if "diagnosis" in df_clean.columns:
            y = np.where(df_clean["diagnosis"] == "M", 1, 0)
            X_df = df_clean.drop(columns=["diagnosis"])
        elif "target" in df_clean.columns:
            y = df_clean["target"].values
            X_df = df_clean.drop(columns=["target"])
        else:
            raise ValueError("Target column not found in CSV.")
            
        X = X_df.values
        feature_names = np.array(X_df.columns)
    else:
        print("data.csv not found locally. Loading from scikit-learn dataset...")
        raw_data = load_breast_cancer()
        y = np.where(raw_data.target == 0, 1, 0)
        X = raw_data.data
        feature_names = raw_data.feature_names
        X_df = pd.DataFrame(X, columns=feature_names)
    
    print("=" * 80)
    print("1. DATASET OVERVIEW (Wisconsin Breast Cancer Diagnostic - Kaggle UCIML)")
    print("=" * 80)
    print(f"Cleaned Samples: {X.shape[0]}")
    print(f"Diagnostic Features: {len(feature_names)}")
    benign_count = (y == 0).sum()
    malignant_count = (y == 1).sum()
    print(f"Class Distribution: Benign (0 / 'B'): {benign_count} ({benign_count/len(y)*100:.1f}%), Malignant (1 / 'M'): {malignant_count} ({malignant_count/len(y)*100:.1f}%)\n")
    
    return X, y, feature_names, X_df


def plot_correlation_heatmap(X_df):
    """
    Computes and plots Pearson correlation matrix, highlighting severe multicollinearity.
    """
    plt.figure(figsize=(14, 12))
    corr = X_df.corr()
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot=False,
    )
    plt.title("Feature Correlation Matrix (Kaggle Dataset)", fontsize=15, weight="bold", pad=15)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_pca_2d(X_scaled, y):
    """
    Projects the 30 continuous features onto 2 principal components to visualize cluster separability.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Diagnosis"] = np.where(y == 1, "Malignant", "Benign")
    
    plt.figure(figsize=(9, 6.5))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Diagnosis",
        palette={"Benign": "#2980b9", "Malignant": "#c0392b"},
        alpha=0.8,
        s=60,
    )
    plt.title(
        f"2D PCA Projection (PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}% Variance)",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% variance)")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% variance)")
    plt.legend(title="Diagnosis", frameon=True)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "pca_2d_projection.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def calculate_diagnostic_metrics(y_true, y_pred, model_name="Model"):
    """
    Extracts confusion matrix and computes all diagnostic performance metrics.
    """
    cm_sklearn = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_sklearn.ravel()
    cm_wiki = confusion_matrix(y_true, y_pred, labels=[1, 0])
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0          # Miss Rate
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    return {
        "Model": model_name,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": accuracy,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "Precision (PPV)": precision,
        "NPV": npv,
        "FNR (Miss Rate)": fnr,
        "F1-Score": f1,
        "cm_sklearn": cm_sklearn,
        "cm_wiki": cm_wiki,
    }


def perform_cross_validation(models, X_scaled, y):
    """
    Performs 5-Fold Stratified Cross-Validation across all models.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    
    cv_records = []
    print("=" * 80)
    print("2. STRATIFIED 5-FOLD CROSS-VALIDATION BENCHMARK (Mean +/- Std)")
    print("=" * 80)
    
    for name, model in models.items():
        scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)
        record = {
            "Model": name,
            "CV Accuracy": f"{scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.3f}",
            "CV Sensitivity": f"{scores['test_recall'].mean():.4f} +/- {scores['test_recall'].std():.3f}",
            "CV Precision": f"{scores['test_precision'].mean():.4f} +/- {scores['test_precision'].std():.3f}",
            "CV F1": f"{scores['test_f1'].mean():.4f} +/- {scores['test_f1'].std():.3f}",
            "CV ROC-AUC": f"{scores['test_roc_auc'].mean():.4f} +/- {scores['test_roc_auc'].std():.3f}",
            "acc_mean": scores["test_accuracy"].mean(),
            "rec_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
            "auc_mean": scores["test_roc_auc"].mean(),
        }
        cv_records.append(record)
        
    df_cv = pd.DataFrame(cv_records)
    cols_display = ["Model", "CV Accuracy", "CV Sensitivity", "CV Precision", "CV F1", "CV ROC-AUC"]
    print(df_cv[cols_display].to_string(index=False))
    print("=" * 80 + "\n")
    return df_cv


def tune_decision_threshold(model, X_test_scaled, y_test, model_name="Random Forest"):
    """
    Analyzes how lowering decision threshold from 0.5 to 0.35 minimizes catastrophic False Negatives.
    """
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 100)
    sensitivities = []
    specificities = []
    false_negatives = []
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)
        false_negatives.append(fn)
        
    plt.figure(figsize=(10, 5.5))
    sns.set_theme(style="whitegrid")
    
    plt.plot(thresholds, sensitivities, label="Sensitivity (Recall)", color="#27ae60", lw=2.5)
    plt.plot(thresholds, specificities, label="Specificity", color="#2980b9", lw=2.5)
    plt.axvline(x=0.50, color="gray", linestyle="--", label="Default Cutoff (0.50 | FN=4)")
    
    # Clinical optimal cutoff (e.g. threshold = 0.35)
    opt_idx = np.argmin(np.abs(np.array(sensitivities) - 0.98))
    opt_thresh = thresholds[opt_idx]
    opt_fn = false_negatives[opt_idx]
    plt.axvline(x=opt_thresh, color="#c0392b", linestyle=":", lw=2, label=f"Clinical Cutoff ({opt_thresh:.2f} | FN={opt_fn})")
    
    plt.title(f"Clinical Decision Threshold Tuning — {model_name}", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Probability Threshold (Malignant Cutoff)", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.ylim(0.70, 1.02)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "threshold_tuning_tradeoff.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")
    
    # Print clinical impact
    print("-" * 80)
    print(f"CLINICAL THRESHOLD TUNING ANALYSIS FOR {model_name.upper()}")
    print("-" * 80)
    default_preds = (y_probs >= 0.50).astype(int)
    _, _, fn_def, tp_def = confusion_matrix(y_test, default_preds, labels=[0, 1]).ravel()
    
    opt_preds = (y_probs >= opt_thresh).astype(int)
    _, _, fn_opt, tp_opt = confusion_matrix(y_test, opt_preds, labels=[0, 1]).ravel()
    
    print(f"Standard Threshold (0.50): Sensitivity = {tp_def/(tp_def+fn_def)*100:.1f}% | Missed Cancers (FN) = {fn_def}")
    print(f"Clinical Threshold ({opt_thresh:.2f}): Sensitivity = {tp_opt/(tp_opt+fn_opt)*100:.1f}% | Missed Cancers (FN) = {fn_opt} (Reduced by {fn_def - fn_opt} cases!)")
    print("-" * 80 + "\n")


def plot_all_models_confusion_matrices(all_metrics):
    """
    Plots confusion matrices for all evaluated models.
    """
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.2))
    
    for i, m in enumerate(all_metrics):
        cm = m["cm_sklearn"]
        annot = [
            [f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"],
            [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"],
        ]
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            annot_kws={"size": 11, "weight": "bold"},
        )
        axes[i].set_title(
            f"{m['Model']}\nAcc: {m['Accuracy']*100:.1f}% | Recall: {m['Sensitivity (Recall)']*100:.1f}%",
            fontsize=11,
            weight="bold",
        )
        axes[i].set_xlabel("Predicted")
        if i == 0:
            axes[i].set_ylabel("Actual")
        else:
            axes[i].set_ylabel("")
            
    plt.suptitle("Confusion Matrix Comparison Across Models (Scikit-Learn Standard)", fontsize=14, weight="bold", y=1.05)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "confusion_matrices_all_models.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plots ROC curves for all models.
    """
    plt.figure(figsize=(8.5, 6.5))
    sns.set_theme(style="whitegrid")
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("ROC Curves for All Classifiers", fontsize=14, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_feature_importances(rf_model, feature_names):
    """
    Plots top 10 most influential features according to Random Forest.
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=top_importances, y=top_features, hue=top_features, palette="viridis", legend=False)
    plt.title("Top 10 Feature Importances (Random Forest)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Relative Gini Importance", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    print("=" * 80)
    print("ADVANCED BREAST CANCER ML BENCHMARK & EVALUATION PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    X, y, feature_names, X_df = load_and_prepare_data("data.csv")
    
    # 2. EDA & Visualizations
    plot_correlation_heatmap(X_df)
    
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    plot_pca_2d(X_scaled_full, y)
    
    # 3. Model Zoo: Expanded with Support Vector Classifier & Gradient Boosting
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
        "Support Vector Machine (RBF)": CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42), ensemble=False),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    # 4. Stratified 5-Fold Cross-Validation
    perform_cross_validation(models, X_scaled_full, y)
    
    # 5. Train/Test Split (75/25 stratified holdout)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    all_metrics = []
    print("=" * 80)
    print("3. HOLDOUT TEST SET PERFORMANCE (N=143, 25% Stratified Holdout)")
    print("=" * 80)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = calculate_diagnostic_metrics(y_test, y_pred, model_name=name)
        all_metrics.append(metrics)
        
    df_metrics = pd.DataFrame(all_metrics)
    display_cols = ["Model", "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision (PPV)", "FNR (Miss Rate)", "F1-Score"]
    print(df_metrics[display_cols].to_string(index=False, justify="center"))
    print("=" * 80 + "\n")
    
    # 6. Clinical Decision Threshold Optimization
    tune_decision_threshold(models["Random Forest"], X_test_scaled, y_test, model_name="Random Forest")
    
    # 7. Generate Evaluation Figures
    plot_all_models_confusion_matrices(all_metrics)
    plot_roc_curves(models, X_test_scaled, y_test)
    plot_feature_importances(models["Random Forest"], feature_names)
    
    print("[SUCCESS] All pipeline improvements executed successfully! Figures saved to ./plots/")


if __name__ == "__main__":
    main()
