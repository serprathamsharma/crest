"""
Breast Cancer Detection using Machine Learning & Confusion Matrix Analysis.

Based on:
- Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
- Randerson112358's Tutorial & Video (Medium / YouTube: NSSOyhJBmWY)
- Wikipedia Confusion Matrix Mathematical Framework

Key Focus:
Resolving the common Scikit-Learn vs. Wikipedia / Diagnostic Confusion Matrix axis convention discrepancy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)

# Output directory for plots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data(csv_filename="data.csv"):
    """
    Load and preprocess the Wisconsin Breast Cancer Diagnostic dataset.
    Prioritizes the official Kaggle dataset CSV (data.csv), exactly as used
    in Randerson112358's tutorial and the Kaggle dataset repository:
    https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
    
    Preprocessing steps:
      1. Load raw CSV (569 rows x 33 columns).
      2. Drop empty trailing column 'Unnamed: 32'.
      3. Drop patient identifier 'id'.
      4. Encode diagnosis: 'M' (Malignant) -> 1, 'B' (Benign) -> 0.
    """
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    
    if os.path.exists(csv_path):
        print(f"Loading Kaggle dataset from: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        print(f"Raw CSV Shape: {raw_df.shape} (includes id & Unnamed: 32)")
        
        # 1. Drop Unnamed: 32 if present (common artifact of trailing commas in Kaggle CSV)
        df_clean = raw_df.dropna(axis=1, how="all")
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
            raise ValueError("Target column ('diagnosis' or 'target') not found in CSV.")
            
        X = X_df.values
        feature_names = np.array(X_df.columns)
    else:
        print("data.csv not found locally. Loading from scikit-learn dataset...")
        raw_data = load_breast_cancer()
        # Remap: 1 = Malignant (Condition Positive), 0 = Benign (Condition Negative)
        y = np.where(raw_data.target == 0, 1, 0)
        X = raw_data.data
        feature_names = raw_data.feature_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df["diagnosis"] = np.where(y == 1, "M", "B")
    df["target"] = y
    
    print("=" * 80)
    print("1. DATASET OVERVIEW (Wisconsin Breast Cancer Diagnostic - Kaggle UCIML)")
    print("=" * 80)
    print(f"Cleaned Samples: {df.shape[0]}")
    print(f"Diagnostic Features: {len(feature_names)}")
    benign_count = (y == 0).sum()
    malignant_count = (y == 1).sum()
    print(f"Class Distribution: Benign (0 / 'B'): {benign_count} ({benign_count/len(y)*100:.1f}%), Malignant (1 / 'M'): {malignant_count} ({malignant_count/len(y)*100:.1f}%)")
    print(f"Cleaned feature list: {', '.join(feature_names[:6])} ...\n")
    
    return X, y, feature_names


def calculate_diagnostic_metrics(y_true, y_pred, model_name="Model"):
    """
    Extracts confusion matrix and computes all diagnostic metrics,
    explaining both Scikit-Learn and Wikipedia layouts.
    """
    # 1. Scikit-Learn default layout: labels=[0, 1]
    # cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    cm_sklearn = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_sklearn.ravel()
    
    # 2. Wikipedia layout: labels=[1, 0] (Positive first)
    # cm[0,0]=TP, cm[0,1]=FN, cm[1,0]=FP, cm[1,1]=TN
    cm_wiki = confusion_matrix(y_true, y_pred, labels=[1, 0])
    
    # Calculate performance metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Selectivity / TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # NPV
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0          # Fall-out (1 - Specificity)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0          # Miss rate (1 - Sensitivity)
    f1 = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )
    
    metrics = {
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
        "FPR": fpr,
        "FNR (Miss Rate)": fnr,
        "F1-Score": f1,
        "cm_sklearn": cm_sklearn,
        "cm_wiki": cm_wiki,
    }
    return metrics


def print_confusion_matrix_deepdive(metrics):
    """
    Demonstrates the difference between Scikit-learn and Wikipedia formats
    and highlights the Randerson112358 tutorial trap.
    """
    cm_sk = metrics["cm_sklearn"]
    cm_wk = metrics["cm_wiki"]
    tp, tn, fp, fn = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]
    
    print("-" * 80)
    print(f"CONFUSION MATRIX ANALYSIS FOR {metrics['Model'].upper()}")
    print("-" * 80)
    print("A. Scikit-Learn Default Format (labels=[0, 1] -> Benign=0, Malignant=1):")
    print("                 Predicted: Benign (0)   Predicted: Malignant (1)")
    print(f"Actual: Benign (0)       TN = {cm_sk[0,0]:<3}              FP = {cm_sk[0,1]:<3}")
    print(f"Actual: Malignant (1)    FN = {cm_sk[1,0]:<3}              TP = {cm_sk[1,1]:<3}")
    print()
    print("B. Wikipedia / Medical Literature Standard (Positive condition first):")
    print("                 Predicted: Malignant (1) Predicted: Benign (0)")
    print(f"Actual: Malignant (1)    TP = {cm_wk[0,0]:<3}              FN = {cm_wk[0,1]:<3}")
    print(f"Actual: Benign (0)       FP = {cm_wk[1,0]:<3}              TN = {cm_wk[1,1]:<3}")
    print()
    print(">>> CRITICAL LESSON (Randerson112358 Tutorial Pitfall):")
    print(f"    - Tutorial original code: TP = cm[0][0] ({cm_sk[0,0]}) -> INCORRECT (Actually TN!)")
    print(f"                              TN = cm[1][1] ({cm_sk[1,1]}) -> INCORRECT (Actually TP!)")
    print(f"    - Correct scikit-learn unpacking: tn, fp, fn, tp = cm.ravel()")
    print(f"      True Negatives (TN): {tn} (Benign correctly identified)")
    print(f"      False Positives (FP): {fp} (Benign mistakenly called Malignant)")
    print(f"      False Negatives (FN): {fn} (Malignant missed! Dangerous in oncology)")
    print(f"      True Positives (TP): {tp} (Malignant correctly identified)")
    print()
    print(f"Metrics Summary for {metrics['Model']}:")
    print(f"  - Accuracy:              {metrics['Accuracy']*100:.2f}%")
    print(f"  - Sensitivity (Recall):  {metrics['Sensitivity (Recall)']*100:.2f}% (Detection rate of cancer)")
    print(f"  - Specificity:           {metrics['Specificity']*100:.2f}% (Avoidance of false alarm)")
    print(f"  - Precision:             {metrics['Precision (PPV)']*100:.2f}%")
    print(f"  - False Negative Rate:   {metrics['FNR (Miss Rate)']*100:.2f}%")
    print(f"  - F1-Score:              {metrics['F1-Score']:.4f}")
    print()


def plot_side_by_side_confusion_matrices(metrics_rf):
    """
    Plots Scikit-Learn convention vs Wikipedia convention side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Scikit-Learn layout
    cm_sk = metrics_rf["cm_sklearn"]
    labels_sk = [
        [f"TN\n{cm_sk[0,0]}\n({cm_sk[0,0]/cm_sk.sum():.1%})", f"FP\n{cm_sk[0,1]}\n({cm_sk[0,1]/cm_sk.sum():.1%})"],
        [f"FN\n{cm_sk[1,0]}\n({cm_sk[1,0]/cm_sk.sum():.1%})", f"TP\n{cm_sk[1,1]}\n({cm_sk[1,1]/cm_sk.sum():.1%})"],
    ]
    sns.heatmap(
        cm_sk,
        annot=labels_sk,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Predicted Benign (0)", "Predicted Malignant (1)"],
        yticklabels=["Actual Benign (0)", "Actual Malignant (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title(
        "Scikit-Learn Default Convention\n(labels=[0, 1] | Row 0 = Negative)",
        fontsize=13,
        pad=12,
        weight="bold",
    )
    
    # 2. Wikipedia / Medical layout
    cm_wk = metrics_rf["cm_wiki"]
    labels_wk = [
        [f"TP\n{cm_wk[0,0]}\n({cm_wk[0,0]/cm_wk.sum():.1%})", f"FN\n{cm_wk[0,1]}\n({cm_wk[0,1]/cm_wk.sum():.1%})"],
        [f"FP\n{cm_wk[1,0]}\n({cm_wk[1,0]/cm_wk.sum():.1%})", f"TN\n{cm_wk[1,1]}\n({cm_wk[1,1]/cm_wk.sum():.1%})"],
    ]
    sns.heatmap(
        cm_wk,
        annot=labels_wk,
        fmt="",
        cmap="Purples",
        cbar=False,
        ax=axes[1],
        xticklabels=["Predicted Malignant (1)", "Predicted Benign (0)"],
        yticklabels=["Actual Malignant (1)", "Actual Benign (0)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[1].set_title(
        "Wikipedia / Medical Diagnostic Convention\n(labels=[1, 0] | Row 0 = Positive)",
        fontsize=13,
        pad=12,
        weight="bold",
    )
    
    plt.suptitle(
        f"Confusion Matrix Orientation Comparison — {metrics_rf['Model']}",
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "confusion_matrices_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_all_models_confusion_matrices(all_metrics):
    """
    Plots confusion matrices for all evaluated models side-by-side.
    """
    fig, axes = plt.subplots(1, len(all_metrics), figsize=(5.5 * len(all_metrics), 4.8))
    
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
            xticklabels=["Benign (0)", "Malignant (1)"],
            yticklabels=["Benign (0)", "Malignant (1)"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        axes[i].set_title(
            f"{m['Model']}\nAcc: {m['Accuracy']*100:.1f}% | Recall: {m['Sensitivity (Recall)']*100:.1f}%",
            fontsize=12,
            weight="bold",
        )
        axes[i].set_xlabel("Predicted Label")
        if i == 0:
            axes[i].set_ylabel("True Label")
        else:
            axes[i].set_ylabel("")
            
    plt.suptitle("Confusion Matrix Comparison Across Models (Scikit-Learn Standard)", fontsize=14, weight="bold")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "confusion_matrices_all_models.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_performance_metrics(df_metrics):
    """
    Plots a multi-metric bar chart comparing all models.
    """
    plot_df = df_metrics.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Sensitivity (Recall)", "Specificity", "Precision (PPV)", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )
    
    plt.figure(figsize=(11, 5.5))
    sns.set_theme(style="whitegrid")
    models_list = plot_df["Model"].unique()
    palette = sns.color_palette("muted", n_colors=len(models_list))
    
    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette=palette)
    plt.title("Model Performance Metrics Comparison", fontsize=15, weight="bold", pad=15)
    plt.ylim(0.80, 1.02)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.legend(title="Model", loc="lower right", frameon=True)
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )
            
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "models_performance_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plots Receiver Operating Characteristic (ROC) curves with AUC scores.
    """
    plt.figure(figsize=(8, 6.5))
    sns.set_theme(style="whitegrid")
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("ROC Curves for Breast Cancer Classification", fontsize=14, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
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
    print("BREAST CANCER DETECTION & CONFUSION MATRIX PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    X, y, feature_names = load_and_prepare_data()
    
    # 2. Strict featurization ordering (ml-best-practices):
    # Split BEFORE fitting StandardScaler to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples (25% stratified holdout)")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train Models as in Randerson112358's tutorial
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
    }
    
    all_metrics = []
    
    print("\n" + "=" * 80)
    print("2. MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    for name, model in models.items():
        # Fit on scaled training data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = calculate_diagnostic_metrics(y_test, y_pred, model_name=name)
        all_metrics.append(metrics)
        print_confusion_matrix_deepdive(metrics)
        
    df_metrics = pd.DataFrame(all_metrics)
    
    # Print Summary Table
    print("=" * 80)
    print("3. COMPARATIVE BENCHMARK TABLE")
    print("=" * 80)
    display_cols = [
        "Model",
        "Accuracy",
        "Sensitivity (Recall)",
        "Specificity",
        "Precision (PPV)",
        "FNR (Miss Rate)",
        "F1-Score",
    ]
    print(df_metrics[display_cols].to_string(index=False, justify="center"))
    print("=" * 80)
    
    # 4. Generate Visualizations
    print("\n4. GENERATING VISUALIZATIONS...")
    rf_metrics = [m for m in all_metrics if m["Model"] == "Random Forest"][0]
    plot_side_by_side_confusion_matrices(rf_metrics)
    plot_all_models_confusion_matrices(all_metrics)
    plot_performance_metrics(df_metrics)
    plot_roc_curves(models, X_test_scaled, y_test)
    plot_feature_importances(models["Random Forest"], feature_names)
    
    print("\n[SUCCESS] Pipeline execution complete! All plots saved in ./plots/")


if __name__ == "__main__":
    main()
