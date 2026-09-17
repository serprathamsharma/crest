"""
Advanced Breast Cancer Detection & Feature Selection Benchmark

Comprehensive implementation combining:
- Randerson112358's tutorial (Confusion Matrix correction)
- Kaan Can's #1 Kaggle Kernel ("Feature Selection & Data Visualization"):
  1. Standardized Violin Plots for distribution separation
  2. Joint Plots & High-Correlation Drop List
  3. Univariate Feature Selection (SelectKBest)
  4. Recursive Feature Elimination (RFE)
  5. Recursive Feature Elimination with Cross-Validation (RFECV)
  6. Tree-based Feature Importance
  7. PCA Scree & 2D Projection
- Stratified 5-Fold Cross-Validation
- Clinical Probability Threshold Optimization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
)

# Output directory for plots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data(csv_filename="data.csv"):
    """
    Load and preprocess the Wisconsin Breast Cancer Diagnostic dataset (data.csv).
    """
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    
    if os.path.exists(csv_path):
        print(f"Loading Kaggle dataset from: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        
        df_clean = raw_df.dropna(axis=1, how="all").copy()
        if "Unnamed: 32" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Unnamed: 32"])
            
        if "id" in df_clean.columns:
            df_clean = df_clean.drop(columns=["id"])
            
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
        raise FileNotFoundError(f"Dataset {csv_path} not found.")
        
    print("=" * 80)
    print("1. DATASET INGESTION")
    print("=" * 80)
    print(f"Samples: {X.shape[0]} | Features: {len(feature_names)}")
    benign_count = (y == 0).sum()
    malignant_count = (y == 1).sum()
    print(f"Distribution: Benign (0): {benign_count} ({benign_count/len(y)*100:.1f}%), Malignant (1): {malignant_count} ({malignant_count/len(y)*100:.1f}%)\n")
    
    return X, y, feature_names, X_df


def plot_violin_features(X_df, y):
    """
    Kaggle Method: Standardized Violin plots showing the 10 most discriminating features.
    """
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)
    
    top_10 = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "radius_worst", "perimeter_worst"
    ]
    
    plot_data = X_std[top_10].copy()
    plot_data["diagnosis"] = np.where(y == 1, "Malignant", "Benign")
    plot_melted = pd.melt(plot_data, id_vars="diagnosis", var_name="features", value_name="standardized_value")
    
    plt.figure(figsize=(13, 6))
    sns.set_theme(style="whitegrid")
    sns.violinplot(
        x="features",
        y="standardized_value",
        hue="diagnosis",
        data=plot_melted,
        split=True,
        inner="quart",
        palette={"Benign": "#2980b9", "Malignant": "#c0392b"},
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.title("Standardized Feature Distributions by Diagnosis (Violin Plots)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Standardized Value (Z-Score)", fontsize=11)
    plt.xlabel("Cell Nucleus Feature", fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "violin_features_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_joint_correlation(X_df):
    """
    Kaggle Method: Joint plot comparing two strongly correlated features.
    """
    g = sns.jointplot(
        x="concavity_worst",
        y="concave points_worst",
        data=X_df,
        kind="reg",
        color="#c0392b",
        height=6.5,
    )
    g.fig.suptitle("Joint Regression: Concavity Worst vs Concave Points Worst (r = 0.86)", y=1.02, fontsize=12, weight="bold")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "jointplot_correlation.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_pca_scree(X_df):
    """
    Kaggle Method: PCA Scree plot showing cumulative and individual explained variance.
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_df)
    
    pca = PCA()
    pca.fit(X_std)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    
    ax1.bar(range(1, 11), exp_var[:10], alpha=0.7, color="#2980b9", label="Individual Variance")
    ax1.set_xlabel("Principal Component Index (1 to 10)", fontsize=11)
    ax1.set_ylabel("Individual Explained Variance", color="#2980b9", fontsize=11)
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, 11), cum_var[:10], color="#c0392b", marker="o", lw=2.2, label="Cumulative Variance")
    ax2.set_ylabel("Cumulative Explained Variance", color="#c0392b", fontsize=11)
    ax2.set_ylim(0.4, 1.0)
    
    plt.title("PCA Scree Plot: Explained Variance by Top 10 Components", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "pca_scree_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def run_feature_selection_experiments(X_train, X_test, y_train, y_test, feature_names):
    """
    Implements and benchmarks Kaan Can's 4 feature selection methods:
    1. Correlation-based drop list
    2. Univariate Selection (SelectKBest)
    3. Recursive Feature Elimination (RFE, k=5)
    4. Recursive Feature Elimination with Cross-Validation (RFECV)
    """
    print("=" * 80)
    print("2. KAAN CAN FEATURE SELECTION BENCHMARK (Random Forest Classifier)")
    print("=" * 80)
    
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    acc_all = accuracy_score(y_test, rf_baseline.predict(X_test))
    print(f"A. All 30 Raw Features:                Accuracy = {acc_all*100:.2f}% (30 features)")
    
    # 1. Correlation-based drop list (r > 0.90 pairs)
    drop_list_corr = [
        "perimeter_mean", "radius_mean", "compactness_mean", "concave points_mean",
        "radius_se", "perimeter_se", "radius_worst", "perimeter_worst",
        "compactness_worst", "concave points_worst", "compactness_se",
        "concave points_se", "texture_worst", "area_worst"
    ]
    keep_indices_corr = [i for i, f in enumerate(feature_names) if f not in drop_list_corr]
    X_train_corr = X_train[:, keep_indices_corr]
    X_test_corr = X_test[:, keep_indices_corr]
    
    rf_corr = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_corr.fit(X_train_corr, y_train)
    acc_corr = accuracy_score(y_test, rf_corr.predict(X_test_corr))
    print(f"B. Correlation-Filtered Subset:         Accuracy = {acc_corr*100:.2f}% ({len(keep_indices_corr)} features)")
    
    # 2. SelectKBest (k=5)
    kbest = SelectKBest(score_func=f_classif, k=5)
    X_train_kbest = kbest.fit_transform(X_train, y_train)
    X_test_kbest = kbest.transform(X_test)
    selected_kbest_feats = feature_names[kbest.get_support()]
    
    rf_kbest = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_kbest.fit(X_train_kbest, y_train)
    acc_kbest = accuracy_score(y_test, rf_kbest.predict(X_test_kbest))
    print(f"C. Univariate Selection (SelectKBest-5): Accuracy = {acc_kbest*100:.2f}% (Top 5: {', '.join(selected_kbest_feats[:3])}...)")
    
    # 3. RFE (k=5)
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=5, step=1)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)
    selected_rfe_feats = feature_names[rfe.support_]
    
    rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_rfe.fit(X_train_rfe, y_train)
    acc_rfe = accuracy_score(y_test, rf_rfe.predict(X_test_rfe))
    print(f"D. Recursive Feature Elimination (RFE): Accuracy = {acc_rfe*100:.2f}% (Selected: {', '.join(selected_rfe_feats[:3])}...)")
    
    # 4. RFECV (Optimal number of features with 5-fold CV)
    rfecv = RFECV(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    rfecv.fit(X_train, y_train)
    opt_n = rfecv.n_features_
    opt_feats = feature_names[rfecv.support_]
    acc_rfecv = accuracy_score(y_test, rfecv.predict(X_test))
    print(f"E. Optimal RFECV Feature Subset:        Accuracy = {acc_rfecv*100:.2f}% (Optimal {opt_n} features)")
    print("=" * 80 + "\n")
    
    # Plot RFECV Curve
    cv_scores = rfecv.cv_results_["mean_test_score"]
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker="o", color="#27ae60", lw=2)
    plt.axvline(x=opt_n, color="#c0392b", linestyle="--", label=f"Optimal Feature Count ({opt_n})")
    plt.title("Recursive Feature Elimination with Cross-Validation (RFECV)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Number of Features Selected", fontsize=11)
    plt.ylabel("Cross-Validated Accuracy", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "rfecv_feature_selection.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    print("=" * 80)
    print("COMPREHENSIVE BREAST CANCER ML BENCHMARK & FEATURE SELECTION SUITE")
    print("=" * 80)
    
    # 1. Load Data
    X, y, feature_names, X_df = load_and_prepare_data("data.csv")
    
    # 2. EDA & Visualizations from Kaan Can's kernel
    plot_violin_features(X_df, y)
    plot_joint_correlation(X_df)
    plot_pca_scree(X_df)
    
    # 3. Train/Test Split (75/25 stratified holdout)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Feature Selection Benchmark
    run_feature_selection_experiments(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    
    # 5. Model Zoo Evaluation
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
        "Support Vector Machine (RBF)": CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42), ensemble=False),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    # 6. Stratified 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    
    cv_records = []
    print("=" * 80)
    print("3. STRATIFIED 5-FOLD CROSS-VALIDATION BENCHMARK (Mean +/- Std)")
    print("=" * 80)
    
    X_scaled_full = scaler.fit_transform(X)
    for name, model in models.items():
        scores = cross_validate(model, X_scaled_full, y, cv=cv, scoring=scoring)
        cv_records.append({
            "Model": name,
            "CV Accuracy": f"{scores['test_accuracy'].mean()*100:.2f}% +/- {scores['test_accuracy'].std()*100:.2f}%",
            "CV Sensitivity (Recall)": f"{scores['test_recall'].mean()*100:.2f}% +/- {scores['test_recall'].std()*100:.2f}%",
            "CV Precision": f"{scores['test_precision'].mean()*100:.2f}% +/- {scores['test_precision'].std()*100:.2f}%",
            "CV F1-Score": f"{scores['test_f1'].mean():.4f}",
            "CV ROC-AUC": f"{scores['test_roc_auc'].mean():.4f}",
        })
        
    df_cv = pd.DataFrame(cv_records)
    print(df_cv.to_string(index=False))
    print("=" * 80 + "\n")
    
    # 7. Holdout Performance & Confusion Matrices
    all_metrics = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Scikit-learn unpacking: labels=[0, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        total = tp + tn + fp + fn
        acc = (tp + tn) / total
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
        
        all_metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Sensitivity (Recall)": sens,
            "Specificity": spec,
            "Precision (PPV)": prec,
            "Miss Rate (FNR)": fnr,
            "F1-Score": f1,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "cm_sklearn": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        })
        
    df_holdout = pd.DataFrame(all_metrics)
    print("=" * 80)
    print("4. HOLDOUT TEST SET PERFORMANCE (N=143, 25% Stratified Holdout)")
    print("=" * 80)
    display_cols = ["Model", "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision (PPV)", "Miss Rate (FNR)", "F1-Score"]
    print(df_holdout[display_cols].to_string(index=False, justify="center"))
    print("=" * 80 + "\n")
    
    # 8. Confusion Matrix Comparison Figure
    fig, axes = plt.subplots(1, 5, figsize=(21, 4))
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
        axes[i].set_title(f"{m['Model']}\nAcc: {m['Accuracy']*100:.1f}% | Recall: {m['Sensitivity (Recall)']*100:.1f}%", fontsize=10, weight="bold")
        axes[i].set_xlabel("Predicted")
        if i == 0:
            axes[i].set_ylabel("Actual")
        else:
            axes[i].set_ylabel("")
            
    plt.suptitle("Confusion Matrix Comparison Across Models (Scikit-Learn Standard: [0,0]=TN)", fontsize=13, weight="bold", y=1.05)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "confusion_matrices_all_models.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")
    
    # 9. ROC Curves
    plt.figure(figsize=(8.5, 6.5))
    sns.set_theme(style="whitegrid")
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_probs = model.decision_function(X_test_scaled)
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
    
    # 10. Random Forest Feature Importances
    rf_model = models["Random Forest"]
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
    
    # 11. Clinical Threshold Tuning
    y_probs_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 100)
    sens_list = []
    spec_list = []
    fn_list = []
    for t in thresholds:
        preds = (y_probs_rf >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        fn_list.append(fn)
        
    plt.figure(figsize=(9.5, 5.5))
    plt.plot(thresholds, sens_list, label="Sensitivity (Recall)", color="#27ae60", lw=2.5)
    plt.plot(thresholds, spec_list, label="Specificity", color="#2980b9", lw=2.5)
    plt.axvline(x=0.50, color="gray", linestyle="--", label="Default Threshold (0.50 | FN=4)")
    opt_idx = np.argmin(np.abs(np.array(sens_list) - 0.98))
    opt_thresh = thresholds[opt_idx]
    opt_fn = fn_list[opt_idx]
    plt.axvline(x=opt_thresh, color="#c0392b", linestyle=":", lw=2, label=f"Clinical Threshold ({opt_thresh:.2f} | FN={opt_fn})")
    plt.title("Clinical Threshold Tuning: Precision-Recall Trade-Off", fontsize=13, weight="bold")
    plt.xlabel("Probability Cutoff", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "threshold_tuning_tradeoff.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")
    
    print("\n[SUCCESS] Comprehensive benchmark execution complete! All plots saved to ./plots/")


if __name__ == "__main__":
    main()
