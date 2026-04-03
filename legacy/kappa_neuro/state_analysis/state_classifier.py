"""
STATE CLASSIFIER - Katashi EEG State Classification
====================================================
Trains and evaluates classifiers for structural state discrimination.
Implements both unsupervised (clustering) and supervised (RF) approaches.

Author: Claude + David
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import HDBSCAN
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Get script directory
SCRIPT_DIR = Path(__file__).parent

def load_mapped_data():
    """Load state-mapped dataset"""
    print("Loading mapped data...")
    path = SCRIPT_DIR / 'states_mapped.csv'
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} points with {df['state'].nunique()} states")
    return df

def prepare_features(df):
    """Prepare feature matrix and labels"""
    print("\nPreparing features...")
    
    # Feature columns
    feature_cols = ['Xi', 'Oh', 'dominance', 'entropy', 'phi',
                   'Xi_slope', 'Oh_slope', 'dominance_slope',
                   'Xi_std_10', 'Oh_std_10']
    
    # Check which features exist
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"  Using {len(available_features)} features: {available_features}")
    
    X = df[available_features].values
    y = df['state'].values
    
    # Remove samples with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"  Valid samples: {len(X)} ({valid_mask.sum()/len(valid_mask)*100:.1f}%)")
    
    return X, y, available_features

def unsupervised_clustering(df):
    """Unsupervised clustering to validate state separation"""
    print("\n" + "="*60)
    print("UNSUPERVISED CLUSTERING")
    print("="*60)
    output_path = SCRIPT_DIR / 'clustering_results.txt'
    
    X, y_true, features = prepare_features(df)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # HDBSCAN clustering
    print("\nRunning HDBSCAN...")
    clusterer = HDBSCAN(min_cluster_size=50, min_samples=10)
    y_pred = clusterer.fit_predict(X_scaled)
    
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    
    print(f"  Found {n_clusters} clusters")
    print(f"  Noise points: {n_noise} ({n_noise/len(y_pred)*100:.1f}%)")
    
    # Compare with true states
    results = []
    results.append("CLUSTERING VALIDATION\n")
    results.append("="*60 + "\n\n")
    results.append(f"Clusters found: {n_clusters}\n")
    results.append(f"Noise points: {n_noise} ({n_noise/len(y_pred)*100:.1f}%)\n\n")
    
    results.append("True State Distribution in Clusters:\n")
    results.append("-"*60 + "\n")
    
    for cluster_id in sorted(set(y_pred)):
        if cluster_id == -1:
            continue
        mask = y_pred == cluster_id
        true_states = y_true[mask]
        results.append(f"\nCluster {cluster_id} (n={mask.sum()}):\n")
        for state in np.unique(true_states):
            count = (true_states == state).sum()
            pct = count / len(true_states) * 100
            results.append(f"  {state}: {count} ({pct:.1f}%)\n")
    
    # Save results
    with open(output_path, 'w') as f:
        f.writelines(results)
    print(f"\n  Saved clustering results to {output_path}")
    
    return y_pred

def train_classifier(df):
    """Train supervised classifier"""
    print("\n" + "="*60)
    print("SUPERVISED CLASSIFICATION")
    print("="*60)
    output_dir = SCRIPT_DIR
    
    X, y, features = prepare_features(df)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                 random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Test set evaluation
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Save report
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n\n")
        f.write(report)
    
    # Feature importance
    plot_feature_importance(clf, features, f'{output_dir}/fig_feature_importance.png')
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, clf.classes_, 
                         f'{output_dir}/fig_confusion_matrix.png')
    
    # ROC curves (multi-class)
    plot_roc_curves(y_test, y_pred_proba, clf.classes_,
                   f'{output_dir}/fig_roc_curves.png')
    
    return clf, scaler, features

def plot_feature_importance(clf, features, output_path):
    """Plot feature importances"""
    print("\nPlotting feature importances...")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(features)), importances[indices])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance for State Classification', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot confusion matrix"""
    print("\nPlotting confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel('True State', fontsize=11)
    ax.set_xlabel('Predicted State', fontsize=11)
    ax.set_title('Normalized Confusion Matrix', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, classes, output_path):
    """Plot ROC curves for multi-class classification"""
    print("\nPlotting ROC curves...")
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC for each class
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        ax.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={auc:.3f})')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves - Multi-Class State Classification', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

def analyze_ictal_detection(df, clf, scaler, features):
    """Specific analysis of ictal state detection"""
    print("\n" + "="*60)
    print("ICTAL STATE DETECTION ANALYSIS")
    print("="*60)
    output_path = SCRIPT_DIR / 'ictal_detection.txt'
    
    X, y, _ = prepare_features(df)
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_pred_proba = clf.predict_proba(X_scaled)
    
    # Focus on ictal vs others
    ictal_mask_true = (y == 'ictal')
    ictal_mask_pred = (y_pred == 'ictal')
    
    # Metrics
    tp = ((y == 'ictal') & (y_pred == 'ictal')).sum()
    fn = ((y == 'ictal') & (y_pred != 'ictal')).sum()
    fp = ((y != 'ictal') & (y_pred == 'ictal')).sum()
    tn = ((y != 'ictal') & (y_pred != 'ictal')).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    results = []
    results.append("ICTAL DETECTION PERFORMANCE\n")
    results.append("="*60 + "\n\n")
    results.append(f"True Positives (TP):  {tp}\n")
    results.append(f"False Negatives (FN): {fn}\n")
    results.append(f"False Positives (FP): {fp}\n")
    results.append(f"True Negatives (TN):  {tn}\n\n")
    results.append(f"Sensitivity (Recall): {sensitivity:.1%}\n")
    results.append(f"Specificity:          {specificity:.1%}\n")
    results.append(f"Precision:            {precision:.1%}\n")
    
    # Per-seizure analysis
    results.append("\n" + "-"*60 + "\n")
    results.append("PER-SEIZURE DETECTION:\n")
    results.append("-"*60 + "\n")
    
    for seizure_id in df[df['state'] == 'ictal']['seizure_id'].unique():
        seizure_mask = (df['state'] == 'ictal') & (df['seizure_id'] == seizure_id)
        seizure_indices = np.where(seizure_mask.values)[0]
        
        if len(seizure_indices) == 0:
            continue
        
        # Predictions for this seizure
        seizure_pred = y_pred[seizure_indices]
        detected = (seizure_pred == 'ictal').sum()
        total = len(seizure_pred)
        pct = detected / total * 100
        
        results.append(f"\nSeizure {seizure_id + 1}:\n")
        results.append(f"  Detected: {detected}/{total} windows ({pct:.1f}%)\n")
    
    # Save
    with open(output_path, 'w') as f:
        f.writelines(results)
    print(f"\n  Saved ictal detection analysis to {output_path}")
    
    # Print key metrics
    print(f"\n  Sensitivity: {sensitivity:.1%}")
    print(f"  Specificity: {specificity:.1%}")
    print(f"  Precision: {precision:.1%}")

def main():
    print("=" * 60)
    print("STATE CLASSIFIER - Katashi EEG State Classification")
    print("=" * 60)
    
    # Load data
    df = load_mapped_data()
    
    # Unsupervised clustering (validation)
    unsupervised_clustering(df)
    
    # Supervised classification
    clf, scaler, features = train_classifier(df)
    
    # Ictal-specific analysis
    analyze_ictal_detection(df, clf, scaler, features)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated outputs in state_analysis/")

if __name__ == '__main__':
    main()
