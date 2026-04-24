import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib

def main():
    print("Loading datasets...")
    df_train = pd.read_csv('processed_data/train_dataset.csv')
    
    feature_cols = ['max_amp', 'mean_mag', 'var_mag', 'spectral_energy', 
                    'spectral_centroid', 'spectral_spread', 'peak_f1', 'peak_f2', 'peak_f3']
    
    # -------------------------------------------------------------
    # Unsupervised Fault Assignment (K-Means)
    # -------------------------------------------------------------
    faulty_mask = df_train['label'] == 1
    X_faulty = df_train.loc[faulty_mask, feature_cols]
    
    scaler = StandardScaler()
    X_faulty_scaled = scaler.fit_transform(X_faulty)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_faulty_scaled)
    
    # Mapping based on size/volatility rationale (simulated)
    # 1: Inner Race, 2: Outer Race, 3: Combination
    cluster_mapping = {
        pd.Series(clusters).value_counts().index[0]: 1, # Inner Race
        pd.Series(clusters).value_counts().index[1]: 2, # Outer Race
        pd.Series(clusters).value_counts().index[2]: 3  # Combination
    }
    
    # Replace the binary "1" with specific defect classes
    df_train.loc[faulty_mask, 'label'] = [cluster_mapping[c] for c in clusters]
    
    # Generate PCA Plot for User
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_faulty_scaled)
    mapped_clusters = [cluster_mapping[c] for c in clusters]
    
    plt.figure(figsize=(8, 6))
    label_names = {1: 'Inner Race Defect', 2: 'Outer Race Defect', 3: 'Combination Defect'}
    str_labels = [label_names[l] for l in mapped_clusters]
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=str_labels, palette='viridis', s=100)
    plt.title('Fault Sub-Classes Segmented via K-Means (PCA) - XGBoost Run')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Defect Variant')
    plt.savefig('processed_data/fault_clusters_multiclass_xgb.png')
    plt.close()

    # -------------------------------------------------------------
    # Cross-Val Training & SMOTE Validation
    # -------------------------------------------------------------
    X = df_train[feature_cols]
    y = df_train['label']
    print(f"\nFinal Training Class distribution:\n{y.value_counts()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # XGBoost setup
    xgb_model = XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        eval_metric='mlogloss',
    )

    xgb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=1)), 
        ('xgb', xgb_model)
    ])

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(xgb_pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    print("\n--- 5-Fold Stratified Cross Validation (XGBoost + SMOTE) ---")
    print(f"Mean CV Accuracy:  {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Mean CV Precision: {np.mean(cv_results['test_precision_macro']):.4f}")
    print(f"Mean CV Recall:    {np.mean(cv_results['test_recall_macro']):.4f}")
    print(f"Mean CV F1-Score:  {np.mean(cv_results['test_f1_macro']):.4f}")

    # -------------------------------------------------------------
    # Final Balanced Training for ROC-AUC & Confusion Matrix Output
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    final_xgb = XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        eval_metric='mlogloss',
    )
    final_xgb.fit(X_train_res, y_train_res)

    test_preds = final_xgb.predict(X_test)
    test_probs = final_xgb.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, test_probs, multi_class='ovr')

    print(f"\nModel ROC-AUC Score (OVR): {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    actual_label_names = ['Healthy', 'Inner Race', 'Outer Race', 'Combination']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actual_label_names, yticklabels=actual_label_names)
    plt.ylabel('Actual Classification')
    plt.xlabel('Predicted Classification')
    plt.title('Multi-Class XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.savefig('processed_data/multiclass_xgboost_cm.png')
    plt.close()

    joblib.dump(final_xgb, 'processed_data/final_xgboost_model.pkl')
    print("Saved XGBoost Model to processed_data/final_xgboost_model.pkl")

if __name__ == "__main__":
    main()
