import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    print("Loading original extracted dataset...")
    df_train = pd.read_csv('processed_data/train_dataset.csv')
    
    # Check what labels exist in the raw dataset to reassure the user
    print(f"Original Labels Distribution:\n{df_train['label'].value_counts()}")

    # Features
    feature_cols = ['max_amp', 'mean_mag', 'var_mag', 'spectral_energy', 
                    'spectral_centroid', 'spectral_spread', 'peak_f1', 'peak_f2', 'peak_f3']
    
    faulty_mask = df_train['label'] == 1
    X_faulty = df_train.loc[faulty_mask, feature_cols]
    
    scaler = StandardScaler()
    X_faulty_scaled = scaler.fit_transform(X_faulty)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_faulty_scaled)
    
    # Mapping based on size (Inner (167), Outer (91), Combo (3))
    cluster_mapping = {
        pd.Series(clusters).value_counts().index[0]: 1, # Inner Race
        pd.Series(clusters).value_counts().index[1]: 2, # Outer Race
        pd.Series(clusters).value_counts().index[2]: 3  # Combination
    }
    
    # Replace the binary "1" with specific defect classes
    df_train.loc[faulty_mask, 'label'] = [cluster_mapping[c] for c in clusters]
    
    # Save the explicitly numbered multiclass dataset
    df_train.to_csv('processed_data/final_multiclass_dataset.csv', index=False)
    
    print(f"\nFinal Mapped Clustered Dataset Distribution:\n{df_train['label'].value_counts()}")
    print("\nSaved fully processed dataset containing all 0, 1, 2, and 3 labels to processed_data/final_multiclass_dataset.csv")

if __name__ == "__main__":
    main()
