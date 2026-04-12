# 🔧 Pump Fault Diagnosis — ML-Based Predictive Maintenance

An end-to-end machine learning pipeline for detecting and classifying bearing faults in industrial pumps using vibration signal analysis. The system moves beyond simple fault detection to identify **which specific component is failing** — Inner Race, Outer Race, or a Combination defect.

---

## 📌 Project Overview

Industrial pump failures are expensive. Traditional maintenance is either too early (wasteful) or too late (catastrophic). This project builds a predictive system that:

- Processes raw vibration sensor readings (time-domain & frequency-domain)
- Extracts meaningful spectral features via FFT
- Clusters faulty signals into specific defect types using unsupervised learning
- Trains a balanced multi-class classifier with 95.48% cross-validated accuracy
- Predicts fault type on new, unlabeled pump readings
Download the dataset from- https://drive.google.com/drive/folders/16ubfp8jvLLgMFg2zIjP5mbllaiw0Dpiz
---

## 🗂️ Project Structure

```
├── data/
│   ├── Healthy Reading/          # Frequency-domain healthy pump readings (.xlsx)
│   ├── Faulty readings/
│   │   └── NEW Time Domain/      # Time-domain faulty pump readings (.xlsx)
│   └── PUMP DIAGNOSIS/           # Unlabeled readings for live prediction (.xlsx)
│
├── processed_data/
│   ├── train_dataset.csv                   # Extracted features with labels
│   ├── diagnosis_dataset.csv               # Extracted features (unknown)
│   ├── diagnosis_balanced_predictions.csv  # Final predictions on unknown pumps
│   ├── balanced_rf_model.pkl               # Binary fault detection model
│   ├── final_multiclass_rf_model.pkl       # Multi-class fault type model
│   ├── fault_clusters_multiclass.png       # PCA cluster visualization
│   ├── multiclass_rf_cm.png                # Confusion matrix
│   └── balanced_feature_importances.png    # Feature importance chart
│
├── extract_features.py      # Step 1 — Raw signal → feature extraction
├── train_model.py           # Step 2a — Binary healthy/faulty classifier
└── train_multiclass.py      # Step 2b — Multi-class fault type classifier
```

---

## ⚙️ Pipeline Architecture

### Step 1 — Feature Extraction (`extract_features.py`)

Raw Excel files are parsed and converted into a 9-feature numerical vector per reading.

**Auto-detects signal domain:**
- Time-domain signals → FFT applied via `numpy.fft.rfft()` to expose frequency content
- Frequency-domain signals → used directly

**9 Extracted Features:**

| Feature | Description |
|---|---|
| `max_amp` | Peak magnitude in the frequency spectrum |
| `mean_mag` | Average spectral energy across all frequencies |
| `var_mag` | Variance — how much energy fluctuates |
| `spectral_energy` | Total signal power (sum of squared magnitudes) |
| `spectral_centroid` | Weighted center-of-mass of the spectrum |
| `spectral_spread` | Width of energy distribution around centroid |
| `peak_f1` | Dominant frequency (highest magnitude) |
| `peak_f2` | Second dominant frequency |
| `peak_f3` | Third dominant frequency |

---

### Step 2 — Unsupervised Fault Sub-Typing (K-Means)

The raw faulty data had no sub-labels — just "Faulty". K-Means clustering automatically discovered 3 physically meaningful defect categories:

```
Faulty Samples (261 total)
        │
        ▼
   K-Means (K=3)
        │
   ┌────┴──────────┐──────────────┐
   ▼               ▼              ▼
Inner Race      Outer Race    Combination
  (167)            (91)           (3)
```

Features were StandardScaler-normalized before clustering to prevent high-magnitude features dominating the distance calculations. PCA (2 components) was used purely for visualization.

---

### Step 3 — Class Balancing (SMOTE)

| Class | Original Count | After SMOTE |
|---|---|---|
| Healthy | 49 | Balanced ✅ |
| Inner Race | 167 | Baseline |
| Outer Race | 91 | Balanced ✅ |
| Combination | 3 | Balanced ✅ |

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples by interpolating between real minority-class samples and their k-nearest neighbors. `k_neighbors=1` was used for the Combination class due to its extremely small size.

> **Critical:** SMOTE was applied inside an `ImbPipeline` during cross-validation, ensuring synthetic samples never leaked into validation folds.

---

### Step 4 — Model Training (Random Forest)

**Why Random Forest over SVM?**

SVM relies on Euclidean distance — the massive scalar gap between vibration frequencies (hundreds of Hz) and amplitudes (micro-decimals) caused SVM to misclassify 100% of samples as Faulty. Random Forest makes sequential split decisions independent of feature scale, handling mixed-magnitude features natively.

```
RandomForestClassifier(
    n_estimators = 100,
    random_state = 42
)
```

---

### Step 5 — Evaluation (Stratified 5-Fold Cross Validation)

Each fold preserves the class ratio from the full dataset. Every sample is validated exactly once across 5 rounds.

```
Round 1: [■■■■] [□]  →  Validate on Fold 5
Round 2: [■■■□] [■]  →  Validate on Fold 4
Round 3: [■■□■] [■]  →  Validate on Fold 3
Round 4: [■□■■] [■]  →  Validate on Fold 2
Round 5: [□■■■] [■]  →  Validate on Fold 1
```

---

## 📊 Results

### Multi-Class Model Performance (5-Fold CV)

| Metric | Score |
|---|---|
| **Accuracy** | 95.48% |
| **Precision (macro)** | 96.88% |
| **Recall (macro)** | 96.52% |
| **F1-Score (macro)** | 96.63% |
| **ROC-AUC (OVR)** | 98.70% |

### Diagnosis Output

All unlabeled pump readings (labeled `IDN`, `INN`, etc.) were classified by the final model. Results saved to `processed_data/diagnosis_balanced_predictions.csv`.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn openpyxl joblib
```

### 2. Extract Features

```bash
python extract_features.py
```

Reads all Excel files from `data/` and outputs `train_dataset.csv` and `diagnosis_dataset.csv` to `processed_data/`.

### 3. Train Binary Model (Healthy vs Faulty)

```bash
python train_model.py
```

### 4. Train Multi-Class Model (Fault Type Classification)

```bash
python train_multiclass.py
```

Outputs trained model, confusion matrix, PCA cluster plot, and feature importances to `processed_data/`.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
openpyxl
joblib
```

---

## 🧠 Key Technical Decisions

**SMOTE inside ImbPipeline** — Prevents data leakage by ensuring synthetic samples are only generated from training folds, never seen during validation.

**Stratified K-Fold** — Ensures each fold has proportional representation of all classes, critical for the tiny Combination Defect class (3 samples).

**K-Means on faulty subset only** — Unsupervised clustering runs exclusively on faulty samples to sub-type defects, preserving healthy samples as a clean reference class.

**FFT for time-domain signals** — Converts raw time-series vibration readings into frequency spectra, making bearing defect frequencies (BPFI, BPFO) detectable as feature peaks.

---

## ⚠️ Limitations

- Combination Defect class has only 3 real samples — predictions for this class should be interpreted cautiously in deployment
- K-Means cluster-to-label mapping (Inner/Outer/Combination) is heuristic, based on cluster size and domain knowledge — not ground-truth verified labels
- Model trained on one pump setup; generalization to different pump models or RPM conditions requires revalidation

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

