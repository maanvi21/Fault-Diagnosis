import pandas as pd
import numpy as np
import os
import glob

def extract_features_from_excel(filepath):
    try:
        df = pd.read_excel(filepath)
        if df.shape[0] < 3 or df.shape[1] < 2:
            return None

        unit_str = str(df.iloc[0, 0]).strip()
        
        # Data starts from index 2
        col1 = pd.to_numeric(df.iloc[2:, 0], errors='coerce').values
        col2 = pd.to_numeric(df.iloc[2:, 1], errors='coerce').values
        
        # Remove NaNs
        mask = ~np.isnan(col1) & ~np.isnan(col2)
        col1 = col1[mask]
        col2 = col2[mask]
        
        if len(col1) == 0:
            return None

        if 's' in unit_str.lower():
            # Time Domain
            t = col1
            x = col2
            dt = t[1] - t[0] if len(t) > 1 and t[1] > t[0] else 0.000667 # default from F10
            
            # Compute FFT
            # Using rfft since real signal
            fft_complex = np.fft.rfft(x)
            mags = np.abs(fft_complex)
            freqs = np.fft.rfftfreq(len(x), d=dt)
        else:
            # Frequency Domain (Hz)
            freqs = col1
            mags = np.abs(col2)

        # Ensure we have valid arrays
        if len(mags) == 0 or np.sum(mags) == 0:
            return None

        # --- Extract Features ---
        max_amp = np.max(mags)
        mean_mag = np.mean(mags)
        var_mag = np.var(mags)
        spectral_energy = np.sum(mags ** 2)
        
        # Spectral centroid
        if np.sum(mags) != 0:
            centroid = np.sum(freqs * mags) / np.sum(mags)
            spread_var = np.sum(((freqs - centroid) ** 2) * mags) / np.sum(mags)
            spectral_spread = np.sqrt(spread_var) if spread_var > 0 else 0
        else:
            centroid = 0
            spectral_spread = 0

        # Peak frequencies
        peak_indices = np.argsort(mags)[-3:][::-1] # top 3
        peak_f1 = freqs[peak_indices[0]] if len(peak_indices) > 0 else 0
        peak_f2 = freqs[peak_indices[1]] if len(peak_indices) > 1 else 0
        peak_f3 = freqs[peak_indices[2]] if len(peak_indices) > 2 else 0

        return {
            'file': os.path.basename(filepath),
            'max_amp': max_amp,
            'mean_mag': mean_mag,
            'var_mag': var_mag,
            'spectral_energy': spectral_energy,
            'spectral_centroid': centroid,
            'spectral_spread': spectral_spread,
            'peak_f1': peak_f1,
            'peak_f2': peak_f2,
            'peak_f3': peak_f3
        }

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

if __name__ == "__main__":
    healthy_files = glob.glob('data/Healthy Reading/*.xlsx')
    faulty_files = glob.glob('data/Faulty readings/NEW Time Domain/*.xlsx')
    unknown_files = glob.glob('data/PUMP DIAGNOSIS/*.xlsx')

    print(f"Found {len(healthy_files)} healthy files.")
    print(f"Found {len(faulty_files)} faulty files.")
    print(f"Found {len(unknown_files)} unknown (diagnosis) files.")

    dataset = []

    # Process Healthy
    for f in healthy_files:
        feat = extract_features_from_excel(f)
        if feat:
            feat['label'] = 0 # 0 for healthy
            dataset.append(feat)

    # Process Faulty
    for f in faulty_files:
        feat = extract_features_from_excel(f)
        if feat:
            feat['label'] = 1 # 1 for faulty
            dataset.append(feat)
            
    # Process Diagnosis (Unknown)
    diagnosis_dataset = []
    for f in unknown_files:
        feat = extract_features_from_excel(f)
        if feat:
            feat['label'] = -1 # -1 for unknown
            diagnosis_dataset.append(feat)

    df_train = pd.DataFrame(dataset)
    df_diag = pd.DataFrame(diagnosis_dataset)

    print(f"Constructed train/val dataset with shape {df_train.shape}")
    print(f"Constructed diagnosis dataset with shape {df_diag.shape}")

    # Save to csv
    os.makedirs('processed_data', exist_ok=True)
    df_train.to_csv('processed_data/train_dataset.csv', index=False)
    df_diag.to_csv('processed_data/diagnosis_dataset.csv', index=False)
    print("Files saved to `processed_data/` folder.")
