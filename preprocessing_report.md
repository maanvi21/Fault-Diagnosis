# Core Preprocessing Report: Mechanical Signal Extraction

The raw datasets presented a monumental preprocessing challenge: The data was structurally entirely different between the healthy base and the faulty sets. Here is a definitive, step-by-step report breaking down exactly how the raw mechanical arrays were parsed, manipulated, and preprocessed into a uniform machine-learning-ready framework without destroying their structural integrity. 

## 1. The Domain Gap (Time vs Frequency)
Upon initial inspection of the Excel arrays (`pandas.read_excel`), there was a fundamental clash in physics:
*   **Healthy Datasets:** Represented the *Frequency Domain* (mostly labeled `Z_FFT` or `X_FFT`), mapping harmonic frequency amplitudes natively.
*   **Faulty Datasets:** Represented the pure *Time Domain* (`X(t)`), representing raw analog wave cycles over time natively.

An ML classifier cannot mathematically evaluate Time inputs against Frequency inputs directly; they are completely different dimensional models. 

## 2. Fast Fourier Transform (FFT) Conversion
To preprocess these completely different structures into a single unified dataset, I utilized `numpy.fft` (specifically `rfft` and `rfftfreq`).

During extraction, the raw Time-Domain records (`X(t)`) from the **261 Faulty Excel sheets** were dynamically fed through the FFT algorithm calculation. This mechanically forced the analog Time waves to shatter into their fundamental harmonic frequencies, outputting pure Frequency/Magnitude arrays that exactly matched the identical scale and layout of the pre-processed Healthy datasets.

## 3. Mathematical Feature Formulation
Rather than feeding tens of thousands of raw Excel rows into the Random Forest (which would instantly crash it), I ran a custom algorithmic pipeline over both the natively Healthy arrays and the newly transformed Faulty FFT arrays to calculate **9 distinct mechanical parameters** per file:
1.  **`max_amp`**: The absolute highest magnitude peak (indicating the core fault rotation).
2.  **`mean_mag`**: The average amplitude baseline across the entire spectrum.
3.  **`var_mag`**: Mathematical variance tracking geometric vibration instability.
4.  **`spectral_energy`**: The sum of all squared magnitudes, calculating total systemic destructive energy load.
5.  **`spectral_centroid`**: The calculated center of mass of the frequencies, shifting massively when structural components crack.
6.  **`spectral_spread`**: The geometric bandwidth spread around the centroid showing wave chaos.
7.  **`peak_f1`, `peak_f2`, `peak_f3`**: The top 3 distinctly dominant Hz frequency limits cleanly isolated by slicing out the heaviest harmonic amplitudes.

## 4. Null & Blank Value Sanitization
Many hardware datasets occasionally drop packet records or output corrupted columns. During Pandas ingestion, any single row lacking finite parameters or throwing `NaN` / infinite numeric loops was completely purged using strict array cleaning. If the hardware failed to read values (throwing generic Exception errors), the script skipped the broken file naturally without crashing.

## 5. Security & Data Leakage Masking
The `PUMP DIAGNOSIS` folder contained ~152 ambiguously titled records. To preprocess this massive threat to our ML model cleanly:
*   `data/Healthy Reading/` datasets were aggressively assigned target classification `= 0`.
*   `data/Faulty readings/` datasets were aggressively assigned target classification `= 1`.
*   All data routed through `data/PUMP DIAGNOSIS/` was formally assigned target classification `= -1`. 
By forcefully tagging these external arrays with an unknown integer loop (`-1`) during Pandas `.to_csv` compilation, the data arrays were successfully compiled into a giant feature matrix while being mathematically firewalled from accidentally polluting the Training models. 

---

### Final Output Structure
Once all raw structural `.xlsx` strings were ingested, transformed via FFT, squeezed down to 9 parameters, formatted over classes, and checked for `NaN` loops, the entire 2-gigabyte directory footprint was flattened elegantly into a single lightweight numerical matrix saved dynamically at: `processed_data/train_dataset.csv`.
