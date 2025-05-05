# Dataset Processing Scripts

This repository contains scripts to handle and process signal datasets collected from RedPitaya experiments. Below is a description of each main script:

## 1. `plot_data.py`

**Purpose:**  
Visualize raw data from the `collected_data` folder.

**Usage:**  
- Loads `.pickle` files for specified channels.  
- Plots the signals and highlights segments visually.  
- Optionally computes and plots displacement from a selected velocity channel.

**Input:**  
Files located in:  
```
collected_data/s<date_id>/s<date_id>_<record_id>_rtb_ch<channel>.pickle
```

---

## 2. `save_csv_data.py`

**Purpose:**  
Convert raw `.pickle` files into CSV format and save them in `saved_csv_data`.

**Usage:**  
- Loads each specified channel from `collected_data`.  
- Saves each signal to a `.csv` file (one row per signal).  
- Optionally computes velocity and saves it as a separate `.csv` file.

**Output:**  
Files saved in:  
```
saved_csv_data/s<date_id>/s<date_id>_<record_id>/
```

---

## 3. `build_data.py`

**Purpose:**  
Prepare segmented datasets from the saved CSVs.

**Usage:**  
- Reads signals and velocity CSVs from `saved_csv_data`.  
- Segments the signals into fixed lengths (e.g., 48, 256 samples).  
- Resamples and computes displacement if needed.  
- Saves processed data as `.csv`, `.npy`, and `.pickle` formats for machine learning pipelines.

**Output:**  
Files saved in:  
```
saved_data/csv/
saved_data/npy/
saved_data/pickle/
```

---

## 4. `buildxy.py`

**Purpose:**  
Create train/test XY datasets from the segmented CSVs.

**Usage:**  
- Loads segmented signal data and velocity/displacement data.  
- Splits into train/test sets (with configurable ratio).  
- Saves the datasets as flattened `.csv` files for model training.  
- Plots train/test distributions for visual inspection.

**Output:**  
Files saved in:  
```
xydataset/csv/s<date_id>/s<date_id>_<record_id>/<segment_length>/
```

---

# Folder Structure

- `collected_data/`: Raw `.pickle` files collected from RedPitaya.
- `saved_csv_data/`: CSV versions of the raw signals.
- `saved_data/`: Processed signals and displacements (csv, npy, pickle formats).
- `xydataset/`: Train/test XY datasets for machine learning.

---

# Notes

- Make sure to **adjust parameters** inside each script (e.g., `date_id`, `record_id`, channels) before running.
- All paths are **hardcoded** for now; consider adapting them for your environment.
- Ensure required dependencies are installed (e.g., `numpy`, `matplotlib`, `scikit-learn`, `scipy`).


