import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# Configurable Parameters
date_id = "20250319"
record_id = "409"
data_len = "short"  # "short" or "long"
data_type = "pickle"  # "npy", "npz", "csv", "pickle"
segment_length = "256"
custom_folder = "Exp_1310_405"
if custom_folder:
    base_path = f"/home/predator/Documents/redpitaya_ws/datasets/saved_data/{data_type}/s{date_id}/s{date_id}_{record_id}/{segment_length}/{custom_folder}"
else:
    base_path = f"/home/predator/Documents/redpitaya_ws/datasets/saved_data/{data_type}/s{date_id}/s{date_id}_{record_id}/{segment_length}"
print(f"Loading data from: {base_path}")


# Generic loader function
def load_data(filepath):
    ext = os.path.splitext(filepath)[1]
    
    if ext == ".npy":
        return np.load(filepath)
    elif ext == ".npz":
        data = np.load(filepath)
        return dict(data)
    elif ext == ".pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif ext == ".csv":
        return pd.read_csv(filepath).values
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# Load training data
m1_train = load_data(os.path.join(base_path, f"M1_SM_signals_training_{data_len}.{data_type}"))
m2_train = load_data(os.path.join(base_path, f"M2_SM_signals_training_{data_len}.{data_type}"))
disp_train = load_data(os.path.join(base_path, f"Displacement_training_{data_len}.{data_type}"))

# Extract from dicts if needed
if isinstance(m1_train, dict):
    m1_train = m1_train.get("signal", m1_train.get("x"))
if isinstance(m2_train, dict):
    m2_train = m2_train.get("signal", m2_train.get("x"))
if isinstance(disp_train, dict):
    disp_train = disp_train.get("truth", disp_train.get("y"))

# Load test data
m1_test = load_data(os.path.join(base_path, f"M1_SM_signals_test_{data_len}.{data_type}"))
m2_test = load_data(os.path.join(base_path, f"M2_SM_signals_test_{data_len}.{data_type}"))
disp_test = load_data(os.path.join(base_path, f"Displacement_test_{data_len}.{data_type}"))


# Extract from dicts if needed
if isinstance(m1_test, dict):
    m1_test = m1_test.get("signal", m1_test.get("x"))
if isinstance(m2_test, dict):
    m2_test = m2_test.get("signal", m2_test.get("x"))
if isinstance(disp_test, dict):
    disp_test = disp_test.get("truth", disp_test.get("y"))

# Ensure displacement shape is (N,)
if disp_train.ndim == 2 and disp_train.shape[1] == 1:
    disp_train = disp_train[:, 0]
if disp_test.ndim == 2 and disp_test.shape[1] == 1:
    disp_test = disp_test[:, 0]

# Print shapes
print("\nTraining Data Shapes:")
print("M1:", m1_train.shape)
print("M2:", m2_train.shape)
print("Displacement:", disp_train.shape)

print("\nTest Data Shapes:")
print("M1:", m1_test.shape)
print("M2:", m2_test.shape)
print("Displacement:", disp_test.shape)

# Print first training sample
print("\nFirst Training Sample:")
print("M1:", m1_train[0].flatten())
print("M2:", m2_train[0].flatten())
print("Displacement:", disp_train[0])

# Print first test sample
print("\nFirst Test Sample:")
print("M1:", m1_test[0].flatten())
print("M2:", m2_test[0].flatten())
print("Displacement:", disp_test[0])

# Plotting
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(m1_test[0].flatten(), color='tab:blue')
plt.title("Modality 1 - First Test Sample")
plt.xlabel("Time index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(m2_test[0].flatten(), color='tab:orange')
plt.title("Modality 2 - First Test Sample")
plt.xlabel("Time index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.hist(disp_test, bins=50, color='gray', edgecolor='black')
plt.title("Displacement Histogram (Test Set)")
plt.xlabel("Velocity")
plt.ylabel("Count")
plt.grid(True)

plt.tight_layout()
plt.show()
