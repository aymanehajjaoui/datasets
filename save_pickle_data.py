import os
import numpy as np
import pickle
import glob

# Input identifiers
date_id = "20250319"
record_id = "303"
custom_folder = "Exp_1310_405M"
data_length = "109"
data_len = "short"  # or "short"

# Base paths
save_base = "/home/predator/Documents/redpitaya_ws/datasets/saved_data"
if custom_folder:
    npy_folder = f"{save_base}/npy/s{date_id}/s{date_id}_{record_id}/{data_length}/{custom_folder}"
    pickle_folder = f"{save_base}/pickle/s{date_id}/s{date_id}_{record_id}/{data_length}/{custom_folder}"
    brainmix_folder = f"/home/predator/Documents/redpitaya_ws/datasets/BrainMix_plugin_datasets/s{date_id}_{record_id}/{data_length}"
else:
    npy_folder = f"{save_base}/npy/s{date_id}/s{date_id}_{record_id}/{data_length}"
    pickle_folder = f"{save_base}/pickle/s{date_id}/s{date_id}_{record_id}/{data_length}"
    brainmix_folder = f"/home/predator/Documents/redpitaya_ws/datasets/BrainMix_plugin_datasets/s{date_id}_{record_id}/{data_length}"

# Ensure destination folders exist
os.makedirs(pickle_folder, exist_ok=True)
os.makedirs(brainmix_folder, exist_ok=True)

# Find all .npy files
npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))

# Storage
train_signal = None
train_truth = None
test_signal = None
test_truth = None

# Loop through and convert all individual .npy to .pickle
for npy_file in npy_files:
    basename = os.path.basename(npy_file)
    pickle_name = os.path.splitext(basename)[0] + ".pickle"
    pickle_path = os.path.join(pickle_folder, pickle_name)

    data = np.load(npy_file)
    wrapped = {}

    if "Displacement_training" in basename or "truth_training" in basename:
        wrapped = {"truth": data.squeeze().astype(np.float32)}
        train_truth = wrapped["truth"]
    elif "M1_SM_signals_training" in basename or "M2_SM_signals_training" in basename:
        wrapped = {"signal": data.astype(np.float32)}
        train_signal = wrapped["signal"]
    elif "Displacement_test" in basename or "truth_test" in basename:
        wrapped = {"truth": data.squeeze().astype(np.float32)}
        test_truth = wrapped["truth"]
    elif "M1_SM_signals_test" in basename or "M2_SM_signals_test" in basename:
        wrapped = {"signal": data.astype(np.float32)}
        test_signal = wrapped["signal"]

    # Save individual file
    if wrapped:
        with open(pickle_path, "wb") as f:
            pickle.dump(wrapped, f)
        print(f"Converted {basename} to {pickle_name} ({wrapped['signal'].shape if 'signal' in wrapped else wrapped['truth'].shape})")

# === Save combined BrainMIX-compatible files ===

# Training
if train_signal is not None and train_truth is not None:
    train_data = {"signal": train_signal, "truth": train_truth}
    train_outfile = os.path.join(brainmix_folder, f"traindata{data_length}_{data_len}.pickle")
    with open(train_outfile, "wb") as f:
        pickle.dump(train_data, f)
    print(f"\nSaved BrainMIX training file: {train_outfile}")
else:
    print("\nWARNING: Missing training signal or truth.")

# Test / Validation
if test_signal is not None and test_truth is not None:
    test_data = {"signal": test_signal, "truth": test_truth}
    test_outfile = os.path.join(brainmix_folder, f"valid{data_length}_{data_len}.pickle")
    with open(test_outfile, "wb") as f:
        pickle.dump(test_data, f)
    print(f"Saved BrainMIX validation file: {test_outfile}")
else:
    print("WARNING: Missing test signal or truth.")

print("\nConversion complete.")
print(f"All .pickle files saved in: {os.path.abspath(pickle_folder)}")
