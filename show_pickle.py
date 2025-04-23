import os
import pickle
import numpy as np

# ========== Parameters ==========
date_id = "20250319"
record_id = "300"
base_path = f"/home/predator/Documents/redpitaya_ws/datasets/collected_data/s{date_id}"
channels = [1, 2, 3, 4]

# ========== Inspect Each Pickle ==========
for ch in channels:
    filename = f"s{date_id}_{record_id}_rtb_ch{ch}.pickle"
    filepath = os.path.join(base_path, filename)

    if not os.path.isfile(filepath):
        print(f"\nFile not found for ch{ch}: {filepath}")
        continue

    print(f"\nInspecting {filename}")
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Show dictionary keys
        print(f"  Keys in file: {list(data.keys())}")

        if "data" in data:
            raw = data["data"]
            print(f"  Type of 'data': {type(raw)}, length: {len(raw)} bytes")

            # Convert to NumPy if it's raw signal bytes
            try:
                signal = np.frombuffer(raw, dtype=">u2")
                print(f"  Decoded signal: shape={signal.shape}, dtype={signal.dtype}")
                print(f"  First 10 samples: {signal[:10]}")
            except Exception as decode_error:
                print(f"  Error decoding 'data': {decode_error}")
        else:
            print("  No 'data' field found in this file.")

    except Exception as e:
        print(f"  Error reading or parsing file: {e}")
