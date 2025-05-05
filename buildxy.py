import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# === CONFIG ===
date_id = "20250319"
record_id = "420"
segment_lengths = [256, 48]
x_channels = ["ch1"]
y_channel = "ch3"
test_ratio = 0.2

root_input_base = f"/home/predator/Documents/redpitaya_ws/datasets/saved_data/csv/s{date_id}/s{date_id}_{record_id}"
output_base = f"/home/predator/Documents/redpitaya_ws/datasets/xydataset/csv/s{date_id}/s{date_id}_{record_id}"

def load_flat_signal(input_base, channel, seg_len):
    file = os.path.join(input_base, f"s{date_id}_{record_id}_{channel}_{seg_len}.csv")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Signal file not found: {file}")
    return np.loadtxt(file, delimiter=",").flatten()

def load_velocity(input_base, channel, seg_len):
    file = os.path.join(input_base, f"s{date_id}_{record_id}_{channel}_velocity.csv")
    if not os.path.isfile(file):
        alt_file = os.path.join(input_base, f"s{date_id}_{record_id}_{channel}_velocity_{seg_len}.csv")
        if os.path.isfile(alt_file):
            print(f"Loaded velocity from alternative file: {alt_file}")
            return np.loadtxt(alt_file, delimiter=",").flatten()
        else:
            raise FileNotFoundError(f"Velocity file not found: {file} or {alt_file}")
    return np.loadtxt(file, delimiter=",").flatten()

def segment_signal(signal, seg_len):
    usable_len = (len(signal) // seg_len) * seg_len
    return signal[:usable_len].reshape(-1, seg_len)

# === PROCESS EACH SEGMENT LENGTH ===
for seg_len in segment_lengths:
    input_base = os.path.join(root_input_base, str(seg_len))
    save_path = os.path.join(output_base, str(seg_len))
    os.makedirs(save_path, exist_ok=True)

    print(f"\n=== Processing segment length {seg_len} ===")

    # Load X
    x_segments = []
    for ch in x_channels:
        flat = load_flat_signal(input_base, ch, seg_len)
        segments = segment_signal(flat, seg_len)
        x_segments.append(segments)
        print(f"Channel {ch} loaded and segmented: {segments.shape}")
    X = np.stack(x_segments, axis=-1)  # (N, seg_len, channels)

    # Load and align y
    y_raw = load_velocity(input_base, y_channel, seg_len)
    if len(y_raw) < X.shape[0]:
        raise ValueError(f"Not enough velocity samples ({len(y_raw)}) for {X.shape[0]} signal segments")
    y = y_raw[:X.shape[0]]
    print(f"Velocity {y_channel} aligned to {len(y)} segments")

    # Train/Test split (no normalization)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )

    # Save
    np.savetxt(os.path.join(save_path, "x_train.csv"), x_train.reshape(x_train.shape[0], -1), delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(save_path, "x_test.csv"), x_test.reshape(x_test.shape[0], -1), delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(save_path, "y_train.csv"), y_train.reshape(-1, 1), delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(save_path, "y_test.csv"), y_test.reshape(-1, 1), delimiter=",", fmt="%.6f")

    print(f"Saved XY dataset for segment length {seg_len} to {save_path}")

    # === Plotting ===
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x_train.flatten(), color='green')
    plt.title("x_train (flattened)")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x_test.flatten(), color='blue')
    plt.title("x_test (flattened)")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(y_train, color='red')
    plt.title("y_train")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(y_test, color='orange')
    plt.title("y_test")
    plt.grid(True)

    plt.suptitle(f"Segment Length {seg_len}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
