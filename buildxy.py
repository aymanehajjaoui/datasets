import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# === CONFIG ===
date_id = "20250319"
record_id = "403"
segment_lengths = [256, 48]
x_channels = ["ch1","ch2"]
y_channel = "ch3"
test_ratio = 0.2

root_input_base = os.path.join(os.getcwd(), f"saved_data/csv/s{date_id}/s{date_id}_{record_id}")
output_base = os.path.join(os.getcwd(), f"xydataset/csv/s{date_id}/s{date_id}_{record_id}")

def load_segments(input_base, channel, seg_len):
    file = os.path.join(input_base, f"s{date_id}_{record_id}_{channel}_{seg_len}.csv")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Signal file not found: {file}")
    segments = np.loadtxt(file, delimiter=",")
    if segments.ndim == 1:
        segments = segments.reshape(-1, seg_len)  # Single channel
    return segments

def load_velocity(input_base, channel):
    file = os.path.join(input_base, f"s{date_id}_{record_id}_{channel}_velocity_raw.csv")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Velocity file not found: {file}")
    return np.loadtxt(file, delimiter=",").flatten()

# === PROCESS EACH CHANNEL AND SEGMENT LENGTH ===
for ch in x_channels:
    print(f"\n=== Processing channel {ch} ===")
    channel_output_base = os.path.join(output_base, ch)

    for seg_len in segment_lengths:
        input_base = os.path.join(root_input_base, str(seg_len))
        save_path = os.path.join(channel_output_base, str(seg_len))
        os.makedirs(save_path, exist_ok=True)

        print(f"\n--- Segment length {seg_len} ---")

        # Load X segments
        segments = load_segments(input_base, ch, seg_len)
        print(f"Channel {ch} loaded: {segments.shape}")

        X = segments[..., np.newaxis]  # Add channel axis (N, seg_len, 1)
        print(f"X shape: {X.shape}")

        # Load and align y
        y = load_velocity(input_base, y_channel)
        if len(y) < X.shape[0]:
            raise ValueError(f"Not enough velocity samples ({len(y)}) for {X.shape[0]} signal segments")
        y = y[:X.shape[0]]
        print(f"Y aligned: {y.shape}")

        # Train/Test split
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, shuffle=False
        )
        print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
        print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

        # Save datasets
        np.savetxt(os.path.join(save_path, "x_train.csv"), x_train.reshape(x_train.shape[0], -1), delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(save_path, "x_test.csv"), x_test.reshape(x_test.shape[0], -1), delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(save_path, "y_train.csv"), y_train.reshape(-1, 1), delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(save_path, "y_test.csv"), y_test.reshape(-1, 1), delimiter=",", fmt="%.6f")

        print(f"Saved XY dataset for channel {ch}, segment length {seg_len} to {save_path}")

        # === Plotting ===
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 2, 1)
        plt.plot(x_train.flatten(), color='green')
        plt.title(f"{ch} - x_train (flattened)")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(x_test.flatten(), color='blue')
        plt.title(f"{ch} - x_test (flattened)")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(y_train, color='red')
        plt.title(f"{ch} - y_train")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(y_test, color='orange')
        plt.title(f"{ch} - y_test")
        plt.grid(True)

        plt.suptitle(f"Channel: {ch} | Segment Length: {seg_len}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
