import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, resample

# ========== Parameters ==========
date_id = "20250319"
record_id = "403"
resampled_lengths = [48, 256]

base_path = os.path.join(os.getcwd(), f"collected_data/s{date_id}")
signal_channels = ["ch1","ch2"]
velocity_channels = ["ch3"]

# ========== Helpers ==========
def load_data(filepath):
    with open(filepath, 'rb') as fd:
        return pickle.load(fd)

def scaledata(data):
    settings = data['settings']
    trace = np.frombuffer(data['data'], dtype='>u2')
    gain = float(settings.split("YINC:")[1].split("\n")[0])
    offset = float(settings.split("YOR:")[1].split("\n")[0])
    return trace * gain - offset

def savitzky(signal):
    window_length = 4001 if 4000 % 2 == 0 else 4000
    return savgol_filter(signal, window_length, 3)

def compute_displacement(signal, segment_len):
    usable_len = (len(signal) // segment_len) * segment_len
    segments = signal[:usable_len].reshape(-1, segment_len)
    return np.array([s[-1] - s[0] for s in segments])

def segment_and_resample(signal, orig_len, target_len):
    usable_len = (len(signal) // orig_len) * orig_len
    segments = signal[:usable_len].reshape(-1, orig_len)
    return np.array([resample(s, target_len) for s in segments])

# ========== Sampling Info ==========
first_file = os.path.join(base_path, f"s{date_id}_{record_id}_rtb_{signal_channels[0]}.pickle")
if not os.path.isfile(first_file):
    raise FileNotFoundError(f"Sampling time source file not found: {first_file}")

sampling_time = float(load_data(first_file)['settings'].split("YINC:")[1].split("\n")[0])
sampling_freq = 1 / sampling_time
segment_orig_len = int(round(sampling_freq * 0.001))  # ~1 ms worth of samples

print(f"Sampling time: {sampling_time:.9f} s")
print(f"Sampling frequency: {sampling_freq:.2f} Hz")
print(f"Original segment length: {segment_orig_len} samples")

# ========== Process for Each Target Length ==========
for resampled_len in resampled_lengths:
    print(f"\n=== Processing for resampled segment length: {resampled_len} ===")

    # Paths for csv, npy, pickle
    base_csv = os.path.join(os.getcwd(), f"saved_data/csv/s{date_id}/s{date_id}_{record_id}/{resampled_len}")
    base_npy = os.path.join(os.getcwd(), f"saved_data/npy/s{date_id}/s{date_id}_{record_id}/{resampled_len}")
    base_pickle = os.path.join(os.getcwd(), f"saved_data/pickle/s{date_id}/s{date_id}_{record_id}/{resampled_len}")

    # Create folders
    Path(base_csv).mkdir(parents=True, exist_ok=True)
    Path(base_npy).mkdir(parents=True, exist_ok=True)
    Path(base_pickle).mkdir(parents=True, exist_ok=True)

    signal_data, signal_titles = [], []
    velocity_data, velocity_titles = [], []

    # --- Signal Channels ---
    for ch in signal_channels:
        path = os.path.join(base_path, f"s{date_id}_{record_id}_rtb_{ch}.pickle")
        if not os.path.isfile(path):
            print(f"Signal file not found: {ch}")
            continue

        data = scaledata(load_data(path))
        resampled_segments = segment_and_resample(data, segment_orig_len, resampled_len)

        base_filename = f"s{date_id}_{record_id}_{ch}_{resampled_len}"

        # Save CSV (one segment per row)
        out_csv = os.path.join(base_csv, base_filename + ".csv")
        np.savetxt(out_csv, resampled_segments, delimiter=",", fmt="%.6f")
        print(f"Saved CSV: {out_csv} — shape: {resampled_segments.shape}")

        # Save NPY
        out_npy = os.path.join(base_npy, base_filename + ".npy")
        np.save(out_npy, resampled_segments)
        print(f"Saved NPY: {out_npy}")

        # Save Pickle
        out_pickle = os.path.join(base_pickle, base_filename + ".pickle")
        with open(out_pickle, 'wb') as f:
            pickle.dump(resampled_segments, f)
        print(f"Saved Pickle: {out_pickle}")

        signal_data.append(data)
        signal_titles.append(ch)

    # --- Velocity Channels / Displacement ---
    for ch in velocity_channels:
        path = os.path.join(base_path, f"s{date_id}_{record_id}_rtb_{ch}.pickle")
        if not os.path.isfile(path):
            print(f"Velocity file not found: {ch}")
            continue

        raw_data = scaledata(load_data(path))
        filtered_data = savitzky(raw_data)

        disp_raw = compute_displacement(raw_data, segment_orig_len)
        disp_filtered = compute_displacement(filtered_data, segment_orig_len)

        # Save raw displacement
        base_filename_raw = f"s{date_id}_{record_id}_{ch}_velocity_raw"
        out_csv_raw = os.path.join(base_csv, base_filename_raw + ".csv")
        np.savetxt(out_csv_raw, disp_raw.reshape(-1, 1), delimiter=",", fmt="%.6f")
        print(f"Saved CSV: {out_csv_raw} — shape: {disp_raw.shape}")

        # Save filtered displacement
        base_filename_filtered = f"s{date_id}_{record_id}_{ch}_velocity_filtered"
        out_csv_filtered = os.path.join(base_csv, base_filename_filtered + ".csv")
        np.savetxt(out_csv_filtered, disp_filtered.reshape(-1, 1), delimiter=",", fmt="%.6f")
        print(f"Saved CSV: {out_csv_filtered} — shape: {disp_filtered.shape}")

        # Save NPY and Pickle for filtered displacement only
        out_npy = os.path.join(base_npy, base_filename_filtered + ".npy")
        np.save(out_npy, disp_filtered)
        print(f"Saved NPY: {out_npy}")

        out_pickle = os.path.join(base_pickle, base_filename_filtered + ".pickle")
        with open(out_pickle, 'wb') as f:
            pickle.dump(disp_filtered, f)
        print(f"Saved Pickle: {out_pickle}")

        velocity_data.append(disp_filtered)
        velocity_titles.append(ch)

    # --- Plotting ---
    all_data = signal_data
    all_titles = [f"Signal: {ch}" for ch in signal_titles]
    colors = ['blue'] * len(signal_data)

    # Add displacement subplot if velocity data exists
    include_displacement = bool(velocity_data)
    n_subplots = len(all_data) + (1 if include_displacement else 0)

    if all_data or include_displacement:
        fig, axs = plt.subplots(n_subplots, 1, figsize=(12, 3 * n_subplots), sharex=False)
        if n_subplots == 1:
            axs = [axs]

        # Plot all signal channels
        for i, data in enumerate(all_data):
            timebase = sampling_time
            t = np.arange(len(data)) * timebase
            axs[i].plot(t, data, color=colors[i], linewidth=0.7)
            axs[i].set_title(f"{all_titles[i]} — {len(data)} samples")
            axs[i].set_ylabel("Amplitude")
            axs[i].grid(True)

        # Plot both raw and filtered displacement
        if include_displacement:
            disp_raw = compute_displacement(raw_data, segment_orig_len)
            disp_filtered = compute_displacement(filtered_data, segment_orig_len)
            t_disp = np.linspace(0, len(raw_data) * sampling_time, len(disp_raw))

            ax_disp = axs[-1]
            ax_disp.plot(t_disp, disp_raw, label="Displacement (Raw)", color='blue', linewidth=1.0)
            ax_disp.plot(t_disp, disp_filtered, label="Displacement (Filtered)", color='orange', linewidth=1.0)
            ax_disp.set_title(f"Displacement (Raw & Filtered)")
            ax_disp.set_ylabel("Displacement")
            ax_disp.set_xlabel("Time (s)")
            ax_disp.legend()
            ax_disp.grid(True)

        fig.suptitle(f"Signals and Displacement (Resampled to {resampled_len})", fontsize=14)
        plt.tight_layout()
        plt.show()