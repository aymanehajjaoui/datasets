import numpy as np
import matplotlib.pyplot as plt
import pickle, glob, os
from scipy.signal import savgol_filter, resample

# ========== Parameters ==========
date_id = "20250319"
record_id = "403"
base_path = os.path.join(os.getcwd(), f"collected_data/s{date_id}")
save_base = os.path.join(os.getcwd(), f"saved_csv_raw/s{date_id}/s{date_id}_{record_id}")
channels = [1, 2, 3, 4]

velocity_channel = "ch3"

# ========== Helpers ==========
def resampling(signal, f, f2):
    return resample(signal, int(len(signal) * f2 / f))

def load_data(fullname):
    with open(fullname, 'rb') as fd:
        return pickle.load(fd)

def scaledata(data):
    settings = data['settings']
    trace = np.frombuffer(data['data'], dtype='>u2')
    gain = float(settings.split("YINC:")[1].split("\n")[0])
    offset = float(settings.split("YOR:")[1].split("\n")[0])
    return trace * gain - offset

def savitzky(data):
    window_length = 4001 if 4000 % 2 == 0 else 4000
    return savgol_filter(data, window_length, 3)

def compute_displacement(signal, segment_len):
    usable_len = (len(signal) // segment_len) * segment_len
    segments = signal[:usable_len].reshape(-1, segment_len)
    return np.array([s[-1] - s[0] for s in segments])

# ========== Load, Save Channels ==========
data_dict = {}
file_titles = {}
sampling_time = None
available_channels = []

os.makedirs(save_base, exist_ok=True)

for ch in channels:
    search_pattern = os.path.join(base_path, f"s{date_id}_{record_id}_rtb_ch{ch}.pickle")
    files = glob.glob(search_pattern)
    if not files:
        print(f"File for channel {ch} not found. Skipping.")
        continue

    file = files[0]
    data_raw = load_data(file)
    signal = scaledata(data_raw)

    if ch == 1 and sampling_time is None:
        sampling_time = float(data_raw['settings'].split("YINC:")[1].split("\n")[0])

    ch_key = f"ch{ch}"
    data_dict[ch_key] = signal
    file_titles[ch_key] = os.path.basename(file)
    available_channels.append(ch)

sampling_frequency = 1 / sampling_time
segment_length = int(round(sampling_frequency * 0.001))  # 1 ms segments
print(f"\nComputed sampling frequency: {sampling_frequency:.2f} Hz")
print(f"Segment length for 1 ms: {segment_length} samples")

# Save channels as CSV (one segment per row)
for ch_key, signal in data_dict.items():
    usable_len = (len(signal) // segment_length) * segment_length
    segments = signal[:usable_len].reshape(-1, segment_length)

    filename = f"s{date_id}_{record_id}_{ch_key}.csv"
    save_path = os.path.join(save_base, filename)
    np.savetxt(save_path, segments, delimiter=",", fmt="%.6f")
    print(f"Saved CSV: {save_path} with shape {segments.shape}")

# ========== Save Displacement (raw & filtered) ==========
if velocity_channel in data_dict:
    raw_signal = data_dict[velocity_channel]
    filtered_signal = savitzky(raw_signal)

    disp_raw = compute_displacement(raw_signal, segment_length)
    disp_filtered = compute_displacement(filtered_signal, segment_length)

    # Save raw displacement
    save_path_raw = os.path.join(save_base, f"s{date_id}_{record_id}_{velocity_channel}_velocity_raw.csv")
    np.savetxt(save_path_raw, disp_raw.reshape(-1, 1), delimiter=",", fmt="%.6f")
    print(f"Saved raw displacement CSV: {save_path_raw} with shape {disp_raw.shape}")

    # Save filtered displacement
    save_path_filtered = os.path.join(save_base, f"s{date_id}_{record_id}_{velocity_channel}_velocity_filtered.csv")
    np.savetxt(save_path_filtered, disp_filtered.reshape(-1, 1), delimiter=",", fmt="%.6f")
    print(f"Saved filtered displacement CSV: {save_path_filtered} with shape {disp_filtered.shape}")
else:
    print(f"Velocity channel '{velocity_channel}' not found in data_dict.")

# ========== Plot Raw Signals + Both Displacements ==========
n_subplots = len(available_channels) + 1  # Add one for displacement
fig, axs = plt.subplots(n_subplots, 1, figsize=(12, 3 * n_subplots), sharex=True)
if n_subplots == 1:
    axs = [axs]

# Plot raw signals
for idx, ch in enumerate(available_channels):
    ch_key = f"ch{ch}"
    signal = data_dict[ch_key]
    t = np.arange(len(signal)) * sampling_time
    ax = axs[idx]
    ax.plot(t, signal, linewidth=0.7)
    ax.set_ylabel(ch_key, fontsize=9)
    ax.set_title(file_titles[ch_key])
    ax.grid(True)

# Plot both raw and filtered displacement
if velocity_channel in data_dict:
    t_disp = np.linspace(0, len(raw_signal) * sampling_time, len(disp_raw))
    ax_disp = axs[-1]
    ax_disp.plot(t_disp, disp_raw, label="Displacement (Raw)", color='blue', linewidth=1.2)
    ax_disp.plot(t_disp, disp_filtered, label="Displacement (Filtered)", color='orange', linewidth=1.2)
    ax_disp.set_title("Displacement – Raw and Filtered")
    ax_disp.set_ylabel("Displacement")
    ax_disp.set_xlabel("Time (s)")
    ax_disp.legend()
    ax_disp.grid(True)

plt.tight_layout()
plt.suptitle("Raw Signals and Displacement", fontsize=14)
plt.subplots_adjust(top=0.93)
plt.show()

# ========== Diagnostics ==========
print("Sampling frequency:", sampling_frequency, "Hz")
print("Duration (s):", len(data_dict[f'ch{available_channels[0]}']) * sampling_time)
