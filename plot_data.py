import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from scipy.signal import savgol_filter

# ========== Parameters ==========
date_id = "20250319"
record_id = "420"
base_path = f"/home/predator/Documents/redpitaya_ws/datasets/collected_data/s{date_id}"
channels = [1, 2, 3, 4]
velocity_channel = "ch3"  # Channel used to compute displacement

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

# ========== Load Available Channels ==========
data_dict = {}
sampling_time = None

for ch in channels:
    ch_key = f"ch{ch}"
    filename = f"s{date_id}_{record_id}_rtb_{ch_key}.pickle"
    filepath = os.path.join(base_path, filename)

    if not os.path.isfile(filepath):
        print(f"File not found for {ch_key}, skipping.")
        continue

    data_raw = load_data(filepath)
    data = scaledata(data_raw)
    data_dict[ch_key] = data

    if sampling_time is None:
        sampling_time = float(data_raw['settings'].split("XINC:")[1].split("\n")[0])

    print(f"Loaded {ch_key} with {len(data)} samples.")

# ========== Validate and Prepare ==========
if not data_dict:
    raise RuntimeError("No valid channel data found.")

sampling_frequency = 1 / sampling_time
segment_length = int(round(sampling_frequency * 0.001))  # 1 ms segments
print(f"\nComputed sampling frequency: {sampling_frequency:.2f} Hz")
print(f"Segment length for 1 ms: {segment_length} samples")

ref_key = list(data_dict.keys())[0]
x = np.arange(len(data_dict[ref_key])) * sampling_time

# ========== Plot Signals ==========
n_channels = len(data_dict)
has_displacement_plot = velocity_channel in data_dict
n_plots = n_channels + (1 if has_displacement_plot else 0)

fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
if n_plots == 1:
    axs = [axs]

plot_index = 0
for ch_key, signal in data_dict.items():
    axs[plot_index].plot(x, signal, linewidth=0.7)
    axs[plot_index].set_title(f"Signal from {ch_key}")
    axs[plot_index].set_ylabel(ch_key)

    for i in range(0, len(signal), segment_length):
        color = 'lightblue' if (i // segment_length) % 2 == 0 else 'lightgreen'
        axs[plot_index].axvspan(x[i], x[min(i + segment_length - 1, len(x) - 1)], color=color, alpha=0.2)

    plot_index += 1

# ========== Displacement Plot ==========
if has_displacement_plot:
    raw = data_dict[velocity_channel]
    filtered = savitzky(raw)

    disp_raw = compute_displacement(raw, segment_length)
    disp_filtered = compute_displacement(filtered, segment_length)

    x_disp = np.linspace(x[0], x[len(raw) - 1], len(disp_raw))

    axs[plot_index].plot(x_disp, disp_raw, label="Displacement (raw)", linewidth=1.2)
    axs[plot_index].plot(x_disp, disp_filtered, label="Displacement (filtered)", linewidth=1.2)
    axs[plot_index].set_title(f"Displacement Analysis – {velocity_channel}")
    axs[plot_index].set_ylabel("Displacement")
    axs[plot_index].legend()

# Force x-tick labels on all subplots
for ax in axs:
    ax.tick_params(labelbottom=True)

# Optional: format x-axis numbers more clearly (6 decimal places)
axs[-1].xaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))

axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Prevent clipping of x-axis labels
plt.show()

# ========== Diagnostics ==========
print("\n========= Diagnostics ==========")
if velocity_channel in data_dict:
    settings_path = os.path.join(base_path, f"s{date_id}_{record_id}_rtb_{velocity_channel}.pickle")
    print(f"Settings for {velocity_channel}:\n{load_data(settings_path)['settings'].strip()}")

print(f"\nSampling time: {sampling_time:.9f} s")
print(f"Segment length: {segment_length} samples")
print(f"Displacement segments: {len(disp_raw) if has_displacement_plot else 'N/A'}")
print(f"Signal duration: {x[-1]:.6f} seconds")
print("================================")
