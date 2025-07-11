import numpy as np
import matplotlib.pyplot as plt
import pickle, glob, os
from scipy.signal import savgol_filter,resample

# ========== Parameters ==========
date_id = "20250319"
record_id = "403"
base_path = f"/home/predator/Documents/redpitaya_ws/datasets/collected_data/s{date_id}"
save_base = f"/home/predator/Documents/redpitaya_ws/datasets/saved_csv_raw/s{date_id}/s{date_id}_{record_id}"
channels = [1, 2, 3, 4]

save_filtered_velocity = True
velocity_channel = "ch3"

# ========== Helpers ==========
def resampling(signal,f,f2):
    signal_resampled = resample(signal, int(len(signal) * f2 / f))
    return (signal_resampled)

def load_data(fullname):
    with open(fullname, 'rb') as fd:
          data = pickle.load(fd)
          fd.close()
    return data

def scaledata(data):
    settings = data['settings']
    trace = np.frombuffer(data['data'], dtype='>u2')
    gain = float(settings.split("YINC:")[1].split("\n")[0])
    offset = float(settings.split("YOR:")[1].split("\n")[0])
    trace = trace * gain - offset
    return(trace)

def savitzky(data):
    window_length = 4001 if 4000 % 2 == 0 else 4000
    return savgol_filter(data, window_length, 3)


def disp_treat(disp):
    disp=disp[:(len(disp)-len(disp)%256)]
    X=len(disp)//256
    disp=np.reshape(disp,(X,256))
    train_displacement=[disp[i,255]-disp[i,0] for i in range (X)]
    train_displacement=np.reshape(train_displacement,(X,1))
    return train_displacement

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

    filename = f"s{date_id}_{record_id}_{ch_key}.csv"
    save_path = os.path.join(save_base, filename)
    np.savetxt(save_path, signal.reshape(1, -1), delimiter=",", fmt="%.6f")
    print(f"Saved CSV: {save_path} with shape {signal.shape}")

# ========== Save Velocity (if enabled) ==========
if velocity_channel in data_dict:
    velocity_source = savitzky(data_dict[velocity_channel]) if save_filtered_velocity else data_dict[velocity_channel]
    velocity = np.diff(velocity_source)
    save_path = os.path.join(save_base, f"s{date_id}_{record_id}_{velocity_channel}_velocity.csv")
    np.savetxt(save_path, velocity.reshape(-1, 1), delimiter=",", fmt="%.6f")
    print(f"Saved velocity CSV: {save_path} with shape {velocity.shape}")
else:
    print(f"Velocity channel '{velocity_channel}' not found in data_dict.")

# ========== Plot Full Signals ==========
fig, axs = plt.subplots(len(available_channels), 1, figsize=(12, 3 * len(available_channels)), sharex=True)
if len(available_channels) == 1:
    axs = [axs]

for idx, ch in enumerate(available_channels):
    ch_key = f"ch{ch}"
    signal = data_dict[ch_key]
    t = np.arange(len(signal)) * sampling_time
    ax = axs[idx]
    ax.plot(t, signal, linewidth=0.7)
    ax.set_ylabel(ch_key, fontsize=9)
    ax.set_title(file_titles[ch_key])

plt.tight_layout()
plt.suptitle("Raw Signals", fontsize=14)
plt.subplots_adjust(top=0.93)
plt.show()

# ========== Diagnostics ==========
print("Sampling frequency:", 1 / sampling_time, "Hz")
print("Duration (s):", len(data_dict[f'ch{available_channels[0]}']) * sampling_time)
