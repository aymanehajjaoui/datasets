import os
import pickle
import numpy as np

# === CONFIGURATION ===
date_id = "20250319"
record_id = "300"
segment_lengths = [48,256]  # Seulement 48 pour BrainMix
source_root = f"/home/predator/Documents/redpitaya_ws/datasets/xydataset/csv/s{date_id}/s{date_id}_{record_id}"
output_root = f"/home/predator/Documents/redpitaya_ws/datasets/datasets_brainmix_plugin/s{date_id}/s{date_id}_{record_id}"

# === TRAITEMENT ===
for seg_len in segment_lengths:
    print(f"\n📦 Traitement du segment de longueur : {seg_len}")
    source_path = os.path.join(source_root, str(seg_len))
    output_path = os.path.join(output_root, str(seg_len))
    os.makedirs(output_path, exist_ok=True)

    # Chemins d'entrée CSV
    x_train_path = os.path.join(source_path, "x_train.csv")
    x_test_path = os.path.join(source_path, "x_test.csv")
    y_train_path = os.path.join(source_path, "y_train.csv")
    y_test_path = os.path.join(source_path, "y_test.csv")

    # Chargement des données
    x_train = np.loadtxt(x_train_path, delimiter=",").reshape(-1, seg_len, 1)
    x_test = np.loadtxt(x_test_path, delimiter=",").reshape(-1, seg_len, 1)
    y_train = np.loadtxt(y_train_path, delimiter=",").reshape(-1)
    y_test = np.loadtxt(y_test_path, delimiter=",").reshape(-1)

    # Construction des dictionnaires
    train_dict = {
        "signal": x_train.astype(np.float32),
        "truth": y_train.astype(np.float32)
    }
    valid_dict = {
        "signal": x_test.astype(np.float32),
        "truth": y_test.astype(np.float32)
    }

    # Sauvegarde des .pickle
    with open(os.path.join(output_path, f"traindata{seg_len}_shuffled.pickle"), "wb") as f:
        pickle.dump(train_dict, f)
    with open(os.path.join(output_path, f"valid{seg_len}.pickle"), "wb") as f:
        pickle.dump(valid_dict, f)

    print(f"✅ Fichiers sauvegardés dans : {output_path}")
