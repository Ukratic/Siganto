"""Script de conversion SigMF vers WAV.
Expérimental : Prise en compte partielle des schémas possibles"""

import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wav

def get_filepath():
    if len(sys.argv) < 2:
        print("Utilisation : python script.py <input_file.sigmf-meta>")
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Erreur : Fichier non trouvé - {filepath}")
        sys.exit(1)
    if not filepath.endswith(".sigmf-meta"):
        print("Erreur : Le fichier doit avoir l'extension .sigmf-meta")
        sys.exit(1)
    return filepath

def conv_sigmf(metadata_path, float_output=False):
    """Fonction de conversion
    -> fichier de métadonnées en argument
    -> fichier de données dans le même répertoire"""
    # Lecture des métadonnées 
    with open(metadata_path, "r") as meta_file:
        metadata = json.load(meta_file)

    try:
        sample_rate = metadata["global"]["core:sample_rate"]
        data_format = metadata["global"].get("core:datatype", "ci16_le")  # Défaut
    except KeyError as e:
        raise ValueError(f"Métadonnées manquantes : {e}") from e

    print(f"Fréquence d'échantillonnage : {sample_rate} Hz")
    print(f"Format de données : {data_format}")

    # Détermination du type numpy 
    format_map = {
        "ci8":  np.int8,
        "ci16": np.int16,
        "ci32": np.int32,
        "ci64": np.int64,
    }
    base_key = data_format.replace("_le", "").replace("_be", "")
    if base_key not in format_map:
        raise ValueError(f"Format non supporté : {data_format}")

    base_type = format_map[base_key]
    endian = "<" if data_format.endswith("_le") else ">"
    dtype = np.dtype(endian + np.dtype(base_type).str[1:])  # ex. '<i2'

    #  Localisation des fichiers de données
    base = metadata_path.replace(".sigmf-meta", ".sigmf-data")
    data_files = []
    if os.path.exists(base):
        data_files.append(base)
    else:
        # Recherche des fichiers segmentés : .sigmf-data0, .sigmf-data1, ...
        i = 0
        while True:
            fname = f"{base}{i}"
            if os.path.exists(fname):
                data_files.append(fname)
                i += 1
            else:
                break
    if not data_files:
        raise FileNotFoundError(f"Aucun fichier de données trouvé pour {metadata_path}")

    print(f"Fichiers de données détectés : {data_files}")

    #  Lecture et concaténation des données
    iq_list = []
    for f in data_files:
        raw_data = np.fromfile(f, dtype=dtype)
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]  # coupe si impair
        iq_chunk = raw_data[::2] + 1j * raw_data[1::2]
        iq_list.append(iq_chunk)

    iq_data = np.concatenate(iq_list)
    print(f"Nombre total d'échantillons IQ : {len(iq_data)}")
    print(f"Durée du signal : {len(iq_data) / sample_rate:.2f} secondes")


    max_amp = np.max(np.abs(iq_data))
    if max_amp > 0:
        iq_data = iq_data / max_amp
        print("Normalisation appliquée (max -> 1.0)")

    # --- Conversion WAV ---
    if float_output:
        print("Conversion en WAV float32")
        stereo_data = np.column_stack((iq_data.real.astype(np.float32),
                                       iq_data.imag.astype(np.float32)))
    else:
        print("Conversion en WAV int16")
        real_part = (iq_data.real * 32767).astype(np.int16)
        imag_part = (iq_data.imag * 32767).astype(np.int16)
        stereo_data = np.column_stack((real_part, imag_part))

    # --- Sauvegarde ---
    output_file = os.path.splitext(metadata_path)[0] + ".wav"
    wav.write(output_file, int(sample_rate), stereo_data)
    print(f"Fichier WAV écrit : {output_file}")

if __name__ == "__main__":
    try:
        metadata_path = get_filepath()
        conv_sigmf(metadata_path)
    except Exception as e:
        print(f"Error: {e}")
