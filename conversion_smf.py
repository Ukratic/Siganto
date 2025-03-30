"""Script de conversion SigMF vers WAV.
Expérimental : Prise en compte partielle des schémas possibles"""

import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wav

def get_filepath():
    """Chemin du fichier"""
    if len(sys.argv) < 2:
        print("Utilisation : python script.py <input_file.sigmf-meta>")
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Erreur : Vérifier chemin - {filepath}")
        sys.exit(1)

    return filepath

def conv_sigmf(metadata_path):
    """Fonction de conversion
    -> fichier de métadonnées en argument
    -> fichier de données dans le même répertoire"""
    # Charge metadonnees en 1er
    with open(metadata_path, 'r') as meta_file:
        metadata = json.load(meta_file)
    # Infos nécessaires
    try:
        data_file = metadata_path.replace(".sigmf-meta", ".sigmf-data")
        sample_rate = metadata['global']['core:sample_rate']
        data_format = metadata['global'].get('core:datatype', 'ci16_le')  # Défaut: 16-bit IQ
    except KeyError as e:
        raise ValueError(f"Métadonnées manquantes : {e}") from e
    print(f"Fréquence d'échantillonnage: {sample_rate} Hz, format : {data_format}")
    print(f"Récupère signal de : {data_file}")
    # Formats en equivalents numpy
    format_map = {'ci8': 'int8','ci16': 'int16','ci32': 'int32','ci64': 'int64'}
    endian = '<' if '_le' in data_format else '>'  # Little/Big endian
    dtype = np.dtype(endian + format_map.get(data_format.replace('_le', '').replace('_be', ''), 'ci16').replace('int', 'i').replace('16', '2'))
    # Charge donnees
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Fichier de données non trouvé : {data_file}")
    raw_data = np.fromfile(data_file, dtype=dtype)
    # Interpretation IQ
    iq_data = raw_data[::2] + 1j * raw_data[1::2]
    # Normalise pour écriture en wav
    max_amplitude = np.max(np.abs(iq_data))
    if max_amplitude > 0:
        iq_data_normalized = iq_data / max_amplitude
    else:
        iq_data_normalized = iq_data
    # Conversion 16-bit wav
    print("Conversion en wav 16bit")
    real_part = (iq_data_normalized.real * 32767).astype(np.int16)
    imag_part = (iq_data_normalized.imag * 32767).astype(np.int16)
    stereo_data = np.column_stack((real_part, imag_part))
    # Enregistre en wav
    output_file = metadata_path.replace(".sigmf-meta", ".wav")
    print(f"Enregistre sous : {output_file}")
    wav.write(output_file, int(sample_rate), stereo_data)

if __name__ == "__main__":
    try:
        metadata_path = get_filepath()
        conv_sigmf(metadata_path)
    except Exception as e:
        print(f"Error: {e}")
