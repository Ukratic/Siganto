"""Script de conversion mp3.
Expérimental : Dépendant de la méthode d'encodage"""

import sys
import os
from pydub import AudioSegment
import numpy as np
import scipy.io.wavfile as wav

def get_filepath():
    if len(sys.argv) < 2:
        print("Utilisation: python script.py <input_file.mp3>")
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Erreur : Vérifier chemin - {filepath}")
        sys.exit(1)

    return filepath

def conv_mp3(filepath):
    # Vérifie si le fichier est un mp3
    if not filepath.endswith('.mp3'):
        print("Erreur : Fichier non mp3.")
        return
    # Charge mp3 et conversion raw audio
    audio = AudioSegment.from_file(filepath, format="mp3")
    frame_rate = audio.frame_rate  # Fs
    # Determine bits par sample
    sample_width = audio.sample_width * 8  # Conversion en bits
    bit_depth_map = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    if sample_width not in bit_depth_map:
        print(f"Encodage non supporté : {sample_width} bits")
        return
    print(f"Bits par échantillon : {sample_width}")
    print(f"Fréquence d'échantillonnage : {frame_rate} Hz")
    # Extraction raw PCM & reshape avec split des canaux stereo (I/Q entrelacé)
    samples = np.array(audio.get_array_of_samples(),dtype=bit_depth_map[sample_width])
    samples = samples.reshape((-1, audio.channels))
    if audio.channels == 2:
        print("Fichier stereo : séparation des canaux I/Q")
        left, right = samples[:,0], samples[:,1]
    else:
        print("Duplication du canal mono en stereo")
        left, right = samples[:,0], samples[:,0]  # duplique mono
    # Combine canaux I (left) et Q (right) pour obtenir le signal IQ complexe
    iq_wave = left + 1j * right
    print(f"Durée du signal : {len(iq_wave) / frame_rate:.2f} secondes")
    # Normalisation
    max_amplitude = np.max(np.abs(iq_wave))
    if max_amplitude > 0:
        iq_data_normalized = iq_wave / max_amplitude
    else:
        iq_data_normalized = iq_wave
    # Formate en 16 bits et sépare reel/imaginaire
    print("Conversion en 2 voies 16 bits")
    real_part = (iq_data_normalized.real * 32767).astype(np.int16)
    imag_part = (iq_data_normalized.imag * 32767).astype(np.int16)
    # transforme reel+imag en 2 canaux pour le wav
    stereo_data = np.column_stack((real_part, imag_part))
    filename = filepath.split('.mp3')[0] + '.wav'
    wav.write(filename, frame_rate, stereo_data)
    print(f"Ecriture du nouveau wav. Sauvegardé : {filename}")

if __name__ == "__main__":
    try:
        filepath = get_filepath()
        conv_mp3(filepath)
    except Exception as e:
        print(f"Error: {e}")
