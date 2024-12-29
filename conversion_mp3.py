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
    # Charge mp3 et conversion raw audio
    audio = AudioSegment.from_file(filepath, format="mp3")
    frame_rate = audio.frame_rate  # Fs
    # Verif présence 1 ou 2 canaux (cas 1 peut générer des résultats incohérents)
    if audio.channels == 1:
        print("Fichier mp3 en mono. Canal dupliqué pour recréer I/Q")
        audio = AudioSegment.from_mono_audiosegments(audio, audio)
    elif audio.channels != 2:
        print("Format non reconnu de mp3 autre que stereo/mono.")
        return
    # Extraction raw PCM
    raw_data = np.array(audio.get_array_of_samples())
    # Determine bits par sample
    sample_width = audio.sample_width * 8  # Conversion en bits
    bit_depth_map = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    n_bits = bit_depth_map.get(sample_width, np.int32)
    # Reshape avec split des canaux stereo (I/Q entrelacé)
    iq_wave = np.frombuffer(raw_data, dtype=n_bits)
    left, right = iq_wave[0::2], iq_wave[1::2]
    # Combine canaux I (left) et Q (right) pour obtenir le signal IQ complexe
    try:
        iq_wave = left + 1j * right
    except:
        # retire quelques échantillons pour éviter les erreurs si nécessaire
        iq_wave = left[:min(len(left), len(right))] + 1j * right[:min(len(left), len(right))]
        print("Error converting IQ. Trimmed samples to match I and Q.")
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