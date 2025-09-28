"""Fonctions de graphe principaux : spectrogramme, PSD + estim BW"""

import numpy as np
import dsp_funcs as df

# Spectrogramme
def compute_spectrogram(iq_wave, frame_rate, N, window_func='hann'):
    """Spectrogramme basé sur la FFT Cooley-Tukey (numpy.fft), cf PyDSP (Dr M. Lichtman), sans overlap"""
    # Calcule nb de lignes pour la matrice
    num_rows = len(iq_wave) // N
    spectrogram = np.zeros((num_rows, N))
    # Choix de fenetre
    window = df.get_window(window_func, N)
    # Calc spectrogramme
    for i in range(num_rows):
        chunk = iq_wave[i*N:(i+1)*N] * window
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2)
    # Bins fréquence et temps
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/frame_rate))
    times = np.arange(num_rows) * (N / frame_rate)

    return freqs, times, spectrogram

# STFT
def compute_stft(iq_wave, frame_rate, window_size, overlap, window_func='hann'):
    """Autre spectrogramme (base identique de FFT et fenêtrage) avec overlap"""
    # Steps
    step_size = window_size - overlap
    num_windows = (len(iq_wave) - overlap) // step_size
    # Choix de fenetre
    window = df.get_window(window_func, window_size)
    # Output arrays
    stft_matrix = []
    times = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        # Segmentation et fenêtrage
        segment = iq_wave[start:end]
        if len(segment) < window_size:
            segment = np.pad(segment, (0, window_size - len(segment)))
        segment = segment * window
        # Compute FFT
        fft_segment = np.fft.fft(segment)
        stft_matrix.append(fft_segment)
        times.append(start / frame_rate)

    stft_matrix = np.array(stft_matrix)
    try:
        stft_matrix = np.fft.fftshift(stft_matrix, axes=1)  # Orientation : Freq X, Temps Y
        freqs = np.fft.fftfreq(window_size, d=1/frame_rate)
    except:
        freqs = None

    return freqs, np.array(times), 20 * np.log10(np.abs(stft_matrix))

# DSP
def compute_dsp(iq_wave, frame_rate, N=256, overlap=128, window_type='hann'):
    """Densité spectrale de puissance"""
    step_size = N - overlap  # match STFT
    num_windows = (len(iq_wave) - overlap) // step_size
    psd_sum = np.zeros(N)
    window = df.get_window(window_type, N)
    # Loop fenêtres
    for i in range(num_windows):
        start = i * step_size
        end = start + N
        segment = iq_wave[start:end]
        segment = segment * window
        # Compute FFT**2
        fft_result = np.fft.fft(segment, N)
        psd_sum += np.abs(fft_result) ** 2
    # Avg puissance sur les fenêtres
    psd = psd_sum / num_windows
    psd /= (frame_rate * N)
    # Freq bins
    freqs = np.fft.fftfreq(N, d=1/frame_rate)

    return freqs, psd

# Mesure de la largeur de bande
def estimate_bandwidth(iq_wave, frame_rate, N=256, overlap=128, window_type='hann'):
    """Estimation de BW à partir de la DSP"""
    freqs, dsp = compute_dsp(iq_wave, frame_rate, N, overlap, window_type)
    # DSP normalisée pour le seuil de la largeur de bande
    dsp_normalisee = dsp / max(dsp)
    # Seuil de la largeur de bande : quart de la DSP normalisée
    mean_rsb_level = np.mean(dsp_normalisee)/2
    # Freqs au-dessus du seuil
    significant_freqs = freqs[dsp_normalisee > mean_rsb_level]
    min_freq = min(significant_freqs)
    max_freq = max(significant_freqs)
    bandwidth = max_freq - min_freq

    return bandwidth, min_freq, max_freq, freqs, 20*np.log10(dsp)
