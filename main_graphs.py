"""Fonctions de graphe principaux : spectrogramme, PSD + estim BW"""

import numpy as np
import dsp_funcs as df

# Spectrogramme
def compute_spectrogram(iq_wave, frame_rate, N, window_func='hann'):
    """Spectrogramme basé sur la FFT Cooley-Tukey (numpy.fft), cf PyDSP (Dr M. Lichtman), sans overlap"""
    # Calcule nb de lignes pour la matrice
    num_rows = len(iq_wave) // N
    spectrogram = np.zeros((num_rows, N))
    # Calc spectrogramme
    # Choix de fenetre
    if window_func == 'hann':
        window = np.hanning(N)
    elif window_func == 'hamming':
        window = np.hamming(N)
    elif window_func == 'blackman':
        window = np.blackman(N)
    elif window_func == 'kaiser':
        window = np.kaiser(N, beta=14)
    elif window_func == 'bartlett':
        window = np.bartlett(N)
    elif window_func == 'flattop':
        window = df.flattop_window(N)
    elif window_func == 'rect':
        window = np.ones(N)
    else:
        raise ValueError("Fenêtre non supportée")
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
    if window_func == 'hann':
        window = np.hanning(window_size)
    elif window_func == 'hamming':
        window = np.hamming(window_size)
    elif window_func == 'blackman':
        window = np.blackman(window_size)
    elif window_func == 'kaiser':
        window = np.kaiser(window_size, beta=14)
    elif window_func == 'bartlett':
        window = np.bartlett(window_size)
    elif window_func == 'flattop':
        window = df.flattop_window(window_size)
    elif window_func == 'rect':
        window = np.ones(window_size)
    else:
        raise ValueError("Fenêtre non supportée")
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
def compute_dsp(iq_wave, frame_rate, N, overlap=0.5):
    """Densité spectrale de puissance"""
    step_size = int(N * (1 - overlap))  # Step selon overlap
    num_windows = (len(iq_wave) - N) // step_size + 1  # Nb fenêtres
    psd_sum = np.zeros(N)
    # Loop fenêtres
    for i in range(num_windows):
        start = i * step_size
        end = start + N
        segment = iq_wave[start:end]
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
def estimate_bandwidth(iq_wave, frame_rate, N):
    """Estimation de BW à partir de la DSP"""
    freqs, dsp = compute_dsp(iq_wave, frame_rate, N)
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
