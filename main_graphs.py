"""Fonctions de graphe principaux : spectrogramme, PSD + estim BW"""

import numpy as np
from dsp_funcs import get_window

# Spectrogramme simple (FFT sans overlap)
def compute_spectrogram(iq_sig, samp_rate, N, window_func='hann'):
    """Spectrogramme basé sur la FFT Cooley-Tukey (numpy.fft), 
    cf PyDSP (Dr M. Lichtman), sans overlap"""
    # Calcule nb de lignes pour la matrice
    num_rows = len(iq_sig) // N
    spectrogram = np.zeros((num_rows, N)) # Initialisation matrice
    # Choix de fenetre
    window = get_window(window_func, N)
    # Calc spectrogramme
    for i in range(num_rows):
        chunk = iq_sig[i*N:(i+1)*N] * window # Segmentation + fenêtrage
        # FFT + Magnitude + dB + Shift
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2) 
    # Bins fréquence et temps
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/samp_rate))
    times = np.arange(num_rows) * (N / samp_rate)

    return freqs, times, spectrogram

# STFT
def compute_stft(iq_sig, samp_rate, window_size, overlap, window_func='hann', nfft=None):
    """Spectrogramme via STFT (Short-Time Fourier Transform) avec recouvrement"""
    if nfft is None:
        nfft = window_size * 4
    elif nfft < window_size:
        raise ValueError(f"nfft ({nfft}) doit être >= window_size ({window_size})")
    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) doit être < window_size ({window_size})")

    num_windows = (len(iq_sig) - overlap) // (window_size - overlap)
    if num_windows <= 0:
        raise ValueError(f"Signal trop court ({len(iq_sig)} échantillons) pour window_size={window_size}")
    # Choix de fenetre
    window = get_window(window_func, window_size)
    # Padding pour améliorer la qualité visuelle (smoothing) si le signal est petit
    if num_windows * nfft > 5e8: # Si le spectrogramme est très grand, sans padding
        n_bins = window_size
    else:
        n_bins = nfft
    # Initialisation matrice STFT et vecteur temps
    stft_matrix = np.empty((num_windows, n_bins), dtype=np.float32) # Orientation : Freq X, Temps Y
    times = np.empty(num_windows, dtype=np.float64)
    # Loop sur les fenêtres
    for i in range(num_windows):
        start = i * (window_size - overlap) # indices
        end = start + window_size
        # Segmentation et fenêtrage
        segment = iq_sig[start:end]
        if len(segment) < window_size: # Padding si la dernière fenêtre est plus courte que window_size
            segment = np.pad(segment, (0, window_size - len(segment)))
        segment = segment * window
        # Compute FFT
        fft_segment = np.fft.fft(segment, n=n_bins)
        stft_matrix[i, :] = np.fft.fftshift(20.0 * np.log10(np.abs(fft_segment) + 1e-12)) # Magnitude en dB (en relatif de la fenêtre)
        times[i] = (start + window_size / 2.0) / samp_rate

    # stft_matrix = np.fft.fftshift(stft_matrix, axes=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / samp_rate))

    return freqs, times, stft_matrix

# DSP
def compute_dsp(iq_sig, samp_rate, N=256, overlap=128, window_type='hann', nfft=None):
    """Densité spectrale de puissance (PSD) via méthode moyenne périodogramme (Welch)"""
    if nfft is None:
        nfft = N * 2
    elif nfft < N:
        raise ValueError(f"nfft ({nfft}) doit être >= N ({N})")
    if overlap >= N:
        raise ValueError(f"overlap ({overlap}) doit être < N ({N})")
    
    num_windows = (len(iq_sig) - overlap) // (N - overlap)
    if num_windows <= 0:
            raise ValueError(f"Signal trop court ({len(iq_sig)} échantillons) pour N={N}")
    psd_sum = np.zeros(nfft)
    window = get_window(window_type, N)
    # Loop fenêtres
    for i in range(num_windows):
        start = i * (N - overlap)
        end = start + N
        segment = iq_sig[start:end]
        segment = segment * window
        # Compute FFT**2
        fft_result = np.fft.fft(segment, n=nfft)
        psd_sum += np.abs(fft_result) ** 2
    # Avg puissance sur les fenêtres
    psd = psd_sum / num_windows
    psd /= samp_rate
    # Freq bins
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/samp_rate))
    psd /= np.sum(window**2)  # Correction fenêtrage
    psd = np.fft.fftshift(psd)  # Shift pour centrer la fréquence 0

    return freqs, psd

# Mesure de la largeur de bande
def estimate_bandwidth(iq_sig=None, samp_rate=None, N=None, overlap=None, window_type=None, psd=None,freqs=None):
    """Estimation de BW à partir de la DSP"""
    if (iq_sig is None or samp_rate is None or N is None or overlap is None or window_type is None) and (psd is None or freqs is None):
        raise ValueError("iq_sig, samp_rate, N, overlap et window_type doivent être fournis si psd et freqs ne sont pas fournis")
    if psd is None or freqs is None:
        freqs, psd = compute_dsp(iq_sig, samp_rate, N, overlap, window_type)
    # DSP normalisée pour le seuil de la largeur de bande
    dsp_normalisee = psd / psd.max()
    # Seuil de la largeur de bande : demi-moyenne de la DSP normalisée
    half_mean_level = np.mean(dsp_normalisee)/2
    # Freqs au-dessus du seuil
    significant_freqs = freqs[dsp_normalisee > half_mean_level]
    min_freq = significant_freqs.min()
    max_freq = significant_freqs.max()
    bandwidth = max_freq - min_freq

    return bandwidth, min_freq, max_freq, freqs, 10*np.log10(psd) # DSP en dB/Hz
