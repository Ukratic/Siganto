"""Fonctions DSP annexes"""
import numpy as np
from scipy.signal import find_peaks, fftconvolve

# Flattop window
def flattop_window(N):
    """Fenêtre flattop pour la FFT"""
    if N < 1:
        return np.array([])

    # Coefficients pour la fenêtre Flattop (source : implémentation Octave de Flattop Harris)
    a0 = 1.0
    a1 = 1.93
    a2 = 1.29
    a3 = 0.388
    a4 = 0.0322

    n = np.arange(0, N)
    w = (
        a0
        - a1 * np.cos(2 * np.pi * n / (N - 1))
        + a2 * np.cos(4 * np.pi * n / (N - 1))
        - a3 * np.cos(6 * np.pi * n / (N - 1))
        + a4 * np.cos(8 * np.pi * n / (N - 1))
    )

    return w / np.max(w)

# Blackman-Harris 7-term window
def blackman_harris_7term_window(N):
    """Fenêtre 7-term Blackman-Harris ()"""
    if N < 1:
        return np.array([])

    # Coefficients pour la fenêtre Blackman-Harris 7-term (source : National Instruments, H.H. Albrecht)
    a0 = 0.27105140069342
    a1 = 0.43329793923448
    a2 = 0.21812299954311
    a3 = 0.06592544638803
    a4 = 0.01081174209837
    a5 = 0.00077658482522
    a6 = 0.00001388721735

    n = np.arange(0, N)
    w = (
        a0
        - a1 * np.cos(2 * np.pi * n / (N - 1))
        + a2 * np.cos(4 * np.pi * n / (N - 1))
        - a3 * np.cos(6 * np.pi * n / (N - 1))
        + a4 * np.cos(8 * np.pi * n / (N - 1))
        - a5 * np.cos(10 * np.pi * n / (N - 1))
        + a6 * np.cos(12 * np.pi * n / (N - 1))
    )

    return w / np.max(w)

def gaussian_window(N, sigma=0.4):
    """Fenêtre gaussienne pour la FFT"""
    if N < 1:
        return np.array([])
    n = np.arange(0, N)
    return np.exp(-0.5 * ((n - (N-1)/2) / (sigma * (N-1)/2))**2)

def center_signal(iq_wave, frame_rate, prominence=0.1):
    """Fonction de centrage du signal sur pics proéminents"""
    N = len(iq_wave)
    t = np.arange(N) / frame_rate

    spectrum = np.fft.fftshift(np.fft.fft(iq_wave))
    f = np.fft.fftfreq(N, 1/frame_rate)
    f = np.fft.fftshift(f)

    peaks_indices, _ = find_peaks(np.abs(spectrum), prominence=prominence)  # indices des pics

    if len(peaks_indices) < 2:
        print("Moins de 2 pics trouvés pour recentrer le signal.")
        return None

    peak_freqs = f[peaks_indices]
    # Sort par puissance
    sorted_indices = np.argsort(np.abs(spectrum[peaks_indices]))[::-1]
    top_2_indices = sorted_indices[:2]
    peak1_freq = peak_freqs[top_2_indices[0]]
    peak2_freq = peak_freqs[top_2_indices[1]]
    # calc & freq shift
    center_freq = (peak1_freq + peak2_freq) / 2
    iq_shifted = iq_wave * np.exp(-1j * 2 * np.pi * center_freq * t)

    return iq_shifted, center_freq

def get_window(window_type, N):
    """Retourne la fenêtre appropriée selon le type"""
    if window_type == 'flattop':
        return flattop_window(N)
    elif window_type == 'blackmanharris7term':
        return blackman_harris_7term_window(N)
    elif window_type == 'hann':
        return np.hanning(N)
    elif window_type == 'hamming':
        return np.hamming(N)
    elif window_type == 'blackman':
        return np.blackman(N)
    elif window_type == 'kaiser':
        return np.kaiser(N, beta=14)
    elif window_type == 'bartlett':
        return np.bartlett(N)
    elif window_type == 'rectangular':
        return np.ones(N)
    elif window_type == 'gaussian':
        return gaussian_window(N)
    else:
        raise ValueError(f"Type de fenêtre inconnu : {window_type}")    
    
def morlet_wavelet(t, s, w=6.0):
    """ Génère une ondelette Morlet dans le domaine temporel.
    t : tableau de temps centré à 0
    s : échelle
    w : paramètre de fréquence centrale de Morlet"""
    wave = np.exp(1j * w * t / s) * np.exp(-0.5 * (t / s)**2)
    wave /= np.sqrt(np.sum(np.abs(wave)**2))  # normalisation de l'énergie
    return wave

def morlet_cwt(iq_wave, fs, fmin=None, fmax=None, nfreq=96, w=6.0):
    """ CWT Morlet avec convolution linéaire basée sur FFT.
    coefs : coefficients complexes (nfreq x len(iq_wave))
    center_freqs : fréquences centrales de Morlet en pi rad/échantillon
    """

    x = np.asarray(iq_wave)
    N = len(x)
    dt = 1.0 / fs

    if fmin is None:
        fmin = fs / N      # résolution en fréquence
    if fmax is None:
        fmax = fs / 2.0    # Nyquist

    # Calcul des échelles
    freqs = np.geomspace(fmin, fmax, nfreq)
    scales = w / (2 * np.pi * freqs * dt)

    coefs = []
    for s in scales:
        M = int(np.ceil(12 * s))
        t = np.arange(-M//2, M//2 + 1)
        wave = morlet_wavelet(t, s, w)
        C = fftconvolve(x, wave, mode='same')
        C /= np.sqrt(s)  # normalisation par l'échelle
        coefs.append(C)

    coefs = np.vstack(coefs)
    center_freqs = w / scales  # radians/échantillon
    center_freqs_pi = center_freqs / np.pi # en pi rad/sample pour cohérence avec d'autres outils

    return coefs, center_freqs_pi