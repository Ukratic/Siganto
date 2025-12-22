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

def estimate_carrier_weighted(x, fs):
    """Max vraisemblance : estimateur de fréquence porteuse avec mag**2 pondérée."""
    x = x - np.mean(x)

    mag2 = np.abs(x)**2
    dphi = np.angle(x[1:] * np.conj(x[:-1])) # différence de phase

    weights = np.minimum(mag2[1:], mag2[:-1]) # pondération par mag² minimale
    f_est = fs / (2 * np.pi) * np.sum(weights * dphi) / np.sum(weights) # estimation de la fréquence
    t = np.arange(len(x)) / fs
    iq_shifted = x * np.exp(-1j * 2 * np.pi * f_est * t) # recentrage du signal

    return iq_shifted, f_est

def center_signal(iq_sig, samp_rate, prominence=0.1):
    """Fonction de centrage du signal sur pics proéminents"""
    sig_len = len(iq_sig)
    t = np.arange(sig_len) / samp_rate
    spectrum = np.fft.fftshift(np.fft.fft(iq_sig))
    f = np.fft.fftfreq(sig_len, 1/samp_rate)
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
    iq_shifted = iq_sig * np.exp(-1j * 2 * np.pi * center_freq * t)

    return iq_shifted, center_freq

def get_window(window_type, N):
    """Retourne la fenêtre appropriée selon le type"""
    if window_type == 'flattop': # précision en amplitude
        return flattop_window(N)
    elif window_type == 'blackmanharris7term': # réjection de lobes secondaires et haute dynamique, bon compromis
        return blackman_harris_7term_window(N)
    elif window_type == 'hann': # cas général
        return np.hanning(N)
    elif window_type == 'hamming': # autre cas général adapté audio
        return np.hamming(N)
    elif window_type == 'blackman': # réjection de lobes secondaires
        return np.blackman(N)
    elif window_type == 'kaiser': # trade-off adaptatif
        return np.kaiser(N, beta=14) # beta=14 pour un bon compromis
    elif window_type == 'bartlett': # triangulaire, lissage simple
        return np.bartlett(N)
    elif window_type == 'rectangular': # pas de fenêtre
        return np.ones(N)
    elif window_type == 'gaussian': # fenêtre gaussienne, réduction de bruit
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

def morlet_cwt(iq_sig, fs, fmin=None, fmax=None, nfreq=96, w=6.0):
    """ CWT Morlet avec convolution linéaire basée sur FFT.
    coefs : coefficients complexes (nfreq x len(iq_sig))
    center_freqs : fréquences centrales de Morlet en pi rad/échantillon
    """

    x = np.asarray(iq_sig)
    N = len(x)
    dt = 1.0 / fs

    if fmin is None:
        fmin = fs / N # résolution en fréquence
    if fmax is None:
        fmax = fs / 2.0 # Nyquist

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

    coefs = np.vstack(coefs) # shape (nfreq, N)
    center_freqs = w / scales  # radians/échantillon
    center_freqs_pi = center_freqs / np.pi # en pi rad/sample pour cohérence avec d'autres outils

    return coefs, center_freqs_pi