"""Fonctions de base pour filtrer, sous-échantillonner, sur-échantillonner, mesurer la largeur de bande"""

from scipy.signal import butter, filtfilt, resample, medfilt, wiener, firwin, lfilter, convolve
from scipy.signal.windows import gaussian
import numpy as np

def bandpass_filter(iq_wave, lowcut, highcut, frame_rate, order=4):
    """Filtre passe-bande, basé sur scipy.signal.butter et filtfilt"""
    nyquist = frame_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, iq_wave)

    return filtered_signal

def lowpass_filter(iq_wave, highcut, frame_rate, order=4):
    """Filtre passe-bas, basé sur scipy.signal.butter et filtfilt"""
    nyquist = frame_rate / 2
    if not 0 < highcut < nyquist:
        raise ValueError("Fréquence de coupure doit être entre 0 et la fréquence de Nyquist.")
    high = highcut / nyquist
    b, a = butter(order, high, btype='low')
    filtered_signal = filtfilt(b, a, iq_wave)

    return filtered_signal

def highpass_filter(iq_wave, lowcut, frame_rate, order=4):
    """Filtre passe-haut, basé sur scipy.signal.butter et filtfilt"""
    nyquist = frame_rate / 2
    if not 0 < lowcut < nyquist:
        raise ValueError("Fréquence de coupure doit être entre 0 et la fréquence de Nyquist.")
    low = lowcut / nyquist
    b, a = butter(order, low, btype='high')
    filtered_signal = filtfilt(b, a, iq_wave)

    return filtered_signal

def downsample(iq_wave, frame_rate, decimation_factor):
    """Sous-échantillonnage par slicing"""
    # facteur de décimation
    decimation_factor = int(decimation_factor)
    if decimation_factor < 1:
        raise ValueError("Le facteur de décimation doit être un entier positif.")
    if decimation_factor == 1:
        return iq_wave, frame_rate
    # sous-échantillonnage avec slicing
    downsampled_signal = iq_wave[::decimation_factor]
    new_frame_rate = int(frame_rate / decimation_factor)

    return downsampled_signal, new_frame_rate

def upsample(iq_wave, frame_rate, oversampling_factor):
    """Suréchantillonnage, basé sur scipy.signal.resample"""
    # facteur de suréchantillonnage
    oversampling_factor = int(oversampling_factor)
    if oversampling_factor < 1:
        raise ValueError("Le facteur de suréchantillonnage doit être un entier positif.")
    if oversampling_factor == 1:
        return iq_wave, frame_rate
    upsampled_signal = resample(iq_wave, len(iq_wave) * oversampling_factor)
    new_frame_rate = int(frame_rate * oversampling_factor)

    return upsampled_signal, new_frame_rate

# Essai de quelques fonctions de filtrage supplémentaires
def median_filter(iq_wave, kernel_size=3):
    """Applique un filtre médian pour réduire le bruit"""
    if kernel_size < 1 or kernel_size % 2 == 0 or not isinstance(kernel_size, int):
        raise ValueError("La taille du noyau doit être un entier positif impair.")
        # fix complex128 not handled by medfilt
    if iq_wave.dtype == np.complex128:
        iq_wave = iq_wave.astype(np.float64)
    return medfilt(iq_wave, kernel_size=kernel_size)

def moving_average(iq_wave, window_size):
    """Applique une moyenne mobile pour lisser le signal"""
    if window_size < 1 or not isinstance(window_size, int):
        raise ValueError("La taille de la fenêtre doit être un entier positif.")
    kernel = np.ones(window_size) / window_size
    return np.convolve(iq_wave, kernel, mode='same')

def wiener_filter(iq_wave, size=None, noise=None):
    """Applique un filtre de Wiener pour réduire le bruit"""
    if size is not None and (not isinstance(size, int) or size < 1):
        raise ValueError("Size doit être un entier positif.")
    if noise is not None and (not isinstance(noise, (int, float)) or noise < 0):
        raise ValueError("noise doit être un nombre positif ou zéro.")
    return wiener(iq_wave, mysize=size, noise=noise)

def fir_filter(iq_wave, fs, cutoff, filter_type='lowpass', numtaps=101):
    """Applique un filtre FIR (Finite Impulse Response) pour filtrer le signal IQ"""
    if filter_type not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError("filter_type doit être 'lowpass', 'highpass', 'bandpass' ou 'bandstop'.")
    if isinstance(cutoff, (list, tuple)):
        if len(cutoff) != 2 and filter_type == 'bandpass':
            raise ValueError("Pour un filtre passe-bande, cutoff doit être une liste de deux fréquences.")
        elif len(cutoff) != 1 and filter_type in ['lowpass', 'highpass']:
            raise ValueError("Pour un filtre passe-bas ou passe-haut, cutoff doit être une liste d'une seule fréquence.")
    if numtaps < 1 or not isinstance(numtaps, int):
        raise ValueError("numtaps doit être un entier positif.")
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq if np.isscalar(cutoff) else [c / nyq for c in cutoff]
    taps = firwin(numtaps, norm_cutoff, pass_zero=filter_type)
    return lfilter(taps, 1.0, iq_wave)

def gaussian_filter(iq_wave, sigma):
    """Applique un filtre gaussien pour lisser le signal IQ"""
    if sigma <= 0:
        raise ValueError("sigma doit être un nombre positif.")
    M = int(6 * sigma + 1)  # Taille du noyau
    if M % 2 == 0:  # Assurez-vous que M est impair
        M += 1
    if M < 1:
        raise ValueError("La taille du noyau doit être au moins 1.")
    kernel = gaussian(M, std=sigma)
    kernel /= np.sum(kernel)  # Normalisation du noyau
    
    return convolve(iq_wave, kernel, mode='same')

def matched_filter(iq_wave, template):
    """Applique un filtre adapté pour détecter un motif spécifique dans le signal IQ"""
    if not isinstance(template, (list, np.ndarray)):
        raise ValueError("template doit être une liste ou un tableau numpy.")
    if len(template) == 0:
        raise ValueError("template ne peut pas être vide.")
    from scipy.signal import correlate
    correlation = correlate(iq_wave, template, mode='same')
    return correlation / np.max(np.abs(correlation))  # Normalisation



# A ajouter dans gui_main.py pour les options de filtrage:
# Filtre médian : OK
# Filtre moyenne mobile (taille fenêtre) : 5 = Léger, 11 = Modéré, 21 = Fort. Dynamique = int(sr / symbol_rate)
# Filtre de Wiener (taille fenêtre) : 7 = Léger, 15 = Modéré, 31 = Fort. Dynamique = int(sr / symbol_rate) * facteur (0.5 à 2.0)
# Filtre FIR (Taps) : 11 = Léger, 31 = Modéré, 61 = Fort. Dynamique = 4 * (sr / symbol_rate) + 1
# Filtre gaussien (sigma) : 0.5 = Léger, 1.5 = Modéré, 3.0 = Fort. Dynamique = (sr / symbol_rate) / 4