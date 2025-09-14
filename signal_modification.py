"""Fonctions de base pour filtrer, sous-échantillonner, sur-échantillonner, mesurer la largeur de bande"""

from scipy.signal import butter, filtfilt, resample, medfilt, wiener, firwin, lfilter, convolve
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
    real_part_filtered = medfilt(np.real(iq_wave), kernel_size=kernel_size)
    imag_part_filtered = medfilt(np.imag(iq_wave), kernel_size=kernel_size)
    filtered_complex_data = real_part_filtered + 1j * imag_part_filtered
    return filtered_complex_data

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

def matched_filter(iq_wave, frame_rate, symbol_rate, factor=0.5, pulse_shape='rectangular'):
    """Applique un filtre adapté pour un motif spécifique dans le signal"""
    sps = int(round(frame_rate / symbol_rate))
    if sps < 1:
        raise ValueError("Le taux d'échantillonnage par symbole (sps) doit être au moins 1.")
    # Calcul de la longueur du noyau en fonction du facteur
    if pulse_shape == 'rectangular':
        kernel = np.ones(sps)
    elif pulse_shape == 'gaussian':
        BT = factor
        t = np.arange(-3*sps, 3*sps+1)
        alpha = np.sqrt(np.log(2)) / (BT * sps)
        kernel = np.exp(-0.5 * (alpha * t)**2)
    elif pulse_shape == 'raised_cosine':
        # RC ou RRC
        beta = factor
        span = 6  # nombre de symboles
        t = np.arange(-span*sps, span*sps+1, dtype=float) / sps
        kernel = np.sinc(t)
        kernel *= np.cos(np.pi*beta*t) / (1 - (2*beta*t)**2 + 1e-12)  # formule RC
    elif pulse_shape == 'root_raised_cosine':
        beta = factor
        span = 6
        t = np.arange(-span*sps, span*sps+1, dtype=float) / sps
        numerator = (np.sin(np.pi*t*(1-beta)) +
                     4*beta*t*np.cos(np.pi*t*(1+beta)))
        denominator = (np.pi*t*(1-(4*beta*t)**2))
        # évite la division par zéro
        mask = ~np.isclose(denominator, 0.0)
        kernel = np.zeros_like(t)
        kernel[mask] = numerator[mask] / denominator[mask]

        # Gestion des singularités
        kernel[np.isclose(t, 0.0)] = 1.0 - beta + 4*beta/np.pi
        kernel[np.isclose(np.abs(t), 1/(4*beta))] = (beta/np.sqrt(2)) * (
            ((1+2/np.pi)*np.sin(np.pi/(4*beta))) +
            ((1-2/np.pi)*np.cos(np.pi/(4*beta))))
    elif pulse_shape in ('sinc', 'rsinc'):
        span = 6
        t = np.arange(-span*sps, span*sps+1, dtype=float) / sps
        kernel = np.sinc(t)
        if pulse_shape == 'rsinc':
            kernel = np.sqrt(np.clip(kernel, 0, None))  # évite sqrt sur valeurs négatives
    else:
        raise ValueError("Le filtre de mise en forme doit être 'rectangular', 'gaussian', 'raised_cosine', 'root_raised_cosine' ou 'sinc'.")
    # Inversion du noyau pour le filtrage adapté
    kernel = np.conjugate(kernel[::-1])
    # Normalisation du noyau
    kernel /= np.sqrt(np.sum(np.abs(kernel)**2))
    # Application du filtre adapté
    filtered_signal = convolve(iq_wave, kernel, mode='same')
    return filtered_signal

def hilbert(x):
    N = len(x)
    Xf = np.fft.fft(x)
    h = np.zeros(N)
    
    if N % 2 == 0:
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    
    Xf *= h  # garde freq positives, zero négatives
    x_analytic = np.fft.ifft(Xf)
    
    return x_analytic