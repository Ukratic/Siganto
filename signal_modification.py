"""Fonctions de base pour filtrer, sous-échantillonner, sur-échantillonner, mesurer la largeur de bande"""

from scipy.signal import butter, filtfilt, resample, medfilt, wiener, firwin, lfilter, convolve, resample_poly
import numpy as np

def bandpass_filter(iq_sig, lowcut, highcut, samp_rate, order=4):
    """Filtre passe-bande, basé sur scipy.signal.butter et filtfilt"""
    nyquist = samp_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band') # Butterworth
    filtered_signal = filtfilt(b, a, iq_sig) # application du filtre en avant et en arrière pour éviter la distorsion de phase

    return filtered_signal

def lowpass_filter(iq_sig, highcut, samp_rate, order=4):
    """Filtre passe-bas, basé sur scipy.signal.butter et filtfilt"""
    nyquist = samp_rate / 2
    if not 0 < highcut < nyquist:
        raise ValueError("Fréquence de coupure doit être entre 0 et la fréquence de Nyquist.")
    high = highcut / nyquist
    b, a = butter(order, high, btype='low') # Butterworth
    filtered_signal = filtfilt(b, a, iq_sig) # application du filtre en avant et en arrière pour éviter la distorsion de phase

    return filtered_signal

def highpass_filter(iq_sig, lowcut, samp_rate, order=4):
    """Filtre passe-haut, basé sur scipy.signal.butter et filtfilt"""
    nyquist = samp_rate / 2
    if not 0 < lowcut < nyquist:
        raise ValueError("Fréquence de coupure doit être entre 0 et la fréquence de Nyquist.")
    low = lowcut / nyquist
    b, a = butter(order, low, btype='high') # Butterworth
    filtered_signal = filtfilt(b, a, iq_sig) # application du filtre en avant et en arrière pour éviter la distorsion de phase

    return filtered_signal

def downsample(iq_sig, samp_rate, decimation_factor):
    """Sous-échantillonnage par slicing.
    Avantages : simple, rapide, pas de dépendance externe.
    Inconvénients : aliasing possible si pas de filtrage préalable"""
    # facteur de décimation
    decimation_factor = int(decimation_factor)
    if decimation_factor < 1:
        raise ValueError("Le facteur de décimation doit être un entier positif.")
    if decimation_factor == 1:
        return iq_sig, samp_rate
    # sous-échantillonnage avec slicing
    downsampled_signal = iq_sig[::decimation_factor]
    new_samp_rate = int(samp_rate / decimation_factor)

    return downsampled_signal, new_samp_rate

# méthode de suréchantillonnage de base pour utilisation dans l'interface graphique
def upsample(iq_sig, samp_rate, oversampling_factor):
    """Suréchantillonnage FFT : zéro-padding dans le domaine fréquentiel"""
    # facteur de suréchantillonnage
    oversampling_factor = int(oversampling_factor)
    if oversampling_factor < 1:
        raise ValueError("Le facteur de suréchantillonnage doit être un entier positif.")
    if oversampling_factor == 1:
        return iq_sig, samp_rate
    upsampled_signal = resample(iq_sig, len(iq_sig) * oversampling_factor)
    new_samp_rate = int(samp_rate * oversampling_factor)

    return upsampled_signal, new_samp_rate

# méthode de rééchantillonnage par interpolation polyphasique : plus précise et efficace, mais plus complexe
def resample_polyphase(iq_sig, samp_rate, fs_new):
    """Resampling par interpolation polyphasique"""
    gcd = np.gcd(samp_rate, fs_new) # plus grand commun diviseur
    up = fs_new // gcd
    down = samp_rate // gcd
    resampled_signal = resample_poly(iq_sig, up, down) # interpolation polyphasique avec scipy

    return resampled_signal, fs_new

# méthode de rééchantillonnage par filtre CIC (Cascaded Integrator-Comb) : simple (cheap computing) mais moins précise. Inutile pour appli graphique avec traitement différé.
def resample_cic(iq_sig, samp_rate, fs_new):
    """Resampling par filtre CIC (Cascaded Integrator-Comb)"""
    ratio = fs_new / samp_rate
    n_out = int(len(iq_sig) * ratio)
    t = np.arange(n_out) / ratio
    i0 = np.floor(t).astype(int) # indices entiers
    i1 = np.minimum(i0 + 1, len(iq_sig) - 1) # indices suivants, avec gestion des bords
    frac = t - i0 # fraction pour interpolation linéaire
    resampled_signal = (1 - frac) * iq_sig[i0] + frac * iq_sig[i1] # interpolation linéaire

    return resampled_signal, fs_new

# Essai de quelques fonctions de filtrage supplémentaires
def median_filter(iq_sig, kernel_size=3):
    """Applique un filtre médian pour réduire le bruit"""
    if kernel_size < 1 or kernel_size % 2 == 0 or not isinstance(kernel_size, int):
        raise ValueError("La taille du noyau doit être un entier positif impair.")
    # Filtrage séparé des parties réelle et imaginaire
    real_part_filtered = medfilt(np.real(iq_sig), kernel_size=kernel_size)
    imag_part_filtered = medfilt(np.imag(iq_sig), kernel_size=kernel_size)
    filtered_complex_data = real_part_filtered + 1j * imag_part_filtered # reconstruction du signal complexe

    return filtered_complex_data

def moving_average(iq_sig, window_size):
    """Applique une moyenne mobile pour lisser le signal"""
    if window_size < 1 or not isinstance(window_size, int):
        raise ValueError("La taille de la fenêtre doit être un entier positif.")
    kernel = np.ones(window_size) / window_size
    # Application de la convolution

    return np.convolve(iq_sig, kernel, mode='same') # 'same' pour garder la même taille de sortie

def wiener_filter(iq_sig, size=None, noise=None):
    """Applique un filtre de Wiener pour réduire le bruit"""
    if size is not None and (not isinstance(size, int) or size < 1):
        raise ValueError("Size doit être un entier positif.")
    if noise is not None and (not isinstance(noise, (int, float)) or noise < 0):
        raise ValueError("noise doit être un nombre positif ou zéro.")
    
    return wiener(iq_sig, mysize=size, noise=noise) # application du filtre de Wiener de scipy

def fir_filter(iq_sig, fs, cutoff, filter_type='lowpass', numtaps=101):
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
    taps = firwin(numtaps, norm_cutoff, pass_zero=filter_type) # conception du filtre FIR

    return lfilter(taps, 1.0, iq_sig)

def matched_filter(iq_sig, samp_rate, symbol_rate, factor=0.5, pulse_shape='rectangular'):
    """Applique un filtre adapté pour un motif spécifique dans le signal"""
    sps = int(round(samp_rate / symbol_rate))
    if sps < 1:
        raise ValueError("Le taux d'échantillonnage par symbole (sps) doit être au moins 1.")
    # Calcul de la taille du noyau en fonction du facteur
    if pulse_shape == 'rectangular':
        kernel = np.ones(sps)
    elif pulse_shape == 'gaussian':
        t = np.arange(-5*sps, 5*sps+1)
        alpha = np.sqrt(np.log(2)) / (factor * sps)
        kernel = np.exp(-0.5 * (alpha * t)**2)
    elif pulse_shape == 'raised_cosine':
        span = 6
        t = np.arange(-span*sps, span*sps+1, dtype=float) / sps
        kernel = np.sinc(t)
        kernel *= np.cos(np.pi*factor*t) / (1 - (2*factor*t)**2 + 1e-12)
    elif pulse_shape == 'root_raised_cosine': # à évaluer
        span = 6
        t = np.arange(-span*sps, span*sps+1, dtype=float) / sps
        numerator = (np.sin(np.pi*t*(1-factor)) +
                     4*factor*t*np.cos(np.pi*t*(1+factor)))
        denominator = (np.pi*t*(1-(4*factor*t)**2))
        # évite la division par zéro
        mask = ~np.isclose(denominator, 0.0)
        kernel = np.zeros_like(t)
        kernel[mask] = numerator[mask] / denominator[mask]
        # Gestion des singularités
        kernel[np.isclose(t, 0.0)] = 1.0 - factor + 4*factor/np.pi
        kernel[np.isclose(np.abs(t), 1/(4*factor))] = (factor/np.sqrt(2)) * (
            ((1+2/np.pi)*np.sin(np.pi/(4*factor))) +
            ((1-2/np.pi)*np.cos(np.pi/(4*factor))))
    elif pulse_shape in ('sinc', 'rsinc'):
        span = 6
        t = np.arange(-span*sps, span*sps+1, dtype=float) / sps
        kernel = np.sinc(t)
        if pulse_shape == 'rsinc':
            kernel = np.sqrt(np.clip(kernel, 0, None))  # évite sqrt sur valeurs négatives. Peut altérer la forme.
    else:
        raise ValueError("Le filtre de mise en forme doit être 'rectangular', 'gaussian', 'raised_cosine', 'root_raised_cosine' ou 'sinc'.")
    # Inversion du noyau pour le filtrage adapté
    kernel = np.conjugate(kernel[::-1])
    # Normalisation du noyau
    kernel /= np.sqrt(np.sum(np.abs(kernel)**2))
    # Application du filtre adapté
    filtered_signal = convolve(iq_sig, kernel, mode='same')

    return filtered_signal

def hilbert(x):
    N = len(x)
    Xf = np.fft.fft(x) # transformée de Fourier du signal
    h = np.zeros(N) # vecteur de gain pour le filtre de Hilbert
    
    if N % 2 == 0:
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    
    Xf *= h  # garde freq positives, zero négatives
    x_analytic = np.fft.ifft(Xf) # transformée inverse pour obtenir le signal analytique
    
    return x_analytic

def doppler_lin_shift(iq_sig, samp_rate, start_freq, end_freq):
    """Applique un décalage Doppler linéaire au signal IQ.
    Méthode grossière : simule ou corrige un décalage Doppler linéaire"""
    n = len(iq_sig)
    t = np.arange(n) / samp_rate
    # Trajectoire de fréquence linéaire
    inst_freq = start_freq + (end_freq - start_freq) * (t / t[-1])
    # Phase = -2π ∫ f(t) dt
    phase = -2 * np.pi * np.cumsum(inst_freq) / samp_rate
    # Correction (ou simulation si start_freq/end_freq choisis arbitrairement)
    correction = np.exp(1j * phase)
    iq_out = iq_sig * correction
    
    return iq_out, inst_freq