"""Fonctions de base pour filtrer, sous-échantillonner, sur-échantillonner, mesurer la largeur de bande"""

from scipy.signal import butter, filtfilt, resample

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
