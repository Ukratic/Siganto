"""
Fonctions pour estimer les paramètres de modulation d'un signal IQ.
- Fonctions de base pour filtrer, sous-échantillonner, sur-échantillonner, mesurer la largeur de bande
- Fonctions pour mesurer la rapidité de modulation, la phase, l'autocorrelation
- Fonctions pour mesurer les paramètres de modulation OFDM : ## Auteur : FX Socheleau - IMT Atlantique, 2020 ##
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
##
# Fonctions de base pour filtrer, sous-échantillonner, sur-échantillonner, mesurer la largeur de bande
##

def bandpass_filter(iq_wave, lowcut, highcut, frame_rate, order=4):
    # Butterworth bandpass
    nyquist = frame_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, iq_wave)

    return filtered_signal

def lowpass_filter(iq_wave, highcut, frame_rate, order=4):
    nyquist = frame_rate / 2
    if not (0 < highcut < nyquist):
        raise ValueError("Fréquence de coupure doit être entre 0 et la fréquence de Nyquist.")
    high = highcut / nyquist
    b, a = butter(order, high, btype='low')
    filtered_signal = filtfilt(b, a, iq_wave)

    return filtered_signal

def highpass_filter(iq_wave, lowcut, frame_rate, order=4):
    nyquist = frame_rate / 2
    if not (0 < lowcut < nyquist):
        raise ValueError("Fréquence de coupure doit être entre 0 et la fréquence de Nyquist.")
    low = lowcut / nyquist
    b, a = butter(order, low, btype='high')
    filtered_signal = filtfilt(b, a, iq_wave)

    return filtered_signal

def downsample(iq_wave, frame_rate, decimation_factor):
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
    # facteur de suréchantillonnage
    oversampling_factor = int(oversampling_factor)
    if oversampling_factor < 1:
        raise ValueError("Le facteur de suréchantillonnage doit être un entier positif.")
    if oversampling_factor == 1:
        return iq_wave, frame_rate
    # Suréchantillonnage avec scipy.signal.resample
    upsampled_signal = resample(iq_wave, len(iq_wave) * oversampling_factor)
    new_frame_rate = int(frame_rate * oversampling_factor)

    return upsampled_signal, new_frame_rate

# Spectrogramme
def compute_spectrogram(iq_wave, frame_rate, N):
    # Calcule nb de lignes pour la matrice
    num_rows = len(iq_wave) // N
    spectrogram = np.zeros((num_rows, N))
    # Calc spectrogramme
    for i in range(num_rows):
        segment = iq_wave[i * N:(i + 1) * N]
        fft_result = np.fft.fftshift(np.fft.fft(segment))
        spectrogram[i, :] = 10 * np.log10(np.abs(fft_result)**2)
    # Bins fréquence et temps
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/frame_rate))
    times = np.arange(num_rows) * (N / frame_rate)
    
    return freqs, times, spectrogram

# STFT
def compute_stft(iq_wave, frame_rate, window_size, overlap, window_func='hann'):
    # Steps
    step_size = window_size - overlap
    num_windows = (len(iq_wave) - overlap) // step_size
    # Choix de fenetre
    if window_func == 'hann':
        window = np.hanning(window_size)
    elif window_func == 'hamming':
        window = np.hamming(window_size)
    else:
        raise ValueError("Fenêtre non supportée") # ajouter peut-être plus tard Bartlett, etc...
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
    stft_matrix = np.fft.fftshift(stft_matrix, axes=1)  # Orientation : Freq X, Temps Y
    freqs = np.fft.fftfreq(window_size, d=1/frame_rate)
    
    return freqs, np.array(times), 20 * np.log10(np.abs(stft_matrix))

# DSP
def compute_dsp(iq_wave, frame_rate, N, overlap=0.5):
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

##
# Fonctions mesures de la rapidité de modulation
##

def power_spectrum_fft(iq_wave, frame_rate):
    spectrum = np.abs(iq_wave) ** 2
    spectrum_fft = np.fft.fft(spectrum, len(iq_wave))
    spectrum_fft = np.fft.fftshift(spectrum_fft)
    # fréquences de -frame_rate/2 à frame_rate/2 pour l'affichage du spectre
    f = np.linspace(frame_rate/-2, frame_rate/2, len(iq_wave))
    # ignorer le pic autour de 0hz pour la détection du pic de fréquence mais le conserver pour l'analyse
    discard_dc = np.abs(spectrum_fft)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0 # ignorer 10 indices autour de 0hz
    peak_freq_index = np.argmax(discard_dc)  # Index du pic de fréquence
    peak_freq = f[peak_freq_index]

    return spectrum_fft, f, peak_freq

def mean_threshold_spectrum(iq_wave, frame_rate):
    # seuil de la moyenne et calcul des pulses pour la détection de la fréquence de modulation
    midpoint = iq_wave > np.mean(iq_wave)
    pulse = np.diff(midpoint)**2
    clock = np.fft.fft(pulse, len(iq_wave))
    clock = np.fft.fftshift(clock)
    # ensuite, idem que power_spectrum_fft
    f = np.linspace(frame_rate/-2, frame_rate/2, len(iq_wave))
    discard_dc = np.abs(clock)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_freq = f[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_freq = 0

    return clock, f, peak_freq

def power_series(iq_wave, frame_rate):
    f = np.linspace(frame_rate/-2, frame_rate/2, len(iq_wave))
    # puissance du signal au carré
    samples_squared = iq_wave**2
    squared_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_squared)))/len(iq_wave)
    squared_metric[len(squared_metric)//2] = 0 # retire la composante DC, c'est-à-dire la moyenne du signal.
    peak_squared_index = (-squared_metric).argsort()[:2]
    peak_squared_freq1,peak_squared_freq2 = f[peak_squared_index[0]],f[peak_squared_index[1]]
    peak_squared_freq = [peak_squared_freq1,peak_squared_freq2]
    # puissance du signal à la puissance 4
    samples_quartic = iq_wave**4
    quartic_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_quartic)))/len(iq_wave)
    quartic_metric[len(quartic_metric)//2] = 0
    peak_quartic_index = (-quartic_metric).argsort()[:2]
    peak_quartic_freq1,peak_quartic_freq1 = f[peak_quartic_index[0]],f[peak_quartic_index[1]]  
    peak_quartic_freq = [peak_quartic_freq1,peak_quartic_freq1]

    return f, squared_metric, quartic_metric, peak_squared_freq, peak_quartic_freq

##
# Fonctions de mesures temporelles
##

# Spectre de persistance
def persistance_spectrum(iq_wave, frame_rate, N, power_bins=50):
    # spectrogramme
    num_rows = len(iq_wave) // N
    spectrogram = np.zeros((num_rows, N))
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq_wave[i * N:(i + 1) * N]))) ** 2)
    min_power = np.min(spectrogram)
    # Si min_power = -inf, remplacer par la valeur minimale (sinon erreur matplotlib)
    if min_power == -np.inf:
        min_power = np.min(spectrogram[np.isfinite(spectrogram)])
    max_power = np.max(spectrogram)
    # définition de la persistance
    power_edges = np.linspace(min_power, max_power, power_bins + 1)
    persistence = np.zeros((N, power_bins))
    # histogramme de puissance
    for i, row in enumerate(spectrogram.T):
        hist, _ = np.histogram(row, bins=power_edges)
        persistence[i, :] = hist
    # normalisation
    persistence = persistence / np.max(persistence)
    f= np.fft.fftshift(np.fft.fftfreq(N, 1 / frame_rate))
    
    return f, min_power, max_power, persistence

# Visualiser la phase dans le domaine temporel
def phase_time_angle(iq_wave, frame_rate, window_size=5):
    # Calcul de la phase
    phase = np.angle(iq_wave)
    # Lissage de la phase
    if window_size > 1:
        window = np.ones(window_size) / window_size
        smoothed_phase = np.convolve(phase, window, mode='same')
        time = np.arange(0, len(smoothed_phase)) / frame_rate
    else:
        smoothed_phase = phase
        time = np.arange(0, len(phase)) / frame_rate

    return time, phase

# Mesures de phase cumulée
def phase_cumulative_distribution(iq_wave, num_bins=250):
    # Calcul de la phase cumulée
    phase = np.angle(iq_wave)
    # Histogramme de la phase cumulée avec les valeurs de phase positives
    hist, bins = np.histogram(phase, bins=num_bins, density=True)
    # Filtrer les valeurs de phase négatives
    hist = hist[bins[:-1] > 0]
    bins = bins[:-1][bins[:-1] > 0]

    return hist, bins

def frequency_transitions(iq_wave, frame_rate, window_size=5):
    # Phase instantanee
    phase = np.unwrap(np.angle(iq_wave))
    # Freq instantanee derivee de la phase
    inst_freq = np.diff(phase) / (2 * np.pi) * frame_rate
    if window_size > 1:
        window = np.ones(window_size) / window_size
        smoothed_freq = np.convolve(inst_freq, window, mode='same')
        time = np.arange(0, len(smoothed_freq)) / frame_rate
    else:
        smoothed_freq = inst_freq
        time = np.arange(0, len(smoothed_freq)) / frame_rate

    return time, inst_freq

##
# ACF 
##

# Autocorrelation rapide
def autocorrelation(iq_wave, frame_rate=None):
    n = len(iq_wave)
    n_padded = 2**np.ceil(np.log2(2 * n - 1)).astype(int)
    # FFT du signal pour l'autocorrelation
    fft_iq_wave = np.fft.fft(iq_wave, n=n_padded)
    yx = np.fft.ifft(fft_iq_wave * np.conj(fft_iq_wave)).real  # Resultat d'autocorrelation
    # Trim : valeurs absolues
    yx = yx[:n]
    yx = np.abs(yx)
    # Compute lags à partir de 0
    lags = np.arange(0, n)
    # Convertir lags en temps
    if frame_rate is not None:
        lags = lags / frame_rate
    
    return yx, lags

def autocorrelation_peak(iq_wave, frame_rate, min_distance=1):
    # Fonction d'autocorrelation
    yx, lags = autocorrelation(iq_wave, frame_rate)
    # Trouver le pic en excluant zéro
    yx[0] = 0
    peak_index = np.argmax(yx[min_distance:]) + min_distance
    peak_value = yx[peak_index]
    lag_time_ms = (lags[peak_index] * 1000)  # s -> ms

    return peak_value, lag_time_ms

def autocorrelation_peak_from_acf(yx, lags, min_distance=1):
    # Trouver le pic en excluant zéro
    yx[0] = 0
    peak_index = np.argmax(yx[min_distance:]) + min_distance
    peak_value = yx[peak_index]
    lag_time_ms = (lags[peak_index] * 1000)  # s -> ms

    return peak_value, lag_time_ms

# Autocorrelation complète (lente)
def full_autocorrelation(iq_wave):
    yx, lags = np.correlate(iq_wave, iq_wave, mode='full'), np.arange(-len(iq_wave)+1, len(iq_wave))
    # les valeurs négatives de lags sont ignorées = doublons
    yx = yx[len(iq_wave)-1:]
    lags = lags[len(iq_wave)-1:]

    return np.abs(yx), lags

##
# OFDM
##

def estimate_ofdm_symbol_duration(iq_wave, frame_rate, min_distance=1):
    # Autocorrelation
    yx, lags = full_autocorrelation(iq_wave)
    # Peak
    peak_index = np.argmax(yx[min_distance:]) + min_distance
    peak_value = yx[peak_index]
    # Lag en ms
    lag_time_ms = (lags[peak_index] * 1000) / frame_rate

    return peak_value, lag_time_ms

def estimate_alpha(iq_wave, frame_rate, estimated_ofdm_symbol_duration):
    index_delay  = round(estimated_ofdm_symbol_duration*1e-3*frame_rate) # delay index
    conj = iq_wave[0:-index_delay]*np.conj(iq_wave[index_delay:])
    caf = np.fft.fftshift(np.fft.fft(conj))
    len_caf = len(caf)
    alpha = (-1/2 + np.arange(len_caf)/len_caf)*frame_rate
    # trouver le pic de la CAF pour estimer alpha peak, en ignorant le pic à 0
    discard_dc = np.abs(caf)
    zero_index = np.abs(alpha).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_index = np.argmax(np.abs(discard_dc))
    alpha_peak = alpha[peak_index]

    return alpha_peak, alpha, np.abs(caf)

def calc_ofdm(alpha0,estimated_ofdm_symbol_duration, bandwidth):
    cy_px = (1/abs(alpha0/1e3)) - estimated_ofdm_symbol_duration
    Df = 1/estimated_ofdm_symbol_duration
    cy_px = cy_px
    Df = Df*1e3
    Tu = estimated_ofdm_symbol_duration
    Tg = round(cy_px,5)
    Ts = round(estimated_ofdm_symbol_duration + cy_px,5)
    Df = round(Df,3)
    # Estime le nombre de sous-porteuses dans le symbole OFDM à partir de : 1) la bande passante estimée et 2) Df = l'espacement des sous-porteuses
    # Nb max de sous-porteuses défini à 1000 pour éviter les erreurs dues à des valeurs aberrantes des 2 paramètres
    for N in range(1, 1000):
        if N*Df >= bandwidth:
            break
        else:
            continue
    N = N - 1
    return Tu, Tg, Ts, Df, N
