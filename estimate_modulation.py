"""
Functions to estimate modulation parameters of an IQ signal :
- Estimate modulation speed (symbol rate), phase, autocorrelation
- Measure OFDM modulation parameters : ## Author (matlab code adapted here in python) : FX Socheleau - IMT Atlantique, 2020 ##
"""
# Comments in French below

import numpy as np
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
    discard_dc = np.abs(squared_metric)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_squared_freq = f[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_squared_freq = 0
    # puissance du signal à la puissance 4
    samples_quartic = iq_wave**4
    quartic_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_quartic)))/len(iq_wave)
    quartic_metric[len(quartic_metric)//2] = 0
    discard_dc = np.abs(quartic_metric)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_quartic_freq = f[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_quartic_freq = 0

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

# Mesures de fréquence cumulée
def frequency_cumulative_distribution(iq_wave, frame_rate, num_bins=250):
    # Calcul de la fréquence instantanée 
    phase = np.angle(iq_wave)
    inst_freq = np.diff(phase) / (2 * np.pi) * frame_rate
    # Histogramme de la fréquence instantanée cumulée
    hist, bins = np.histogram(phase, bins=num_bins, density=True)
    bins = bins[:-1]

    return hist, bins

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
# Fonctions pour estimer les paramètres de modulation OFDM, basés sur le code Matlab de FX Socheleau - IMT Atlantique, 2020
# Cours "Analyse aveugle de signaux de communication" - Telecom Paris 2024
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
    # estime écart de fréquence alpha à partir de la CAF
    index_delay  = round(estimated_ofdm_symbol_duration*1e-3*frame_rate) # delay index
    conj = iq_wave[0:-index_delay]*np.conj(iq_wave[index_delay:])
    caf = np.fft.fftshift(np.fft.fft(conj))
    len_caf = len(caf)
    alpha = (-1/2 + np.arange(len_caf)/len_caf)*frame_rate
    # trouver automatiquement le pic de la CAF pour estimer le pic alpha, en ignorant le pic à 0
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
    # Nb max de sous-porteuses
    for N in range(1, 25000):
        if N*Df >= bandwidth:
            break
        else:
            continue
    N = N - 1
    return Tu, Tg, Ts, Df, N
