"""
Fonctions d'estimation des paramètres de modulation d'un signal IQ :
- Rapidité de modulation, autocorrelation, transitions phase & fréquence
- Mesures OFDM : ## FX Socheleau - IMT Atlantique, 2020 ## (code matlab source adapté ici en python)
"""

import numpy as np
import dsp_funcs as df
##
# Fonctions mesures de la rapidité de modulation
##

def power_spectrum_envelope(iq_sig, samp_rate):
    """Spectre de puissance basé sur l'enveloppe du signal"""
    spectrum = np.abs(iq_sig) ** 2
    spectrum_fft = np.fft.fft(spectrum, len(iq_sig))
    spectrum_fft = np.fft.fftshift(spectrum_fft)
    # fréquences de -samp_rate/2 à samp_rate/2 pour l'affichage du spectre
    f = np.linspace(samp_rate/-2, samp_rate/2, len(iq_sig))
    # ignorer le pic autour de 0hz pour la détection du pic de fréquence mais le conserver pour l'analyse
    discard_dc = np.abs(spectrum_fft)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0 # ignorer 10 indices autour de 0hz
    peak_freq_index = np.argmax(discard_dc)  # Index du pic de fréquence
    peak_freq = f[peak_freq_index]

    return spectrum_fft, f, peak_freq

def envelope_spectrum(iq_sig, samp_rate):
    """Spectre de l'enveloppe du signal"""
    envelope = np.abs(iq_sig)
    envelope -= np.mean(envelope)  # retire la moyenne
    N = len(envelope)
    env_fft = np.fft.fft(envelope)
    env_fft = np.fft.fftshift(env_fft)  # centre sur la fréquence zéro
    env_freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/samp_rate))
    # Spectre de puissance de l'enveloppe
    env_power = np.abs(env_fft)**2
    discard_dc = np.abs(env_power)
    zero_index = np.abs(env_freqs).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_freq = env_freqs[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(env_power) < 2*np.mean(env_power):
        peak_freq = 0

    return env_power, env_freqs, peak_freq


def mean_threshold_spectrum(iq_sig, samp_rate):
    """Détection de la rapidité de modulation basée sur le seuillage moyen"""
    # seuil de la moyenne et calcul des pulses pour la détection de la fréquence de modulation
    midpoint = iq_sig > np.mean(iq_sig)
    pulse = np.diff(midpoint)**2
    clock = np.fft.fft(pulse, len(iq_sig))
    clock = np.fft.fftshift(clock)
    # ensuite, idem que power_spectrum_envelope
    f = np.linspace(samp_rate/-2, samp_rate/2, len(iq_sig))
    discard_dc = np.abs(clock)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_freq = f[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_freq = 0

    return clock, f, peak_freq

def power_series(iq_sig, samp_rate):
    """Signal puissance 2 et 4"""
    f = np.linspace(samp_rate/-2, samp_rate/2, len(iq_sig))
    # puissance du signal au carré
    samples_squared = iq_sig**2
    squared_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_squared)))/len(iq_sig)
    discard_dc = np.abs(squared_metric)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_squared_freq = f[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_squared_freq = 0
    # puissance du signal à la puissance 4
    samples_quartic = iq_sig**4
    quartic_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_quartic)))/len(iq_sig)
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

# Cyclospectre
def cyclic_spectrum_fft(iq_sig, samp_rate, alpha_list):
    """Cyclospectre FFT. Retourne la corrélation cyclique pour une liste de fréquences alpha."""
    N = len(iq_sig)
    y = np.abs(iq_sig)**2  # |x[n]|^2
    Y = np.fft.fftshift(np.fft.fft(y, n=N))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/samp_rate)) # axe des fréquences
    # Interpolation de FFT puissance pour la liste de fréquences alpha
    magnitude = np.abs(Y)
    cyclic_corr = np.interp(alpha_list, freqs, magnitude)
    return cyclic_corr

def cyclic_spectrum_sliding_fft(iq_sig, samp_rate, window, frame_len=512, step=256):
    """Cyclospectre avec FFT glissante.
    Estimation de la rapidité de modulation basée sur la moyenne du cyclospectre.
    Demande plus de calculs (nb d'échantillons conséquent) mais plus robuste que d'une seule FFT sur tout le signal."""
    window = df.get_window(window, frame_len)
    alpha_list = np.linspace(-samp_rate/2, samp_rate/2, int(np.log(len(iq_sig))*1000))  # fréquences cycliques
    # Initialisation de l'accumulateur de corrélation cyclique
    cyclic_corr_accum = np.zeros(len(alpha_list))
    frame_count = 0

    for start in range(0, len(iq_sig) - frame_len + 1, step):
        frame = iq_sig[start:start + frame_len]
        frame_win = frame * window
        cyclic_corr = cyclic_spectrum_fft(frame_win, samp_rate, alpha_list)
        cyclic_corr_accum += cyclic_corr
        frame_count += 1

    cyclic_corr_avg = cyclic_corr_accum / frame_count # moyenne des corrélations cycliques

    # On retire les pics de puissance autour de 0 Hz pour déterminer la rapidité de modulation
    discard_dc = np.abs(cyclic_corr_avg)
    zero_index = np.abs(alpha_list).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_freq = alpha_list[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_freq = 0

    return alpha_list, cyclic_corr_avg, peak_freq

##
# Fonctions de mesures temporelles
##

# Spectre de persistance
def persistance_spectrum(iq_sig, samp_rate, N=256, power_bins=50, window_type='hann', overlap=128):
    """Spectre de persistance des fréquences"""
    step_size = N - overlap
    num_windows = (len(iq_sig) - overlap) // step_size
    spectrogram = np.zeros((num_windows, N))
    window = df.get_window(window_type, N)
    for i in range(num_windows): # calcul du spectrogramme
        start = i * step_size
        end = start + N
        segment = iq_sig[start:end] * window
        fft_result = np.fft.fft(segment, N)
        psd_segment = (np.abs(np.fft.fftshift(fft_result))**2) / (samp_rate * N)
        spectrogram[i, :] = 10*np.log10(psd_segment)
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
    f= np.fft.fftshift(np.fft.fftfreq(N, 1 / samp_rate))

    return f, min_power, max_power, persistence

# Visualiser la phase dans le domaine temporel
def phase_time_angle(iq_sig, samp_rate, window_size=5, window_type='rectangular', unwrap=False):
    """Transitions de phase dans le domaine temporel (phase instantanée)"""
    phase = np.angle(iq_sig)
    # Lissage de la phase
    if unwrap:
        phase = np.unwrap(phase)
    if window_size > 1:
        window = df.get_window(window_type,window_size)
        window /= np.sum(window)
        smoothed_phase = np.convolve(phase, window, mode='same')
    else:
        smoothed_phase = phase

    time = np.arange(0, len(phase)) / samp_rate

    return time, smoothed_phase

# Mesures de phase cumulée
def phase_cumulative_distribution(iq_sig, num_bins=250):
    """Estimation de la distribution de la phase positive
    Peut aider à identifier certains types de modulation mais requiert une bonne synchronisation en fréquence"""
    phase = np.angle(iq_sig)  # wrapped [-π, π]
    phase_pos = phase[phase > 0]  # garder seulement les phases positives
    hist, bin_edges = np.histogram(phase_pos, bins=num_bins, range=(0, np.pi), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers

def frequency_transitions(iq_sig, samp_rate, window_size=5, window_type='rectangular'):
    """Transitions de fréquence dans le domaine temporel (fréquence instantanée)"""
    phase = np.unwrap(np.angle(iq_sig))
    # Freq instantanee derivee de la phase
    inst_freq = np.diff(phase) / (2 * np.pi) * samp_rate
    if window_size > 1:
        window = df.get_window(window_type,window_size)
        window /= np.sum(window)
        smoothed_freq = np.convolve(inst_freq, window, mode='same')
    else:
        smoothed_freq = inst_freq

    time = np.arange(0, len(smoothed_freq)) / samp_rate

    return time, smoothed_freq

# Mesures de fréquence cumulée
def frequency_cumulative_distribution(iq_sig, samp_rate, num_bins=250, window_size=5, window_type='rectangular'):
    """Distribution de fréquence instantanée"""
    # optionnel : lisser la fréquence instantanée pour une distribution plus propre
    inst_freq = frequency_transitions(iq_sig, samp_rate, window_size, window_type)[1]
    # Histogramme de la fréquence instantanée cumulée
    hist, bin_edges = np.histogram(inst_freq, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers

##
# ACF
##

# Autocorrelation rapide
def autocorrelation(iq_sig, samp_rate=None):
    """Fonction d'autocorrélation sur la FFT"""
    n = len(iq_sig)
    n_padded = 2**np.ceil(np.log2(2 * n - 1)).astype(int) # zero padding
    # FFT du signal pour l'autocorrelation
    fft_iq_sig = np.fft.fft(iq_sig, n=n_padded)
    yx = np.fft.ifft(fft_iq_sig * np.conj(fft_iq_sig)).real  # Resultat d'autocorrelation
    # Trim : valeurs absolues
    yx = yx[:n]
    yx = np.abs(yx)
    # Compute lags à partir de 0
    lags = np.arange(0, n)
    # Convertir lags en temps
    if samp_rate is not None:
        lags = lags / samp_rate

    return yx, lags

def autocorrelation_peak(iq_sig, samp_rate, min_distance=1):
    """Recherche de pic d'autocorrélation"""
    yx, lags = autocorrelation(iq_sig, samp_rate)
    # Trouver le pic en excluant zéro
    yx[0] = 0
    peak_index = np.argmax(yx[min_distance:]) + min_distance
    peak_value = yx[peak_index]
    lag_time_ms = lags[peak_index] * 1000  # s -> ms

    return peak_value, lag_time_ms

def autocorrelation_peak_from_acf(yx, lags, min_distance=1):
    """Rechercher le pic d'autocorrélation sans recalculer"""
    yx[0] = 0
    peak_index = np.argmax(yx[min_distance:]) + min_distance
    peak_value = yx[peak_index]
    lag_time_ms = lags[peak_index] * 1000  # s -> ms

    return peak_value, lag_time_ms

# Autocorrelation complète (lente)
def full_autocorrelation(iq_sig):
    """Fonction d'autocorrélation sur le signal complexe.
    Fonction plus lente que l'autocorrélation par FFT mais retourne la fonction complète, plus précise"""
    yx, lags = np.correlate(iq_sig, iq_sig, mode='full'), np.arange(-len(iq_sig)+1, len(iq_sig))
    # les valeurs négatives de lags sont ignorées = doublons
    yx = yx[len(iq_sig)-1:]
    lags = lags[len(iq_sig)-1:]

    return np.abs(yx), lags

#SCF
def scf_tsm(samples, fs=48000, Nw=512, window=None, overlap=256, max_alpha=None, alpha_step_hz=5):
    """Compute the Spectral Correlation Function (SCF) using a sliding-window cross-cyclic method (TSM)."""
    """Fonction de corrélation spectrale (SCF) en utilisant une méthode de fenêtre glissante trans-cyclique (TSM)"""
    if max_alpha is None:
        max_alpha = fs / 2

    alphas = np.arange(0, max_alpha + alpha_step_hz, alpha_step_hz)
    N = len(samples)
    step_size = Nw - overlap
    num_windows = (N - overlap) // step_size

    if window is None:
        window = np.ones(Nw)
    else:
        window = df.get_window(window,Nw)

    SCF = np.zeros((len(alphas), Nw), dtype=complex) # initialisation de SCF

    for ii, alpha in enumerate(alphas): # on boucle sur les fréquences cycliques
        n = np.arange(N)
        neg = samples * np.exp(-1j * 2 * np.pi * alpha / fs * n)
        pos = samples * np.exp( 1j * 2 * np.pi * alpha / fs * n)

        for i in range(num_windows):
            idx_start = i * step_size
            idx_end = idx_start + Nw
            pos_slice = window * pos[idx_start:idx_end]
            neg_slice = window * neg[idx_start:idx_end]
            SCF[ii, :] += np.fft.fft(neg_slice) * np.conj(np.fft.fft(pos_slice))

    SCF = np.fft.fftshift(SCF, axes=1)
    SCF = np.abs(SCF)
    SCF[0, :] = 0  # zero α=0 qui est la DSP

    faxis = np.fft.fftshift(np.fft.fftfreq(Nw, d=1/fs))
    return SCF, faxis, alphas

##
# OFDM
# Fonctions pour estimer les paramètres de modulation OFDM, basés sur le code Matlab de FX Socheleau - IMT Atlantique, 2020
# Cours "Analyse aveugle de signaux de communication" - Telecom Paris 2024
##

def estimate_ofdm_symbol_duration(iq_sig, samp_rate, min_distance=1):
    """Estimation de durée symbole OFDM avec la fonction d'autocorrélation"""
    yx, lags = full_autocorrelation(iq_sig)
    # Peak
    peak_index = np.argmax(yx[min_distance:]) + min_distance
    peak_value = yx[peak_index]
    # Lag en ms
    lag_time_ms = (lags[peak_index] * 1000) / samp_rate

    return peak_value, lag_time_ms

def estimate_alpha(iq_sig, samp_rate, estimated_ofdm_symbol_duration):
    """Estimation écart de fréquence alpha à partir de la fonction d'autocorrélation"""
    index_delay  = round(estimated_ofdm_symbol_duration*1e-3*samp_rate) # delay index
    conj = iq_sig[0:-index_delay]*np.conj(iq_sig[index_delay:]) # produit conjugué décalé
    caf = np.fft.fftshift(np.fft.fft(conj)) # corrélation cyclique
    len_caf = len(caf)
    alpha = (-1/2 + np.arange(len_caf)/len_caf)*samp_rate
    # trouver automatiquement le pic de la CAF dans les fréquences alpha en ignorant le pic DC
    discard_dc = np.abs(caf)
    zero_index = np.abs(alpha).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_index = np.argmax(np.abs(discard_dc))
    alpha_peak = alpha[peak_index]

    return alpha_peak, alpha, np.abs(caf)

def calc_ofdm(alpha0,estimated_ofdm_symbol_duration, bandwidth):
    """Calcul des paramètres OFDM à partir de la fréquence alpha, durée symbole OFDM et BW"""
    cy_px = (1/abs(alpha0/1e3)) - estimated_ofdm_symbol_duration # en ms
    Df = 1/estimated_ofdm_symbol_duration # en kHz
    Df = Df*1e3 # en Hz
    Tu = estimated_ofdm_symbol_duration # en ms
    Tg = round(cy_px,5) # en ms
    Ts = round(estimated_ofdm_symbol_duration + cy_px,5) # en ms
    Df = round(Df,3)
    # Estime le nombre de sous-porteuses dans le symbole OFDM à partir de : 1) la bande passante estimée et 2) Df = l'espacement des sous-porteuses
    # Nb max de sous-porteuses
    for N in range(1, 25000):
        if N*Df >= bandwidth:
            break

    N = N - 1

    return Tu, Tg, Ts, Df, N
