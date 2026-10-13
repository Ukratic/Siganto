"""
Fonctions d'estimation des paramètres de modulation d'un signal IQ :
- Rapidité de modulation, autocorrelation, transitions phase & fréquence
- Mesures OFDM : ## FX Socheleau - IMT Atlantique, 2020 ## (code matlab source adapté ici en python)
"""

import numpy as np
from dsp_funcs import get_window
from scipy.signal import find_peaks
##
# Fonctions mesures de la rapidité de modulation
##

def power_spectrum_envelope(iq_sig, samp_rate):
    """Spectre de puissance basé sur l'enveloppe du signal"""
    spectrum = np.abs(iq_sig) ** 2
    spectrum -= np.mean(spectrum)  # retire la moyenne
    spectrum_fft = np.fft.fft(spectrum, len(iq_sig))
    spectrum_fft = np.fft.fftshift(spectrum_fft)
    # fréquences de -samp_rate/2 à samp_rate/2 pour l'affichage du spectre
    f = np.fft.fftshift(np.fft.fftfreq(len(spectrum), 1/samp_rate))
    # ignorer autour de 0hz pour la détection du pic de fréquence mais conserver pour l'analyse
    discard_dc = np.abs(spectrum_fft)
    zero_index = np.abs(f).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0 # ignorer 10 indices autour de 0hz
    peak_freq_index = np.argmax(discard_dc)  # Index du pic de fréquence
    if 0 < peak_freq_index < len(discard_dc) - 1:
        y0, y1, y2 = (discard_dc[peak_freq_index-1],
                    discard_dc[peak_freq_index],
                    discard_dc[peak_freq_index+1])
        d = y0 - 2*y1 + y2
        if abs(d) > 1e-12:
            delta = 0.5*(y0 - y2) / d
            peak_freq = f[peak_freq_index] + delta * (f[1] - f[0])
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_freq = 0

    return spectrum_fft, f, peak_freq

# méthode très similaire
def envelope_spectrum(iq_sig, samp_rate):
    """Spectre de l'enveloppe du signal"""
    envelope = np.abs(iq_sig)
    envelope -= np.mean(envelope)  # retire la moyenne
    env_fft = np.fft.fft(envelope)
    env_fft = np.fft.fftshift(env_fft)  # centre sur la fréquence zéro
    env_freqs = np.fft.fftshift(np.fft.fftfreq(len(envelope), 1/samp_rate))
    # Spectre de puissance de l'enveloppe
    env_power = np.abs(env_fft)**2 
    discard_dc = np.abs(env_power)
    zero_index = np.abs(env_freqs).argmin()
    discard_dc[zero_index-10:zero_index+10] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_freq = env_freqs[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.mean(discard_dc):
        peak_freq = 0

    return env_power, env_freqs, peak_freq

def mean_threshold_spectrum(iq_sig, samp_rate, mode="real"):
    """Détection de la rapidité de modulation basée sur le seuillage moyen.
    Mode='real' : fonctionne pour PSK/ASK aligné sur axe réel, sensible à une rotation.
    Mode='phase' : protégé d'une rotation globale constante mais aveugle à une FSK symétrique."""
    if mode == 'real':
        midpoint = iq_sig > np.mean(iq_sig)
    elif mode == 'phase':
        d = np.real(iq_sig[1:] * np.conj(iq_sig[:-1]))
        midpoint = d < 0
    pulse = np.diff(midpoint)**2 
    clock = np.fft.fft(pulse, len(iq_sig))
    clock = np.fft.fftshift(clock)
    # ensuite, idem que power_spectrum_envelope
    f = np.fft.fftshift(np.fft.fftfreq(len(iq_sig), d=1.0 / samp_rate))
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
    f = np.fft.fftshift(np.fft.fftfreq(len(iq_sig), d=1.0 / samp_rate))
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

def power_signal(iq_sig, samp_rate, power=2):
    """Signal à la puissance demandée pour corriger la fréquence centrale
    avec le résidu de porteuse"""
    f = np.fft.fftshift(np.fft.fftfreq(len(iq_sig), d=1.0 / samp_rate))
    # puissance du signal à la puissance N
    # normalisation avant de prendre la puissance pour éviter le risque d'overflow
    norm_sig = iq_sig / np.max(np.abs(iq_sig))
    samples_power = norm_sig**power
    powered_sig = np.abs(np.fft.fftshift(np.fft.fft(samples_power)))/len(iq_sig)
    # chercher le pic de fréquence dans le signal à la puissance N pour estimer la fréquence centrale résiduelle
    peak_freq_index = np.argmax(powered_sig)
    peak_freq = f[peak_freq_index]
    # offset de la fréquence centrale résiduelle en prenant en compte la puissance N : si power=2, la fréquence résiduelle est à 2*peak_freq, etc.
    if power > 1:
        peak_freq = peak_freq / power
    
    return f, powered_sig, peak_freq

# Cyclospectre
def cyclic_spectrum_fft(iq_sig):
    """Cyclospectre FFT. Retourne la magnitude du cyclospectre pour une 
    fenêtre donnée du signal."""
    y = np.abs(iq_sig)**2  # |x[n]|^2
    y = y - np.mean(y)
    Y = np.fft.fftshift(np.fft.fft(y, n=len(iq_sig)))
    # On peut aussi retourner la partie complexe du cyclospectre pour une analyse plus approfondie,
    # mais ici on se concentre sur la magnitude pour l'estimation de la rapidité de modulation
    magnitude = np.abs(Y)
    return magnitude

def cyclic_spectrum_sliding_fft(iq_sig, samp_rate, window, frame_len=512, step=256, estim_bw=None):
    """Cyclospectre avec FFT glissante.
    Estimation de la rapidité de modulation basée sur la moyenne de la corrélation cyclique 
    sur plusieurs fenêtres glissantes."""
    if len(iq_sig) < frame_len:
        return None,None,None
    window = get_window(window, frame_len)
    # fréquences cycliques
    freqs_list = np.fft.fftshift(np.fft.fftfreq(frame_len, d=1/samp_rate)) # axe des fréquences
    # Initialisation de l'accumulateur de corrélation cyclique
    cyclic_corr_accum = np.zeros(len(freqs_list))
    frame_count = 0

    for start in range(0, len(iq_sig) - frame_len + 1, step):
        frame = iq_sig[start:start + frame_len]
        frame_win = frame * window
        cyclic_corr = cyclic_spectrum_fft(frame_win)
        cyclic_corr_accum += cyclic_corr
        frame_count += 1

    cyclic_corr_avg = cyclic_corr_accum / frame_count # moyenne des corrélations cycliques
    bin_width = samp_rate / frame_len
    estim_bw = estim_bw if estim_bw is not None else samp_rate / 30 # repli grossier si non fourni
    n_bins = max(1, int(np.ceil(estim_bw / bin_width)))
    # On retire les pics de puissance autour de 0 Hz pour déterminer la rapidité de modulation
    discard_dc = np.abs(cyclic_corr_avg)
    zero_index = np.abs(freqs_list).argmin()
    discard_dc[zero_index-n_bins:zero_index+n_bins] = 0
    peak_freq_index = np.argmax(discard_dc)
    peak_freq = freqs_list[peak_freq_index]
    # pas de pic de fréquence si ce n'est pas clairement au-dessus du bruit
    if np.max(discard_dc) < 2*np.median(discard_dc):
        peak_freq = 0

    return freqs_list, cyclic_corr_avg, peak_freq

# Experimental : Estimateur utilisant le cepstre
def cepstral_estimator(iq_sig, sample_rate, window='hann', fmin=None, fmax=None):
    """Estimation de la rapidité de modulation basée sur le cepstre de la puissance du signal."""
    x = np.abs(iq_sig)**2
    x = x - np.mean(x)
    N = len(x)
    N_fft = 8 * N # zero padding pour une meilleure résolution en quefrency
    window = get_window(window, N)
    # Spectrum
    X = np.fft.fft(x * window, n=N_fft)
    mag = np.abs(X) + 1e-12 # éviter log(0)
    mag[0:10] = np.mean(mag) # ignore basses quefrencies
    mag = mag / np.max(mag)
    mag = np.maximum(mag, 1e-6)
    # Cepstrum
    cep = np.fft.ifft(np.log(mag)).real
    # Axe des quefrencies
    q = np.arange(N_fft) / sample_rate
    # Ignore les quefrencies trop basses ou trop hautes
    min_q = 1 / (fmax if fmax else sample_rate/2)
    max_q = 1 / (fmin if fmin else sample_rate/1000)
    mask = (q > min_q) & (q < max_q)
    if not np.any(mask):
        return 0, q, cep
    # recherche de pic dans le cepstrum pour estimer la rapidité de modulation
    peaks, properties = find_peaks(cep[mask],prominence=np.std(cep[mask]) * 0.5)
    if len(peaks) == 0:
        return 0, q, cep
    
    best_peak = peaks[np.argmax(properties["prominences"])]
    q_peak = q[mask][best_peak]
    sr = 1 / q_peak

    return sr, q, cep

# Expérimental
def compute_bispectrum(signal, fs, nfft=256, overlap=128,window_type='hann'):
    """Calcul du bispectre d'un signal IQ, mesure de la non-linéarité et de 
    la dépendance de phase entre les composantes fréquentielles."""
    # Segmentation en fenêtres avec overlap
    hop_size = int(nfft - overlap)
    num_segments = (len(signal) - nfft) // hop_size + 1
    if num_segments <= 0:
        return None, None # Signal trop court
    # Fenêtrage
    window = get_window(window_type, nfft)
    # FFT pour chaque segment
    ffts = np.zeros((num_segments, nfft), dtype=complex)
    for i in range(num_segments):
        start = i * hop_size
        chunk = signal[start:start+nfft] * window
        ffts[i, :] = np.fft.fft(chunk)

    # Bispectrum : B(f1, f2) = E[ X(f1) * X(f2) * conj(X(f1+f2)) ]
    B = np.zeros((nfft, nfft), dtype=complex)
    # Loop sur toutes les combinaisons de fréquences f1 et f2
    for f1 in range(nfft):
        for f2 in range(nfft):
            # Combinaison de fréquences modulo nfft
            f3 = (f1 + f2) % nfft
            # Vectorisation sur les segments pour calculer la moyenne
            prod = ffts[:, f1] * ffts[:, f2] * np.conj(ffts[:, f3])
            B[f1, f2] = np.mean(prod)
            
    B_shifted = np.fft.fftshift(B)
    B_mag = np.abs(B_shifted)
    # Axe des fréquences pour affichage
    f = np.linspace(-fs/2, fs/2, nfft)
    
    return f, B_mag

##
# Fonctions de mesures temporelles
##

# Spectre de persistance
def persistance_spectrum(iq_sig, samp_rate, N=256, power_bins=50, window_type='hann', overlap=128):
    """Spectre de persistance des fréquences"""
    step_size = N - overlap
    num_windows = (len(iq_sig) - overlap) // step_size
    spectrogram = np.zeros((num_windows, N))
    window = get_window(window_type, N)
    for i in range(num_windows): # calcul du spectrogramme
        start = i * step_size
        end = start + N
        segment = iq_sig[start:end] * window
        fft_result = np.fft.fft(segment, N)
        psd_segment = (np.abs(np.fft.fftshift(fft_result))**2) / (samp_rate * N)
        psd_segment /= sum(window**2)
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
    phase = np.unwrap(phase)
    if window_size > 1:
        window = get_window(window_type, window_size)
        window /= np.sum(window)
        smoothed = np.convolve(phase, window, mode='same')
    else:
        smoothed = phase
    if not unwrap:
        smoothed = (smoothed + np.pi) % (2*np.pi) - np.pi

    time = np.arange(len(phase)) / samp_rate

    return time, smoothed

# Mesures de phase cumulée
def phase_cumulative_distribution(iq_sig, samp_rate, num_bins=250, window_size=5, window_type='rectangular'):
    """Estimation de la distribution de la phase positive
    Peut aider à identifier certains types de modulation 
    mais requiert une bonne synchronisation en fréquence"""
    phase = phase_time_angle(iq_sig, samp_rate, window_size, window_type)[1] # wrapped [-π, π]
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
        window = get_window(window_type,window_size)
        window /= np.sum(window)
        smoothed_freq = np.convolve(inst_freq, window, mode='same')
    else:
        smoothed_freq = inst_freq

    time = np.arange(0, len(smoothed_freq)) / samp_rate

    return time, smoothed_freq

# Mesures de fréquence cumulée
def frequency_cumulative_distribution(iq_sig, samp_rate, num_bins=250, 
                                      window_size=5, window_type='rectangular'):
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
    yx = np.fft.ifft(fft_iq_sig * np.conj(fft_iq_sig))  # Resultat d'autocorrelation
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

# Autocorrelation sur le signal directement et non la FFT
def full_autocorrelation(iq_sig):
    """Fonction d'autocorrélation sur le signal complexe.
    Plus lente que l'autocorrélation par FFT """
    yx, lags = np.correlate(iq_sig, iq_sig, mode='full'), np.arange(-len(iq_sig)+1, len(iq_sig))
    # les valeurs négatives de lags sont ignorées = doublons
    yx = yx[len(iq_sig)-1:]
    lags = lags[len(iq_sig)-1:]

    return np.abs(yx), lags

# SCF
def scf_tsm(samples, fs=48000, Nw=512, window=None, overlap=256,
           max_alpha=None, alpha_step_hz=5, renorm_every=2000):
    """Fonction de corrélation spectrale (SCF), méthode TSM.
    alpha est la fréquence cyclique au sens standard (décalage +-alpha/2)."""
    if max_alpha is None:
        max_alpha = fs / 2
    alphas = np.arange(0, max_alpha + alpha_step_hz, alpha_step_hz)
    len_sig = len(samples)
    step_size = Nw - overlap
    num_windows = (len_sig - overlap) // step_size
    if num_windows <= 0:
        raise ValueError(f"Signal trop court ({len_sig}) pour Nw={Nw}, overlap={overlap}")
    if window is None: 
        win = np.ones(Nw)
    else:
        win = get_window(window, Nw)

    #indices de toutes les fenêtres
    starts = np.arange(num_windows) * step_size
    idx = starts[:, None] + np.arange(Nw)[None, :] # (num_windows, Nw)

    # rotateur incrémental 
    n = np.arange(len_sig)
    step_rot = np.exp(-1j * np.pi * alpha_step_hz / fs * n) # = exp(-j*2*pi*(alpha_step/2)*n/fs)
    cur = np.ones(len_sig, dtype=complex) # décalage pour alpha=0

    SCF = np.zeros((len(alphas), Nw), dtype=complex) # initialisation de SCF
    for ii in range(len(alphas)): # on boucle sur les fréquences cycliques
        if ii > 0:
            cur *= step_rot
            if ii % renorm_every == 0:
                cur /= np.abs(cur)
        neg = samples * cur # décalage de +alpha/2
        pos = samples * np.conj(cur) # décalage de -alpha/2

        #toutes les fenêtres en un seul appel FFT (axis=1)
        neg_frames = win[None, :] * neg[idx] # (num_windows, Nw)
        pos_frames = win[None, :] * pos[idx]
        Xneg = np.fft.fft(neg_frames, axis=1)
        Xpos = np.fft.fft(pos_frames, axis=1)
        SCF[ii, :] = np.sum(Xneg * np.conj(Xpos), axis=0)

    SCF = np.fft.fftshift(SCF, axes=1)
    SCF = np.abs(SCF)
    SCF[0, :] = 0
    faxis = np.fft.fftshift(np.fft.fftfreq(Nw, d=1/fs))
    return SCF, faxis, alphas

##
# OFDM
# Fonctions pour estimer les paramètres de modulation OFDM, 
# basés sur le code Matlab de FX Socheleau - IMT Atlantique, 2020
# Cours "Analyse aveugle de signaux de communication" - Telecom Paris 2024
##

def estimate_ofdm_symbol_duration(iq_sig, samp_rate, min_distance=1):
    """Estimation de durée symbole OFDM avec la fonction d'autocorrélation"""
    yx, lags = autocorrelation(iq_sig, samp_rate)
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
    alpha = np.fft.fftshift(np.fft.fftfreq(len_caf, d=1/samp_rate))
    # trouver automatiquement le pic de la CAF dans les fréquences alpha en ignorant le pic DC
    tu = estimated_ofdm_symbol_duration*1e-3 # en secondes
    abs_alpha = np.abs(alpha)
    band = (abs_alpha > 0.4/tu) & (abs_alpha < 1/tu)
    res = np.where(band, np.abs(caf), 0)
    zero_index = abs_alpha.argmin()
    res[zero_index-10:zero_index+10] = 0
    peak_index = np.argmax(np.abs(res))
    alpha_peak = abs_alpha[peak_index]

    return alpha_peak, abs_alpha, res

def calc_ofdm(alpha0,estimated_ofdm_symbol_duration, bandwidth):
    """Calcul des paramètres OFDM à partir de la fréquence alpha, durée symbole OFDM et BW"""
    cy_px = (1/abs(alpha0/1e3)) - estimated_ofdm_symbol_duration # en ms
    Df = 1/estimated_ofdm_symbol_duration # en kHz
    Df = Df*1e3 # en Hz
    Tu = estimated_ofdm_symbol_duration # en ms
    Tg = round(cy_px,5) # en ms
    Ts = round(estimated_ofdm_symbol_duration + cy_px,5) # en ms
    Df = round(Df,3)
    # Estime le nombre de sous-porteuses du signal OFDM à partir de : 
    # 1) la bande passante estimée et 2) Df = l'espacement des sous-porteuses
    # Nb max de sous-porteuses
    N = int(bandwidth // Df)

    return Tu, Tg, Ts, Df, N
