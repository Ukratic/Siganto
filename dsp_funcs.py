"""Fonctions DSP annexes"""
import numpy as np
from scipy.signal import find_peaks, fftconvolve
from scipy.fft import next_fast_len

# Flattop window
def flattop_window(N):
    """Fenêtre flattop pour la FFT"""
    if N <= 1:
        return np.array([])

    # Coefficients pour la fenêtre Flattop (source : implémentation Octave de Flattop Harris)
    a0 = 1.0
    a1 = 1.9326
    a2 = 1.2861
    a3 = 0.3877
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
    """Fenêtre 7-term Blackman-Harris"""
    if N <= 1:
        return np.array([])

    # Coefficients pour Blackman-Harris 7-term (source : National Instruments & H.H. Albrecht)
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
    if N <= 1:
        return np.array([])
    
    n = np.arange(0, N)
    return np.exp(-0.5 * ((n - (N-1)/2) / (sigma * (N-1)/2))**2)

def lanczos_window(N):
    if N <= 1:
        return np.array([])

    n = np.arange(N)
    x = 2.0 * n / (N - 1) - 1.0

    return np.sinc(x)

def get_window(window_type, N):
    """Retourne la fenêtre (périodique) appropriée selon le type"""
    if window_type == 'flattop': # précision en amplitude
        return flattop_window(N+1)[:-1]
    elif window_type == 'blackmanharris7term': # réjection de lobes secondaires & haute dynamique
        return blackman_harris_7term_window(N+1)[:-1]
    elif window_type == 'hann': # cas général
        return np.hanning(N+1)[:-1]
    elif window_type == 'hamming': # autre cas général adapté audio
        return np.hamming(N+1)[:-1]
    elif window_type == 'blackman': # réjection de lobes secondaires
        return np.blackman(N+1)[:-1]
    elif window_type == 'kaiser': # trade-off adaptatif
        return np.kaiser(N+1, beta=14)[:-1] # peut être ajusté pour un compromis entre largeur de bande et réjection
    elif window_type == 'bartlett': # triangulaire, lissage simple
        return np.bartlett(N+1)[:-1]
    elif window_type == 'rectangular': # pas de fenêtre
        return np.ones(N+1)[:-1]
    elif window_type == 'gaussian': # fenêtre gaussienne
        return gaussian_window(N+1)[:-1]
    elif window_type == 'lanczos':
        return lanczos_window(N+1)[:-1] # alternative
    else:
        raise ValueError(f"Type de fenêtre inconnu : {window_type}")

def correct_constellation_rotation(iq_sig, M, family='psk'):
    """Corrige une rotation de phase statique (pas d'offset de fréquence résiduel).
    family='psk' : M = 2, 4, 8 (π/4-QPSK : 8).
    family='qam' : M = 4 imposé quel que soit l'ordre réel.
    Ambiguïté résiduelle : 2π/M (90° pour QPSK/QAM)."""
    family = family.lower()
    if family not in ['psk', 'qam']:
        raise ValueError("family doit être 'psk' ou 'qam'")
    if family == 'qam':
        M = 4
    M = int(M)
    if M < 2:
        raise ValueError("M doit être >= 2")
    m = np.mean(iq_sig ** M, dtype=np.complex128)
    if family == 'qam':
        m = -m 
    theta = np.angle(m) / M

    return iq_sig * np.complex64(np.exp(-1j * theta)), theta

def lora_score(x, fs, bw, sf):
    """Calcule un score de détection LoRa pour un signal donné.
    Valeur supérieure à 100 indique une forte probabilité."""
    Ts = 2**sf / bw
    N = int(Ts*fs)
    t = np.arange(N)/fs
    
    step = N // 8
    if step == 0:
        step = 1
    scores = []
    # fenetre glissante pour éviter pb de timing
    for start in range(0, len(x)-N, step):
        # dechirpage classique
        chirp = np.exp(1j*np.pi*bw/Ts*t*t)
        block = x[start:start+N]
        if len(block) < len(chirp):
            block = np.pad(block, (0, len(chirp) - len(block)), 'constant')
        dechirped = block * np.conj(chirp)
        if len(dechirped) == 0:
            continue
        spectrum = abs(np.fft.fft(dechirped))
        peak = spectrum.max()
        avg = np.median(spectrum)
        score = peak / avg if avg != 0 else 0
        scores.append(score)

    return np.mean(scores) if scores else 0

# fonction de décalage d'un signal complexe sur une nouvelle fréquence centrale
def frequency_shift(iq_sig, fs, f, chunk=1 << 20):
    """Décale iq_sig de f Hz. Phase en float64 (indice absolu), sortie complex64."""
    # par défaut chunk de 2^20 échantillons (env 1M) pour limiter la mémoire
    out = np.empty(len(iq_sig), dtype=np.complex64)
    w = 2.0 * np.pi * f / fs
    for s in range(0, len(iq_sig), chunk):
        e = min(s + chunk, len(iq_sig))
        n = np.arange(s, e, dtype=np.float64)
        out[s:e] = iq_sig[s:e] * np.exp(-1j * w * n) # temporairement complex128 sur 1 seul chunk

    return out

def estimate_carrier_offset(iq_sig, fs, chunk=1 << 20):
    """Discriminateur de phase pondéré par mag² minimale.
    Adapté pour FSK équilibré, PSK/QAM suréchantillonné."""
    acc = 0j
    for s in range(0, len(iq_sig) - 1, chunk):
        e = min(s + chunk, len(iq_sig) - 1)
        a, b = iq_sig[s:e], iq_sig[s + 1:e + 1]
        d = b * np.conj(a)
        mag = np.abs(d)
        w = np.minimum(np.abs(a), np.abs(b)) ** 2 # pondération par mag² minimale
        ok = mag > 0
        acc += np.sum(w[ok] * d[ok] / mag[ok], dtype=np.complex128)
    if acc == 0:
        raise ValueError("Signal nul ou constant : fréquence indéterminée")
    
    return fs / (2 * np.pi) * np.angle(acc)

def estimate_carrier_mpower(iq_sig, fs, M, max_len=1 << 24):
    """Raie de x**M (FFT fenêtrée + interpolation parabolique) / M. 
    Recherche sur plage max_len échantillons (défaut 2^24 = 16M).
    Adapté pour PSK/QAM."""
    xm = iq_sig[:max_len] ** M
    n = len(xm)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    mag = np.abs(np.fft.fft(xm * get_window('hann', n), n=nfft))
    k = int(np.argmax(mag))
    a, b, c = np.log(mag[[(k - 1) % nfft, k, (k + 1) % nfft]] + 1e-30)
    den = a - 2 * b + c
    p = 0.5 * (a - c) / den if den != 0 else 0.0
    if k >= nfft // 2:
        k -= nfft

    return (k + p) / nfft * fs / M

def estimate_carrier_weighted(iq_sig, fs, M=None):
    """M=None : Discriminateur pondéré (FSK équilibré, PSK/QAM suréchantillonné).
    M=1 : porteuse pure. M=2/4/8 : PSK/QAM."""
    iq_sig = iq_sig - np.complex64(np.mean(iq_sig, dtype=np.complex128)) # retire DC
    center = estimate_carrier_mpower(iq_sig, fs, M) if M else estimate_carrier_offset(iq_sig, fs)

    return frequency_shift(iq_sig, fs, center), center

def center_signal(iq_sig, samp_rate, prominence=6.0, min_distance_hz=None,
                  n_pts_segment=None):
    """Centre sur le milieu des 2 pics les plus hauts du spectre lissé (FSK2).
    prominence : creux minimal en dB entre un pic et le reste du spectre.
    min_distance_hz : écart minimal entre pics (défaut fs/50) ;
    doit être > largeur d'un lobe et < écart entre tons."""
    iq_sig = iq_sig - np.complex64(np.mean(iq_sig, dtype=np.complex128)) # retire DC
    len_sig = len(iq_sig) 
    if n_pts_segment is None:
        n_pts_segment = int(np.clip(2 ** int(np.log2(max(len_sig, 1) / 32)), 64, 8192))
    if len_sig < 2 * n_pts_segment:
        raise ValueError(f"Signal trop court ({len_sig}) pour n_pts_segment={n_pts_segment}")

    hop = n_pts_segment // 2
    win = get_window('hann', n_pts_segment).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(iq_sig, n_pts_segment)[::hop]
    acc = np.zeros(n_pts_segment)
    for s in range(0, len(frames), 128): # par blocs pour mémoire bornée
        spec = np.fft.fft(frames[s:s + 128] * win, axis=1)
        acc += (spec.real ** 2 + spec.imag ** 2).sum(axis=0)
    psd = np.fft.fftshift(acc / len(frames))
    if psd.max() <= 0:
        raise ValueError("Spectre nul")
    psd_db = 10 * np.log10(np.maximum(psd, 1e-30))

    df_bin = samp_rate / n_pts_segment
    dist = max(1, int(round((min_distance_hz or samp_rate / 50) / df_bin)))
    peaks, _ = find_peaks(psd_db, prominence=prominence, distance=dist)
    if len(peaks) < 2:
        raise ValueError(f"Moins de 2 pics (prominence {prominence} dB, distance {dist} bins)")
    top2 = peaks[np.argsort(psd_db[peaks])[::-1][:2]]
    freqs = []
    for k in top2: # interpolation parabolique pour précision sub-bin
        a, b, c = psd_db[k - 1], psd_db[k], psd_db[k + 1]
        den = a - 2 * b + c
        p = 0.5 * (a - c) / den if den != 0 else 0.0
        freqs.append((k - n_pts_segment // 2 + p) * df_bin)
    center = float(np.mean(freqs))

    return frequency_shift(iq_sig, samp_rate, center), center

def morlet_scalogram(x, fs, fmin=None, fmax=None, nfreq=96, w=6.0,
                     two_sided=True, ncols=1024, density=True):
    """Scalogramme Morlet (|W|²), calculé dans le domaine fréquentiel.
    Retourne (power[nrows, ncols], freqs_Hz, times_s, coi_s).
    density=True  : W²/Hz, plancher de bruit blanc plat (même échelle que la DSP).
    density=False : une raie d'amplitude A donne A² sur sa ligne.
    coi_s : demi-largeur (s) du cône d'influence par ligne (3 écarts-types)."""
    x = np.asarray(x)
    N = len(x)
    # FC maximale admissible. Pour une ondelette de Morlet de largeur w, l’écart-type fréquentiel
    # est sigma_f = |f| / w. On impose que la queue de la gaussienne, jusqu’à 4 sigma_f, 
    # reste sous la fréquence de Nyquist pour éviter évite de créer des artefacts d’enroulement
    fmax_lim = fs / (2 * (1 + 4 / w))
    # Fmin par défaut
    if fmin is None:
        fmin = 8 * fs / N
    if fmax is None:
        fmax = fmax_lim
    if not (0 < fmin < fmax <= fmax_lim):
        raise ValueError(f"Il faut 0 < fmin < fmax <= {fmax_lim:.6g} Hz (reçu {fmin}, {fmax})")
    # espacement géométrique des freq pour une résolution logarithmique en fréquence
    fpos = np.geomspace(fmin, fmax, nfreq)
    fc = np.concatenate((-fpos[::-1], fpos)) if two_sided else fpos # FC de chaque bande
    scale = w * fs / (2 * np.pi * np.abs(fc)) # échelle = écart-type temporel de l'ondelettes (en échantillons)
    fft_len = next_fast_len(N + int(np.ceil(4 * scale.max()))) # zero padding pour éviter l'enroulement
    X = np.fft.fft(x, n=fft_len) # une seule FFT
    # résolution fréquentielle de la FFT principale, en Hz par bin
    freqbin_width = fs / fft_len
    edges = np.linspace(0, N, ncols + 1) # bords des colonnes d'affichage (en échantillons)
    # tableau de sortie, puissance par fréquence et colonne temporelle
    power = np.empty((len(fc), ncols), dtype=np.float32)
    for row, f in enumerate(fc):
        sigma_f = abs(f) / w # largeur de la gaussienne (Hz)
        central_bin = int(round(f / freqbin_width))
        half_bw_bins = int(np.ceil(4 * sigma_f / freqbin_width))
        band_len = next_fast_len(2 * half_bw_bins + 1)
        ind_band_bins = central_bin - band_len // 2 + np.arange(band_len) # tranche de bins autour de f
        # fenetre gaussienne dans le domaine frequentiel
        gaussian_window  = np.exp(-0.5 * ((ind_band_bins * freqbin_width - f) / sigma_f) ** 2) # gain 1 au centre
        # enveloppe temporelle en bande de base
        band_envelope = np.fft.ifft(X[ind_band_bins % fft_len] * gaussian_window) * (band_len / fft_len)
        # puissance instantanée
        inst_puiss = band_envelope.real ** 2 + band_envelope.imag ** 2
        # normalisation en équivalent DSP
        if density:
            inst_puiss /= np.sum(gaussian_window * gaussian_window) * freqbin_width # bande équivalente de bruit
        cumul_power = np.concatenate(([0.0], np.cumsum(inst_puiss, dtype=np.float64)))
        # positions des bords des colonnes temporelles sur l'enveloppe
        time_edges = edges * (band_len / fft_len) + 0.5 # moyenne par bloc sur les colonnes
        power[row] = np.diff(np.interp(time_edges, np.arange(band_len + 1), cumul_power)) / np.diff(time_edges)
    # centres des colonnes en s
    times = 0.5 * (edges[1:] + edges[:-1]) / fs
    # cône d’influence : mesure de la zone temporelle où la valeur du scalogramme 
    # est influencée par les bords du signal. Approximé ici par 3 écarts-types temporels
    coi = 3*scale/fs
    return power, fc, times, coi
