"""Fonctions de démodulation et/ou d'estimation du débit symbole. 
WPCR pour demod NRZ basée sur les fonctions de Michael Ossmann, 
"clock recovery experiments" : https://github.com/mossmann/clock-recovery"""
import string
import numpy as np
from scipy.signal import argrelextrema, find_peaks
from helper_funcs import parabolic_interpol

def find_clock_frequency(fdiff, sample_rate, target_rate=None, precision=0.9):
    """Détermine la fréquence d'horloge"""
    maxima = argrelextrema(fdiff, np.greater_equal)[0] # indices des maxima locaux
    while maxima.size > 0 and maxima[0] < 2:
        maxima = maxima[1:]
    if not maxima.any():
        return 0

    if target_rate: # bornes autour de la fréquence cible
        min_rate = target_rate * precision
        max_rate = target_rate / precision
    else:
        min_rate, max_rate = None, None

    # Convertit les fréquences min/max en bins FFT
    nfft = len(fdiff)
    min_bin = int((min_rate / sample_rate) * nfft) if min_rate else 2
    max_bin = int((max_rate / sample_rate) * nfft) if max_rate else nfft // 2

    # Filtre les maxima en dehors des bornes
    maxima = maxima[(maxima >= min_bin) & (maxima <= max_bin)]

    if maxima.size == 0:
        return 0  # pas de pic dans la plage

    threshold = max(fdiff[min_bin:max_bin+1]) * 0.8
    indices_above_threshold = np.argwhere(fdiff[maxima] > threshold)

    # renvoie le premier pic au-dessus du seuil
    return 0 if indices_above_threshold.size < 1 else maxima[indices_above_threshold[0]].item()

def midpoint(a):
    """Fonction pour déterminer le milieu de segment"""
    mean_a = np.mean(a)
    mean_a_greater = np.ma.masked_greater(a, mean_a)
    high = np.ma.median(mean_a_greater) # médiane des valeurs > moyenne
    mean_a_less_or_equal = np.ma.masked_array(a, ~mean_a_greater.mask)
    low = np.ma.median(mean_a_less_or_equal) # médiane des valeurs ≤ moyenne
    return (high + low) / 2 # milieu entre les deux médianes

# whole packet clock recovery
def wpcr(a, sample_rate, target_rate, tau, precision, debug):
    """Fonction principale de la méthode de démodulation"""
    if len(a) < 4:
        return []
    b = (a > midpoint(a)) * 1.0 # seuil
    d = np.diff(b)**2 # détection des transitions
    if len(np.argwhere(d > 0)) < 2:
        return []
    f = np.fft.fft(d, len(a))
    p = find_clock_frequency(abs(f),sample_rate,target_rate,precision)
    if p == 0:
        return [], 0.0
    cycles_per_sample = (p*1.0)/len(f) # conversion index bin en cycles/sample
    clock_phase = 0.5 + np.angle(f[p])/(tau) # phase initiale de l'horloge. tau par défaut = 2*pi
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i]) # échantillon au centre du symbole
        clock_phase += cycles_per_sample
    clock_frequency = p * sample_rate / len(f) # conversion en Hz
    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("detected clock frequency: %d Hz" % (p * sample_rate // len(f)))
        print("samples per symbol: %f" % (1.0/cycles_per_sample))
        print("clock cycles per sample: %f" % (cycles_per_sample))
        print("clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase))
        print("clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2))
        print("symbol count: %d" % (len(symbols)))
    return symbols, clock_frequency

def estimate_baud_rate(a, sample_rate, target_rate=None, precision=0.9, debug=False):
    """Estimation de rapidité de modulation"""
    if len(a) < 4:
        return 0
    # seuil
    b = (a > midpoint(a)) * 1.0
    d = np.diff(b) ** 2
    if len(np.argwhere(d > 0)) < 2:
        return 0, np.array([]), np.array([])
    # FFT des transitions
    f = np.fft.fft(d, len(a))
    p = find_clock_frequency(abs(f), sample_rate, target_rate, precision)
    if p == 0:
        return 0, np.array([]), np.array([])
    # conversions index des bins en Hz
    baud_rate = p * sample_rate / len(f)
    # fréquences et mag pour le graphe
    freqs = np.fft.fftfreq(len(f), 1/sample_rate)
    mag = np.abs(f)
    mag[0] = 0  # ignore DC

    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("estimated baud rate: %f" % baud_rate)

    return baud_rate, freqs, mag

# Fonctions de slicing : binaire, 4-FSK, MFSK. Soft -> bits.
def slice_binary(symbols):
    """Convertit les symboles en bits 0 et 1"""
    symbols_average = np.average(symbols)
    bits = symbols >= symbols_average
    return np.array(bits, dtype=np.uint8)

def slice_4ary(symbols,mapping):
    """Convertit les symboles en bits doubles (00 à 11)"""
    # chaque symbole = 2 bits
    if len(symbols) == 0:
        return np.array([], dtype=np.uint8)
    symbols = np.asarray(symbols)
    # 4 niveaux de décision sur symboles. la fonction attend que les symboles soient déjà normalisés
    q25, q50, q75 = -0.50, 0.00, 0.50
    # 2 mappings, sinon custom
    mappings = {
        "natural": {
            0: (0, 0),  # Frequence la plus basse
            1: (0, 1),
            2: (1, 0),
            3: (1, 1)   # Frequence la plus haute
        },
        "gray": {
            0: (0, 0),  # Frequence la plus basse
            1: (0, 1),
            2: (1, 1),
            3: (1, 0)   # Frequence la plus haute
        }
    }
    # Vérification du mapping
    if not isinstance(mapping, list) and mapping not in mappings :
        raise ValueError("Mapping must be 'natural', 'gray', or a list of 4 tuples.")
    # Sélection du mapping
    if isinstance(mapping, list) and len(mapping) == 4:
        bit_map = {i: tuple(mapping[i]) for i in range(4)}
    else:
        bit_map = mappings.get(mapping, mappings["natural"])  # Defaut : natural
    # Assigne chaque symbole au niveau le plus proche
    mid = midpoint(symbols) # sépare les 2 niveaux hauts des 2 niveaux bas
    lower, upper = symbols[symbols <= mid], symbols[symbols > mid]
    mid_low  = midpoint(lower) if len(lower) > 1 else mid - 1 # sépare niveau 0 / niveau 1
    mid_high = midpoint(upper) if len(upper) > 1 else mid + 1 # sépare niveau 2 / niveau 3

    idx = np.select(
        [symbols <= mid_low, symbols <= mid, symbols <= mid_high],
        [0, 1, 2], default=3
    )
    bits = np.array([b for i in idx for b in bit_map[i]], dtype=np.uint8)
    return bits

def fm_demodulate(iq_signal, sample_rate):
    """Démodulation FM"""
    # Détection en quadrature
    phase = np.angle(iq_signal)
    return np.diff(np.unwrap(phase)) * sample_rate / (2 * np.pi)

def am_demodulate(iq_signal):
    """Démodulation AM"""
    # Détection d'enveloppe
    return np.abs(iq_signal)

def ssb_to_iq(iq, fs, sideband="USB"):
    """Convertit un signal SSB (USB/LSB) en IQ en supprimant la bande latérale non désirée."""
    sig_len = len(iq)
    spectrum = np.fft.fft(iq)
    freqs = np.fft.fftfreq(sig_len, 1/fs)

    if sideband == "USB":
        spectrum[freqs < 0] = 0
    elif sideband == "LSB":
        spectrum[freqs > 0] = 0
    else:
        raise ValueError("sideband must be 'USB' or 'LSB'")

    return np.fft.ifft(spectrum)

def cluster_freqs(freqs, weights=None, merge_hz=0.0):
    """Cluster des fréquences proches pour éviter les doublons de détection."""
    if len(freqs) == 0:
        return np.array([])
    # Tri des fréquences
    order = np.argsort(freqs)
    freqs = np.asarray(freqs, float)[order]
    # poids associés aux fréquences triées
    w = np.ones_like(freqs) if weights is None else np.asarray(weights, float)[order]

    clusters = []
    cf, cw = freqs[0], w[0] # fréquence centrale et poids du cluster en cours
    for f, ww in zip(freqs[1:], w[1:]):
        if abs(f - cf) <= merge_hz:
            # si FC suffisamment proche du cluster, on fusionne
            cf = (cf*cw + f*ww) / (cw + ww) # moy pondérée de la FC du cluster
            cw += ww # incrémentation du poids
        else:
            clusters.append((cf, cw)) # sinon, terminé et enregistre le cluster
            cf, cw = f, ww # on part sur un autre cluster
    clusters.append((cf, cw)) # ajout du dernier cluster
    clusters.sort(key=lambda x: -x[1]) # tri par poids décroissant
    return np.array([c[0] for c in clusters]) # renvoie uniquement les fréquences de clusters

def detect_and_track_mfsk_auto(
    iq, fs, baud,
    num_tones=None, # nb max de tons après clustering (si None = Tous)
    peak_thresh_db=8,   # seuil de détection par tranche en relatif du max de la tranche
    peak_prominence=None,   # optionnel : proéminence minimale (Si None, pas de contrainte)
    win_factor=1.0, # facteur de longueur de fenêtre, qui est : win_factor * (fs/baud)
    hop_factor=0.25,    # facteur de saut entre tranches : hop_factor * window
    merge_bins=1.2, # largeur de clusters pour fusion
    switch_penalty=0.05,    # pénalité Viterbi de switch relative à la puissance des tons
    manual_freqs=None   # optionnel : liste de fréquence à track (override auto-detection)
):
    """Détection de tons MFSK en bande de base et tracking dans le temps.
    - Par tranches temporelles (fenêtre Hann, détection en dB, estim F en interpol parabolique)
    - Fusion des fréquences proches (pour éviter doublons) et conservation N tons les plus forts
    - Suivi temporel ; filtre pour chaque ton candidat (type Goertzel), 
    calcul de puissance par tranche, lissage type Viterbi avec pénalité de changement de ton)"""

    # Fenetre d'analyse basée sur la durée symbole
    sym_len = fs / float(baud)
    win_len = max(16, int(round(win_factor * sym_len))) # min 16 échantillons
    hop_len = max(1, int(round(hop_factor * win_len))) # overlap 75% par défaut
    nfft = win_len
    freqbin_w = fs / nfft # largeur d'un bin

    # Grille fréquentielle
    samples_indices = np.arange(win_len) # indices dans la fenêtre
    win = 0.5 - 0.5*np.cos(2*np.pi*samples_indices/(win_len-1)) # fenêtre de Hann
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs))

    # Fréquences et poids associés, sur toutes les tranches
    obs_freqs = []
    obs_w = []
    # tranches temporelles
    for start in range(0, len(iq)-win_len+1, hop_len):
        bloc = iq[start:start+win_len] * win # block temporel fenêtré
        spec = np.fft.fftshift(np.fft.fft(bloc, nfft)) # fft du bloc
        mag = np.abs(spec)
        db = 20*np.log10(mag + 1e-12) # en dB + petite constante pour éviter log(0)
        mx = np.max(db) # max tranche en dB

        # Selections des pics dans la tolérance du max dB de la tranche
        height = mx - float(peak_thresh_db)
        # Si proéminence définie, elle est imposée
        if peak_prominence is not None:
            peaks, _ = find_peaks(db, height=height, prominence=peak_prominence)
        else:
            peaks, _ = find_peaks(db, height=height)

        for k in peaks:
            # interpolation parabolique autour du pic
            delta = parabolic_interpol(db, k) # décalage fractionnaire en bin
            f_est = freqs[k] + delta * freqbin_w # estimation de freq en Hz
            obs_freqs.append(f_est)
            # Poids du pic : puissance linéaire 
            obs_w.append(mag[k]**2)

    # Seuil de fusion des fréquences proches
    merge_hz = max(freqbin_w * merge_bins, 0.5)  # au moins 0.5 Hz pour éviter une fragmentation excessive
    if manual_freqs is not None:
        # mode manuel : on suit directement les fréquences fournies
        tones_all = np.array(manual_freqs, float)
    else:
        # mode auto : on regroupe les fréquences détectées
        tones_all = cluster_freqs(obs_freqs, weights=obs_w, merge_hz=merge_hz)
    # conservation des N tons les plus forts (poids décroissants)
    if num_tones is not None and len(tones_all) > num_tones:
        tones_all = tones_all[:num_tones]
    # fréquences des tons suivis
    tone_freqs = np.sort(tones_all)
    # nb de tons suivis
    nb_tons = len(tone_freqs)
    if nb_tons == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.empty((0,0)), 0.0

    # Filtres adaptés pour chaque ton candidat
    # Fenêtre normalisée en énergie pour rendre les filtres compatibles
    norm_win = win / np.sqrt(np.sum(win**2) + 1e-12)
    # oscillateur par ton
    matched_filters = np.exp(-1j * 2*np.pi * (tone_freqs[:, None] / fs) * samples_indices) * norm_win

    starts = np.arange(0, len(iq)-win_len+1, hop_len)
    num_frames = len(starts)
    # puissance de chaque ton candidat pour chaque tranche
    tone_pows = np.empty((num_frames, nb_tons), float)
    # temps centraux des tranches (s)
    times = np.empty(num_frames, float)

    for i, s in enumerate(starts):
        # normalise un bloc temporel par son RMS
        rms = np.sqrt(np.mean(np.abs(iq[s:s+win_len])**2) + 1e-12) # RMS avec petit offset
        block = iq[s:s+win_len] / rms
        # sortie des filtres adaptés
        matched_fil_output = (matched_filters * block).sum(axis=1)
        # puissance du ton candidat sur cette tranche
        tone_pows[i] = np.abs(matched_fil_output)**2
        # temps central de la tranche
        times[i] = (s + (win_len-1)/2) / fs

    # Lissage temporel type Viterbi. Pour éviter de changer de ton trop souvent à cause du bruit, 
    # on pénalise le changement tout en favorisant le ton le plus puissant de chaque instant
    dp = np.empty_like(tone_pows) # table de programmation dynamique
    back = np.empty_like(tone_pows, dtype=np.int32) # backpointers
    dp[0] = tone_pows[0]
    back[0] = -1
    # matrice de pénalité de changements de tons
    if nb_tons > 1:
        pen = np.ones((nb_tons, nb_tons)) * switch_penalty
        np.fill_diagonal(pen, 0.0) # pas de pénalité en restant sur le même

    for i in range(1, num_frames):
        # si in seul candidat
        if nb_tons == 1:
            dp[i] = dp[i-1] + tone_pows[i]
            back[i] = 0
        # cas général : plusieurs candidats
        else:
            prev_scores = dp[i-1][:, None] # scores des tons à la tranche précédente
            transition_scores = prev_scores - pen # scores de transition de previous_t
            best_prev_tone_ind = np.argmax(transition_scores, axis=0) # meilleur ton précédent pour chaque ton courant
            # meilleurs scores associés
            best_prev_scores = transition_scores[best_prev_tone_ind, np.arange(nb_tons)]
            dp[i] = best_prev_scores + tone_pows[i] # màj avec scores cumulés
            # sauvegarde des pointeurs arrières pour backtrack
            back[i] = best_prev_tone_ind

    # Backtrack : on remonte depuis la meilleure tranche finale
    best_final_tone_idx = int(np.argmax(dp[-1]))
    tone_idx = np.empty(num_frames, dtype=np.int32) # index du ton sélectionné par tranche
    for i in range(num_frames-1, -1, -1):
        tone_idx[i] = best_final_tone_idx
        best_final_tone_idx = back[i, best_final_tone_idx] if back[i, best_final_tone_idx] >= 0 else best_final_tone_idx

    tone_trace = tone_freqs[tone_idx]

    # Estimation de durée symbole à partir des durées de séquences de tons
    avg_frame_step_s = np.mean(np.diff(times))
    runs = []
    start = 0
    # suite de tranches consécutives où le même ton est sélectionné
    for i in range(1, len(tone_idx)):
        if tone_idx[i] != tone_idx[i-1]:
            runs.append(i - start) # longueur de "run" en tranches
            start = i
    runs.append(len(tone_idx) - start) # ajout de dernière séquence

    # Fusion des séquences isolées d'une seule tranche
    merged = []
    i = 0
    # une séquence de longueur 1 peut être du bruit, on l'absorbe dans la précédente quand possible
    while i < len(runs):
        if runs[i] == 1 and i > 0:
            merged[-1] += 1
        else:
            merged.append(runs[i])
        i += 1

    # Estimation de débit symbole avec la médiane des longueurs de séquences
    if len(merged) > 0:
        avg_run = np.median(merged)
        measured_rate = 1.0 / (avg_run * avg_frame_step_s)
    else:
        measured_rate = 0.0

    return tone_freqs, times, tone_idx, tone_trace, tone_pows, measured_rate

def estimate_phase0(tone_idx, frames_per_symbol, resolution=0.1):
    """Cherche le décalage de grille qui aligne le mieux une grille périodique de pas
    frames_per_symbol sur les transitions observées dans tone_idx (sans info externe)."""
    transitions = np.zeros(len(tone_idx))
    transitions[1:] = (tone_idx[1:] != tone_idx[:-1]).astype(float)
    n_test = max(8, int(round(frames_per_symbol / resolution)))
    phases = np.linspace(0, frames_per_symbol, n_test, endpoint=False)
    best_phase, best_score = 0.0, -1.0
    for phase in phases:
        grid = np.round(np.arange(phase, len(tone_idx), frames_per_symbol)).astype(int)
        grid = grid[(grid >= 0) & (grid < len(tone_idx))]
        score = transitions[grid].sum()
        if score > best_score:
            best_score, best_phase = score, phase
    return best_phase

def majority_vote_resample(tone_idx, frames_per_symbol, K, phase0=None):
    """Regroupe tone_idx en fenêtres de frames_per_symbol, vote majoritaire par fenêtre."""
    if phase0 is None:
        phase0 = estimate_phase0(tone_idx, frames_per_symbol)
    out, pos, T = [], phase0, len(tone_idx)
    while int(round(pos)) < T:
        start = int(round(pos))
        end = min(int(round(pos + frames_per_symbol)), T)
        if start >= end:
            break
        out.append(np.argmax(np.bincount(tone_idx[start:end], minlength=K)))
        pos += frames_per_symbol
    return np.array(out), phase0

def slice_mfsk(symbols, order, mapping="natural", return_format="binary"):
    """Convertit en bits selon le mapping spécifié, selon en entrée
    des positions normalisées de -1 à +1."""
    symbols = np.asarray(symbols, dtype=float)
    if len(symbols) == 0:
        return np.array([], dtype=np.uint8 if return_format == "binary" else int)
    if order < 2:
        raise ValueError("Order must be ≥ 2.")
    # Assigne chaque symbole au niveau le plus proche
    levels = np.linspace(-1, 1, order)
    indices = np.argmin(np.abs(symbols[:, None] - levels[None, :]), axis=1)
    # fonc pour obtenir les bits ou caractères au format voulu, selon les indices
    return indices_to_output(indices, order, mapping, return_format)

def indices_to_output(indices, order, mapping="natural", return_format="binary"):
    """Convertit des indices déjà entiers (0..order-1) en bits ou caractères."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return np.array([], dtype=np.uint8 if return_format == "binary" else int)
    if return_format == "int":
        return indices
    if return_format == "char":
        charset = string.digits + string.ascii_uppercase + string.ascii_lowercase + string.punctuation
        if order > len(charset):
            return np.array([format(i, "X") for i in indices], dtype=object)
        return np.array([charset[i] for i in indices], dtype=object)
    num_bits = int(np.ceil(np.log2(order)))
    def to_gray(n): return n ^ (n >> 1)
    def bin_tuple(n): return tuple(int(b) for b in np.binary_repr(n, width=num_bits))
    if isinstance(mapping, list):
        if len(mapping) != order:
            raise ValueError("Mapping list must have the same length as order.")
        bit_map = {i: tuple(mapping[i]) for i in range(order)}
    elif mapping == "gray":
        bit_map = {i: bin_tuple(to_gray(i)) for i in range(order)}
    else:
        bit_map = {i: bin_tuple(i) for i in range(order)}
    return np.array([b for idx in indices for b in bit_map[idx]], dtype=np.uint8)

def constel_eye_sampling_offset(x, sps):
    """Instant d'échantillonnage (dans [0, sps)) maximisant la moyenne de |x|."""
    cands = np.arange(0, sps, max(sps / 32.0, 0.25))
    ks = np.arange(0, int((len(x) - sps - 1) // sps))
    if len(ks) < 8:
        return 0.0
    scores = []
    for c in cands:
        pos = c + ks * sps
        i = np.floor(pos).astype(int); mu = pos - i
        scores.append(np.mean((1 - mu) * x[i] + mu * x[i + 1]))
    return float(cands[int(np.argmax(scores))])

def constel_points(x, fs, symbol_rate):
    """Renvoie les points de constellation normalisés et le seuil de visualisation"""
    sps = fs / symbol_rate if symbol_rate is not None else None
    if sps is not None and sps >= 2:
        tau = constel_eye_sampling_offset(np.abs(x), sps)
        k = np.arange(0, int((len(x) - sps - 1) // sps))
        pos = tau + k * sps
        i = np.floor(pos).astype(int); mu = pos - i
        pts = (1 - mu) * x[i] + mu * x[i + 1]
    else:
        pts = x
    scale = np.percentile(np.abs(pts), 99.9)
    pts = pts / (scale if scale > 0 else 1.0)
    
    return pts

def find_active_region(x, sps, margin_frac=0.1):
    """(start, end) de la plus grande région de signal,
    ou signal entier si pas de contraste clair"""
    n = len(x)
    L = max(int(4 * sps), 1)
    env = np.convolve(np.abs(x) ** 2, np.ones(L) / L, mode="same")
    lo, hi = np.percentile(env, [10, 90])
    if hi < 2 * lo: # seulement signal continu (ou bruit)
        return 0, n
    active = env > 0.5 * (lo + hi)
    d = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    j = np.argmax(ends - starts)
    s, e = starts[j], ends[j]
    m = int(margin_frac * (e - s))# drop ramp up/down edges
    return s + m, e - m

def eye_diagram(samples, fs, baud_rate, mod_order, channel="I", num_traces=500,
                    symbols_per_trace=2, costas_bw_factor=0.01, skip_frac=0.3, align=True):
    """Trace un diagramme de l'oeil"""
    lock = None
    x = np.asarray(samples)
    sps = fs / float(baud_rate)
    s, e = find_active_region(x, sps)
    x = x[s:e] 
    if np.iscomplexobj(x):
        if mod_order in (2, 4): # Costas seulement si supporté
            x, lock = costas_loop(x, fs, baud_rate*costas_bw_factor, order=mod_order)
        ch = channel.upper()
        if   ch == "I":   sig = x.real
        elif ch == "Q":   sig = x.imag
        elif ch == "MAG": sig = np.abs(x)
        else: raise ValueError("channel must be 'I', 'Q', or 'mag'")
    else:
        sig = x
    # Estimation du nombre de points par symbole
    n_points = int(max(2, round(symbols_per_trace * sps)))
    time_axis = np.linspace(0, symbols_per_trace, n_points, endpoint=False)
    required = symbols_per_trace * sps
    n = len(sig)

    # fenêtre utile : on écarte le début de la sortie de Costas et on prend la fin
    first = int(skip_frac * n) if lock is not None else 0
    need = (num_traces + symbols_per_trace) * sps
    first = max(first, int(n - need))
    first = max(first, 0)

    # alignement : instant d'échantillonnage tau maximisant la moyenne de |sig|
    tau = 0.0
    if align:
        tau = constel_eye_sampling_offset(np.abs(sig[first:]), sps)
    start0 = first + tau - sps/2.0 # le pic tombe à t = 0.5 symbole
    if start0 < first: 
        start0 += sps

    max_traces = int(np.floor((n - 1 - start0 - required) / sps)) + 1
    k = max(0, min(int(num_traces), max_traces))
    if k == 0:
        return time_axis, np.empty((0, n_points)), lock
    step = required / n_points
    pos = (start0 + np.arange(k)*sps)[:, None] + (np.arange(n_points)*step)[None, :]
    i = np.floor(pos).astype(int); mu = pos - i
    traces = (1-mu)*sig[i] + mu*sig[i+1]
    return time_axis, traces, lock

def costas_loop(x, fs, loop_bandwidth, order=2, damping=0.707):
    """Boucle de Costas : verrouillage de phase"""
    # Normalise amplitude & bw
    rms = np.sqrt(np.mean(np.abs(x)**2))
    x = x / (rms + 1e-12)
    Bn = loop_bandwidth / fs
    zeta = damping  # damping
    # Coefficients
    denom = 1 + 2*zeta*Bn + Bn**2
    Kp = (4*zeta*Bn) / denom
    Ki = (4*Bn**2) / denom

    phase = 0.0
    freq_acc = 0.0
    out = []
    for sample in x:
        mixed = sample * np.exp(-1j*phase)
        # Detection d'erreur (BPSK/QPSK)
        if order == 2:  # BPSK
            error = np.real(mixed) * np.imag(mixed)
        elif order == 4:  # QPSK
            # error = np.imag(mixed**4)
            error = np.sign(mixed.real) * mixed.imag - np.sign(mixed.imag) * mixed.real
        else:
            raise ValueError("Only BPSK (2) and QPSK (4) supported.")

        # filtre de la boucle (contrôleur P&I)
        freq_acc += Ki * error
        phase += freq_acc + Kp * error
        # Wrap phase
        if phase > np.pi:
            phase -= 2*np.pi
        elif phase < -np.pi:
            phase += 2*np.pi

        out.append(mixed)

    # Métrique de verrouillage
    if len(out) == 0:
        coherence = 0.0
    else:
        start = len(out) // 4
        tail = np.asarray(out[start:], dtype=np.complex128)
        mag = np.abs(tail)
        valid = mag > 1e-12

        if not np.any(valid):
            coherence = 0.0
        else:
            phase_norm = tail[valid] / mag[valid]
            coherence = float(np.abs(np.mean(phase_norm ** order)))
            coherence = float(np.clip(coherence, 0.0, 1.0))

    return np.array(out), coherence

def psk_demodulate(sig, fs, symbol_rate, order=2,
                   gray=False, differential=False, offset=False, pi4=False,
                   costas_damping=0.707, costas_bw_factor=0.01, 
                   timing_recovery=True,loop_gain=0.01, direct_symbols=None):
    """Démodulation BPSK/QPSK"""
    sps_exact = fs / symbol_rate
    bw_loop = symbol_rate * costas_bw_factor
    if direct_symbols is None:
        if timing_recovery:
            symbols, positions, _ = gardner_timing_recovery(sig, fs, symbol_rate, loop_gain, return_debug=True)
        else:
            n_symbols = int(round(len(sig) / sps_exact))
            positions = np.array([k * sps_exact + sps_exact / 2 for k in range(n_symbols)])
            positions = positions[positions < len(sig) - 1]
            symbols = sig[np.round(positions).astype(int)]
        
        if order == 4 and offset and not pi4:
            symbols, coarse_lock_score = oqpsk_recover(sig, positions, sps_exact, symbol_rate, bw_loop, costas_damping)
        else:
            symbols = symbols / (np.sqrt(np.mean(np.abs(symbols)**2)) + 1e-12)
            if pi4:
                rot = np.exp(-1j*np.pi/4*(np.arange(len(symbols)) % 2))
                symbols, coarse_lock_score = costas_loop(symbols*rot, symbol_rate, bw_loop, order=4, damping=costas_damping)
                symbols = symbols * np.conj(rot)
            else:
                symbols, coarse_lock_score = costas_loop(symbols, symbol_rate, bw_loop, order=order, damping=costas_damping)

        if len(symbols) == 0:
            return np.array([]), coarse_lock_score
    else:
        symbols = direct_symbols

    lock_score = symbol_coherence(symbols[len(symbols)//4:], 8 if pi4 else order)

    if order == 2 and differential is False: # BPSK
        bits = (np.real(symbols) > 0).astype(int)

    elif order == 2 and differential is True: # DBPSK
        bits = (np.real(symbols[1:] * np.conj(symbols[:-1])) < 0).astype(np.uint8)

    elif order == 4 and differential is False and offset is False and pi4 is False: # QPSK
        bits = []
        for s in symbols:
            if s.real >= 0 and s.imag >= 0:
                dibit = (0,0) if gray else (0,0)  # pareil gray ou naturel
            elif s.real < 0 and s.imag >= 0:
                dibit = (0,1) if gray else (0,1)
            elif s.real < 0 and s.imag < 0:
                dibit = (1,1) if gray else (1,0)
            else:
                dibit = (1,0) if gray else (1,1)
            bits.extend(dibit)

    elif order == 4 and offset is True and differential is False and pi4 is False:  # OQPSK
        I = (symbols.real > 0).astype(np.uint8)
        Q = (symbols.imag > 0).astype(np.uint8)
        if not gray:
            Q = Q ^ I
        bits = np.empty(2 * len(I), dtype=np.uint8)
        bits[0::2] = I
        bits[1::2] = Q

    elif order == 4  and differential is True and pi4 is False: # DQPSK
        diffs = symbols * np.conj(np.roll(symbols, 1))
        phases = np.angle(diffs)
        bits = []

        # tables de mapping
        phase_bits_gray = {0: (0,0),  np.pi/2: (0,1),  np.pi: (1,1), -np.pi/2: (1,0)}
        phase_bits_nat  = {0: (0,0),  np.pi/2: (0,1),  np.pi: (1,0), -np.pi/2: (1,1)}
        mapping = phase_bits_gray if gray else phase_bits_nat

        for ph in phases[1:]:
            # Normalisation de phase [-pi, pi]
            ph = (ph + np.pi) % (2*np.pi) - np.pi
            # Map au quadrant le plus proche
            if -np.pi/4 <= ph < np.pi/4:
                bits.extend(mapping[0])
            elif np.pi/4 <= ph < 3*np.pi/4:
                bits.extend(mapping[np.pi/2])
            elif -3*np.pi/4 <= ph < -np.pi/4:
                bits.extend(mapping[-np.pi/2])
            else:
                bits.extend(mapping[np.pi])

    elif order == 4 and pi4:  # π/4-QPSK
        # On gère le cas général qui est en fait π/4-DQPSK
        diffs = symbols * np.conj(np.roll(symbols, 1))
        phases = np.angle(diffs)

        bits = []
        # Table de mapping π/4
        phase_bits = {
            np.pi/4:  (0,0) if gray else (0,0),
            3*np.pi/4:(0,1) if gray else (0,1),
            -3*np.pi/4:(1,1) if gray else (1,0),
            -np.pi/4:(1,0) if gray else (1,1)
        }

        for ph in phases[1:]:
            ph = (ph + np.pi) % (2*np.pi) - np.pi
            if -np.pi/2 < ph <= 0:
                dibit = phase_bits[-np.pi/4]
            elif -np.pi <= ph <= -np.pi/2:
                dibit = phase_bits[-3*np.pi/4]
            elif 0 < ph <= np.pi/2:
                dibit = phase_bits[np.pi/4]
            else:
                dibit = phase_bits[3*np.pi/4]

            bits.extend(dibit)
    else:
        raise ValueError("Unsupported combination of order/differential/offset/pi4.")

    return np.array(bits, dtype=np.uint8), lock_score

def symbol_coherence(symbols, order):
    v = np.abs(symbols) > 1e-12
    return float(np.abs(np.mean((symbols[v] / np.abs(symbols[v])) ** order)))

def interp_complex(x, pos):
    """Interpolation linéaire complexe en positions fractionnaires (pos dans [0, len(x)-2])."""
    pos = np.asarray(pos, dtype=float)
    i = np.floor(pos).astype(int)
    mu = pos - i
    return (1.0 - mu) * x[i] + mu * x[i + 1]

def oqpsk_recover(sig, positions, sps, symbol_rate, bw_loop, damping):
    """Positions Gardner -> symboles OQPSK verrouillés (I + jQ, ~ +/- 1 chaque)."""
    half = sps / 2.0
    p = positions[(positions >= 0) & (positions + half < len(sig) - 1)]
    if len(p) < 4:
        return np.array([], dtype=complex), 0.0

    z = interp_complex(sig, p) # ~ +/- I·e^{jθ}
    w = -1j * interp_complex(sig, p + half) # ~ +/- Q·e^{jθ}

    u = np.empty(2 * len(p), dtype=complex) # stream BPSK-like à 2*Rs
    u[0::2] = z
    u[1::2] = w

    u, lock = costas_loop(u, 2 * symbol_rate, bw_loop,
                          order=2, damping=damping)
    return u.real[0::2] + 1j * u.real[1::2], lock

def gardner_timing_recovery(x, fs, symbol_rate, loop_gain=0.05, return_debug=False, t0=None):
    """TED (Gardner) pour la récupération du timing avec interpolation linéaire complexe.
    Suppose débit connu et suit principalement le timing fractionnel des symboles, 
    pas l'erreur de débit."""
    x = np.asarray(x)

    if fs <= 0 or symbol_rate <= 0:
        raise ValueError("fs & symbol_rate must be > 0")
    if len(x) < 4:
        empty_c = np.array([], dtype=complex)
        if return_debug:
            return (empty_c,np.array([], dtype=float),np.array([], dtype=float))
        return empty_c
    
    sps = fs / float(symbol_rate)
    if sps < 2.0:
        raise ValueError("Gardner timing recovery need >= 2 samples/symbol")

    symbols = []
    positions = []
    errors = []
    # Estimation initiale du timing : 
    if t0 is None:
        tau, conf = estimate_timing_offset(x, sps)
        if conf > 0.05:
            t = tau
            if t < sps / 2.0: # Gardner a besoin de : t - sps/2 >= 0
                t += sps
        else:
            t = sps / 2.0
    else:
        t = t0

    while True:
        early_pos = t - sps / 2.0
        late_pos = t + sps / 2.0
        if early_pos < 0:
            t += sps
            continue
        if late_pos >= len(x) - 1:
            break

        early = interp_complex(x,early_pos)
        center = interp_complex(x,t)
        late = interp_complex(x,late_pos)
        if early is None or center is None or late is None:
            break

        error = np.real((early - late) * np.conj(center))
        # normalisation de l'erreur par l'énergie du symbole pour éviter les problèmes de gain variable
        energy = (np.abs(early)**2 + np.abs(center)**2 + np.abs(late)**2 + 1e-12)
        error /= energy

        symbols.append(center)
        positions.append(t)
        errors.append(error)
        # mise à jour du timing, 1 symbole + correction
        t += sps - loop_gain * sps * error

    symbols = np.asarray(symbols)
    positions = np.asarray(positions)
    errors = np.asarray(errors)
    if return_debug:
        return symbols, positions, errors
    
    return symbols

def estimate_timing_offset(x, sps, n_symbols=512):
    """Estimation de timing symbole (Oerder-Meyr, non linéarité quadratique).
    Retourne (tau, confiance) avec tau entre [0, sps] et confiance [0, 1]"""
    n = min(len(x), int(n_symbols * sps))
    n = max(n, int(4 * sps))
    p = np.abs(x[:n]) ** 2
    p = p - np.mean(p)
    k = np.arange(n)
    X = np.sum(p * np.exp(-2j * np.pi * k / sps))
    tau = (-np.angle(X) / (2 * np.pi)) * sps
    tau %= sps
    conf = np.abs(X) / (np.sum(np.abs(p)) + 1e-12)
    return tau, conf