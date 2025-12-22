"""Fonctions de démodulation et/ou d'estimation du débit symbole. 
WPCR pour demod NRZ basée sur les fonctions de Michael Ossmann, "clock recovery experiments" : https://github.com/mossmann/clock-recovery"""
import numpy as np
import scipy.signal as signal
import string


def find_clock_frequency(fdiff, sample_rate, target_rate=None, precision=0.9):
    """Détermine la fréquence d'horloge"""
    maxima = signal.argrelextrema(fdiff, np.greater_equal)[0] # indices des maxima locaux
    while maxima[0] < 2:
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

    threshold = max(fdiff[2:-1]) * 0.8
    indices_above_threshold = np.argwhere(fdiff[maxima] > threshold)

    # renvoie le premier pic au-dessus du seuil
    return int(maxima[indices_above_threshold[0]]) if indices_above_threshold.size > 0 else 0

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
        return []
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
        return 0
    # FFT des transitions
    f = np.fft.fft(d, len(a))
    p = find_clock_frequency(abs(f), sample_rate, target_rate, precision)
    if p == 0:
        return 0
    # conversions index des bins en Hz
    baud_rate = p * sample_rate / len(f)
    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("estimated baud rate: %f" % baud_rate)
    return baud_rate

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
    # 4 niveaux de décision sur les symboles. la fonction attend que les symboles soient déjà normalisés
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
    bit_pairs = []
    for sym in symbols:
        if sym < q25:
            bit_pairs.append(bit_map[0])
        elif sym < q50:
            bit_pairs.append(bit_map[1])
        elif sym < q75:
            bit_pairs.append(bit_map[2])
        else:
            bit_pairs.append(bit_map[3])

    # Flatten en array de bits
    bits = np.array([b for pair in bit_pairs for b in pair], dtype=np.uint8)
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

def slice_mfsk(symbols, order, mapping="natural", return_format="binary"):
    """Convertit les symboles MFSK en bits selon le mapping spécifié."""
    symbols = np.asarray(symbols, dtype=float)
    if len(symbols) == 0:
        return np.array([], dtype=np.uint8 if return_format == "binary" else int)
    if order < 2:
        raise ValueError("Order must be ≥ 2.")

    # Assigne chaque symbole au niveau le plus proche
    levels = np.linspace(-1, 1, order)
    indices = np.argmin(np.abs(symbols[:, None] - levels[None, :]), axis=1)

    # Option pour renvoyer les indices en format entier
    if return_format == "int":
        return indices.astype(int)
    
    # Option pour renvoyer les indices en caractères : jusqu'à 94 symboles différents sur un seul caractère
    if return_format == "char":
        charset = (
            string.digits +                # 0-9
            string.ascii_uppercase +       # A-Z
            string.ascii_lowercase +       # a-z
            string.punctuation             # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        )
        # si on dépasse le charset : retombe sur format hexadécimal
        if order > len(charset):
            return np.array([format(i, "X") for i in indices], dtype=object)
        else:
            return np.array([charset[i] for i in indices], dtype=object)

    # Mapping des indices en bits
    num_bits = int(np.ceil(np.log2(order)))

    def to_gray(n): return n ^ (n >> 1)
    def bin_tuple(n): return tuple(int(b) for b in np.binary_repr(n, width=num_bits))

    if isinstance(mapping, list):
        if len(mapping) != order:
            raise ValueError("Mapping list must have the same length as order.")
        bit_map = {i: tuple(mapping[i]) for i in range(order)}
    elif mapping == "gray":
        bit_map = {i: bin_tuple(to_gray(i)) for i in range(order)}
    else:  # naturel par défaut
        bit_map = {i: bin_tuple(i) for i in range(order)}

    bits = np.array([b for idx in indices for b in bit_map[idx]], dtype=np.uint8)
    return bits

def parabolic_interpol(mag, k):
    # Parabolic peak interpolation around bin k (on linear power or magnitude)
    # Returns fractional bin offset delta in [-0.5, 0.5] roughly
    if k <= 0 or k >= len(mag)-1:
        return 0.0
    a = mag[k-1]
    b = mag[k]
    c = mag[k+1]
    denom = (a - 2*b + c) # 2nd derivative approx
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (a - c) / denom # vertex of the parabola
    delta = np.clip(delta, -1.0, 1.0) # avoid crazy offsets

    return delta

def cluster_freqs(freqs, weights=None, merge_hz=0.0):
    if len(freqs) == 0:
        return np.array([])
    order = np.argsort(freqs)
    freqs = np.asarray(freqs, float)[order]
    w = np.ones_like(freqs) if weights is None else np.asarray(weights, float)[order] # sorted weights

    clusters = []
    cf, cw = freqs[0], w[0] # current freq & weight
    for f, ww in zip(freqs[1:], w[1:]):
        if abs(f - cf) <= merge_hz:
            # merge
            cf = (cf*cw + f*ww) / (cw + ww) # weighted average
            cw += ww # total weight
        else:
            clusters.append((cf, cw)) # save previous
            cf, cw = f, ww # start new cluster
    clusters.append((cf, cw))
    clusters.sort(key=lambda x: -x[1])  # by weight
    return np.array([c[0] for c in clusters]) # return only frequencies

def detect_and_track_mfsk_auto( # Méthode à évaluer sur plusieurs exemples de MFSK différents (espacement, nombre de tons, SNR, dérive freq, etc)
    iq, fs, baud,
    num_tones=None,              # keep top-N tones after clustering (None=all)
    peak_thresh_db=8,            # per-frame relative threshold
    peak_prominence=None,        # optional: e.g. 3 (dB) : converted internally
    win_factor=1.0,              # window length : win_factor * (fs/baud)
    hop_factor=0.25,             # hop : hop_factor * window
    merge_bins=1.2,              # cluster width in bins for tone dedup
    switch_penalty=0.05          # Viterbi penalty (linear power units)
): 
    """
    Auto-detect baseband MFSK tones (±freq) and track them over time.
    - Uses sub-bin (parabolic) peak interpolation for tone detection
    - Clusters near-duplicate frequencies
    - Tracks via matched complex oscillators (Goertzel-style) + Viterbi
    """

    # Analysis window & hop based on symbol duration
    sym_len = fs / float(baud) # fractional allowed, no strict alignment needed
    N = max(16, int(round(win_factor * sym_len)))
    H = max(1, int(round(hop_factor * N)))
    nfft = N
    df = fs / nfft

    # Window & frequency grid
    n = np.arange(N)
    win = 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))   # Hann window
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs))

    # Tone detection over full spectrum with sub-bin interpolation
    obs_freqs = []
    obs_w = []

    for start in range(0, len(iq)-N+1, H): # frame start
        blk = iq[start:start+N] * win # windowed block
        spec = np.fft.fftshift(np.fft.fft(blk, nfft))
        mag = np.abs(spec)
        db = 20*np.log10(mag + 1e-12) # avoid log(0)
        mx = np.max(db) # frame max in dB

        # Select all peaks within threshold of the frame's max
        height = mx - float(peak_thresh_db)
        # Optional prominence in dB -> convert to linear magnitude-ish gate by comparing to local baseline
        if peak_prominence is not None:
            # Use scipy's prominence on dB directly (approx OK)
            peaks, _ = signal.find_peaks(db, height=height, prominence=peak_prominence)
        else:
            peaks, _ = signal.find_peaks(db, height=height)

        for k in peaks:
            # Sub-bin interpolation
            delta = parabolic_interpol(mag, k) # on linear magnitude
            f_est = freqs[k] + delta * df # estimated freq in Hz
            obs_freqs.append(f_est)
            # Weight: use linear power as weight
            obs_w.append(mag[k]**2)

    # Cluster close frequencies so each tone appears once
    merge_hz = max(df * merge_bins, 0.5)  # at least 0.5 Hz to avoid crazy fragmentation
    tones_all = cluster_freqs(obs_freqs, weights=obs_w, merge_hz=merge_hz)
    if num_tones is not None and len(tones_all) > num_tones:
        tones_all = tones_all[:num_tones]
    tone_freqs = np.sort(tones_all)
    K = len(tone_freqs)
    if K == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.empty((0,0))

    # Tracking with matched oscillators (bin-free; exact Hz ok)
    # Energy-normalized window
    win_e = win / np.sqrt(np.sum(win**2) + 1e-12)
    # Oscillators per tone (K x N)
    osc = np.exp(-1j * 2*np.pi * (tone_freqs[:, None] / fs) * n) * win_e

    starts = np.arange(0, len(iq)-N+1, H)
    T = len(starts)
    tone_pows = np.empty((T, K), float) # per-frame tone powers
    times = np.empty(T, float)

    def agc_block(x):
        r = np.sqrt(np.mean(np.abs(x)**2) + 1e-12) # RMS with small offset
        return x / r

    for i, s in enumerate(starts):
        blk = agc_block(iq[s:s+N])
        c = (osc * blk).sum(axis=1) # matched filter outputs
        tone_pows[i] = np.abs(c)**2
        times[i] = (s + (N-1)/2) / fs # center time of frame

    # Viterbi-like smoothing with simple switch penalty
    dp = np.empty_like(tone_pows) # dynamic programming table
    back = np.empty_like(tone_pows, dtype=np.int32) # backpointers
    dp[0] = tone_pows[0]
    back[0] = -1
    if K > 1:
        pen = np.ones((K, K)) * switch_penalty # penalty matrix
        np.fill_diagonal(pen, 0.0) # no penalty for staying on same tone

    for i in range(1, T):
        if K == 1:
            dp[i] = dp[i-1] + tone_pows[i]
            back[i] = 0
        else:
            prev = dp[i-1][:, None] # (K,1)
            scores = prev - pen # (K,K): from j→k
            j_star = np.argmax(scores, axis=0) # best previous tone for each current tone
            best_prev = scores[j_star, np.arange(K)] # best scores
            dp[i] = best_prev + tone_pows[i] # update dp
            back[i] = j_star

    # Backtrack
    kT = int(np.argmax(dp[-1])) # best final tone
    tone_idx = np.empty(T, dtype=np.int32) # tone index per frame
    for i in range(T-1, -1, -1):
        tone_idx[i] = kT
        kT = back[i, kT] if back[i, kT] >= 0 else kT

    tone_trace = tone_freqs[tone_idx]

    # Estimate actual symbol duration from tone runs
    dt = np.mean(np.diff(times))  # average time step between frames
    runs = []
    start = 0
    for i in range(1, len(tone_idx)):
        if tone_idx[i] != tone_idx[i-1]:
            runs.append(i - start) # run length in frames
            start = i
    runs.append(len(tone_idx) - start) # last run

    if len(runs) > 0:
        avg_run = np.median(runs) # median = robust against outliers
        measured_rate = 1.0 / (avg_run * dt)
    else:
        measured_rate = 0.0

    return tone_freqs, times, tone_idx, tone_trace, tone_pows, measured_rate

def eye_diagram_with_metrics(samples, fs, baud_rate, channel="I", num_traces=500, symbols_per_trace=2):
    # A travailler : correction de timing de phase
    """Trace un diagramme de l'oeil avec métriques associées"""
    # Sélection du canal
    if np.iscomplexobj(samples):
        ch = channel.upper()
        if ch == "I":
            sig = samples.real
        elif ch == "Q":
            sig = samples.imag
        elif ch == "MAG":
            sig = np.abs(samples)
        else:
            raise ValueError("channel must be 'I', 'Q', or 'mag'")
    else:
        sig = np.asarray(samples)

    # Estimation du nombre de points par symbole
    sps = fs / float(baud_rate)
    n_points = int(max(2, round(symbols_per_trace * sps)))
    time = np.linspace(0, symbols_per_trace, n_points, endpoint=False)

    # Détermine le nombre de traces à extraire
    sig_len = len(sig)
    required_samples = symbols_per_trace * sps
    max_possible_traces = int(np.floor((sig_len - required_samples) / sps)) + 1
    num_traces = max(0, min(int(num_traces), max_possible_traces))

    # Extraction des segments avec interpolation
    traces = []
    idx = 0.0
    for _ in range(num_traces):
        start = idx
        stop = idx + required_samples
        if stop > sig_len - 1e-9:
            break
        xp = np.linspace(start, stop, n_points, endpoint=False)
        segment = np.interp(xp, np.arange(sig_len), sig)
        traces.append(segment)
        idx += sps

    metrics = {"eye_height": None, "eye_width": None, "eye_opening_ratio": None}

    if len(traces) == 0:
        return np.empty((0, n_points)), metrics

    traces = np.array(traces)

    # Métriques
    # Hauteur de l'oeil au milieu du symbole
    mid_idx = int(round(0.5 * sps))
    mid_idx = np.clip(mid_idx, 0, n_points - 1)
    mids = traces[:, mid_idx]
    low = np.percentile(mids, 25)
    high = np.percentile(mids, 75)
    eye_height = float(high - low)
    metrics["eye_height"] = eye_height

    # Largeur de l'oeil : moyenne des largeurs à mi-hauteur
    threshold = 0.5 * (np.max(traces) + np.min(traces))
    widths = []
    for seg in traces:
        above = seg > threshold
        diff = np.diff(above.astype(int))
        cross_points = np.where(diff != 0)[0]
        if cross_points.size >= 2:
            # complet, prend la distance entre le premier et le dernier
            width_symbols = (cross_points[-1] - cross_points[0]) * (symbols_per_trace / n_points)
            widths.append(width_symbols)
        elif cross_points.size == 1:
            # partiel, prend la distance entre le point de croisement et le bord le plus proche
            width_symbols = cross_points[0] * (symbols_per_trace / n_points)
            widths.append(width_symbols)
        else:
            # pas de croisement (durée complète ou nulle)
            widths.append(symbols_per_trace)
    eye_width = float(np.mean(widths)) if widths else None
    metrics["eye_width"] = eye_width

    # Ratio d'ouverture de l'oeil
    sig_min = np.min(traces)
    sig_max = np.max(traces)
    sig_range = sig_max - sig_min
    eye_opening_ratio = (eye_height / sig_range) if (sig_range > 0) else None
    metrics["eye_opening_ratio"] = float(eye_opening_ratio) if eye_opening_ratio is not None else None

    return time, traces, metrics

def costas_loop(x, fs, loop_bandwidth, order=2, damping=0.707):
    """Boucle de Costas : verrouillage de phase"""
    # Normalise amplitude & bw
    x = x/np.max(np.abs(x))
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
    
    return np.array(out)

def psk_demodulate(sig, fs, symbol_rate, order=2, gray=False, differential=False, offset=False, pi4=False, costas_damping=0.707, costas_bw_factor=0.01):
    """Démodulation BPSK/QPSK"""
    # Recup par boucle de Costas
    bw_loop = symbol_rate * costas_bw_factor # BW boucle sur critère de rapidité. Compromis actuel suppose RSB faible mais correct.
    recovered = costas_loop(sig, fs, loop_bandwidth=bw_loop, order=order, damping=costas_damping)

    # Timing symbole, sps fractionnel
    sps_exact = fs / symbol_rate
    n_symbols = int(round(len(recovered) / sps_exact))

    symbols = []
    for k in range(n_symbols):
        idx = int(round(k * sps_exact + sps_exact / 2)) # échantillon au centre du symbole
        if idx < len(recovered):
            symbols.append(recovered[idx])
    symbols = np.array(symbols)

    if order == 2 and differential == False:  # BPSK
        bits = (np.real(symbols) > 0).astype(int)
        
        return np.array(bits, dtype=np.uint8)

    elif order == 2 and differential == True: # DBPSK
        phases = np.angle(symbols * np.conj(np.roll(symbols, 1)))
        bits = (phases > 0).astype(np.uint8)[1:] 

        return np.array(bits, dtype=np.uint8)

    elif order == 4 and differential == False and offset == False:  # QPSK
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

        return np.array(bits, dtype=np.uint8)

    elif order == 4 and offset == True and differential == False: # OQPSK
        I = (np.real(symbols) > 0).astype(np.uint8)
        Q = (np.imag(symbols) > 0).astype(np.uint8)

        if not gray:
            Q = Q ^ I 
        # Décalage Q par demi symbole
        Q = np.roll(Q, -1)
        bits = np.empty(2*len(I), dtype=np.uint8)
        bits[0::2] = I
        bits[1::2] = Q

        return np.array(bits, dtype=np.uint8)

    elif order == 4 and offset == False and differential == True: # DQPSK
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

        return np.array(bits, dtype=np.uint8)

    elif order == 4 and offset and differential:  # DOQPSK
        # Décalage OQPSK d'abord
        I = (np.real(symbols) > 0).astype(np.uint8)
        Q = (np.imag(symbols) > 0).astype(np.uint8)
        Q = np.roll(Q, -1)  # Décalage demi symbole

        oqpsk_syms = I + 1j*Q  # recombiner

        # Puis décodage différentiel comme en DQPSK
        diffs = oqpsk_syms * np.conj(np.roll(oqpsk_syms, 1))
        phases = np.angle(diffs)

        bits = []
        phase_bits_gray = {0:(0,0), np.pi/2:(0,1), np.pi:(1,1), -np.pi/2:(1,0)}
        phase_bits_nat  = {0:(0,0), np.pi/2:(0,1), np.pi:(1,0), -np.pi/2:(1,1)}
        mapping = phase_bits_gray if gray else phase_bits_nat

        for ph in phases[1:]:
            ph = (ph + np.pi) % (2*np.pi) - np.pi
            if -np.pi/4 <= ph < np.pi/4:
                bits.extend(mapping[0])
            elif np.pi/4 <= ph < 3*np.pi/4:
                bits.extend(mapping[np.pi/2])
            elif -3*np.pi/4 <= ph < -np.pi/4:
                bits.extend(mapping[-np.pi/2])
            else:
                bits.extend(mapping[np.pi])

        return np.array(bits, dtype=np.uint8)
    
    elif order == 4 and pi4:  # π/4-QPSK
        # On gère le cas général qui est en fait π/4-DQPSK
        diffs = symbols * np.conj(np.roll(symbols, 1))
        phases = np.angle(diffs)

        bits = []
        # Table de mapping π/4 en gray
        phase_bits = {
            np.pi/4:  (0,0),
            3*np.pi/4:(0,1),
            -3*np.pi/4:(1,1),
            -np.pi/4:(1,0)
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

        return np.array(bits, dtype=np.uint8)

    else:
        raise ValueError("Unsupported combination of order/differential/offset/pi4.")

# Inutilisée pour l'instant ; peut servir pour affiner le timing CPM. A tester + trouver autre méthode pour PSK et ordre supérieur à 2.
def estimate_timing_phase(signal, sps, zero=0):
    """Estime le décalage de phase dans un signal NRZ via zero-crossings.
    Méthode basée sur un algorithme proposé par Eduardo Fuentetaja"""
    sum_cos = 0.0
    sum_sin = 0.0

    last_value = signal[0]
    last_gt_zero = last_value >= zero

    for i in range(1, len(signal)):
        value = signal[i]
        value_gt_zero = value >= zero # test de passage par zéro

        if value_gt_zero != last_gt_zero:
            # interp linéaire
            x = float(last_value) / float(last_value - value)
            crossing = (i - 1) + x

            # modulo sps
            crossing %= sps

            # normalise l'angle
            angle = crossing * 2.0 * np.pi / sps
            slope = abs(value - last_value) # pente au crossing
            sum_cos += np.cos(angle) * slope
            sum_sin += np.sin(angle) * slope

        last_value = value
        last_gt_zero = value_gt_zero

    # calcul de l'offset par moyenne vectorielle
    offset = np.arctan2(sum_sin, sum_cos)
    offset *= sps / (2.0 * np.pi) # conversion en échantillons
    return offset