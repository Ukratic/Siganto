"""Basé sur les fonctions de Michael Ossmann, "clock recovery experiments" : https://github.com/mossmann/clock-recovery
Avec quelques modifications et ajouts pour compléter les cas d'usage"""
import numpy as np
import scipy.signal as signal
import string

# Commentaires et code de Michael laissés bruts pour mieux repérer les modifications

# input: magnitude spectrum of clock signal (np array)
# output: FFT bin number of clock frequency
def find_clock_frequency(spectrum, sample_rate, target_rate=None, precision=0.9):
    """Détermine la fréquence d'horloge"""
    maxima = signal.argrelextrema(spectrum, np.greater_equal)[0]
    while maxima[0] < 2:
        maxima = maxima[1:]
    if not maxima.any():
        return 0

    if target_rate:
        min_rate = target_rate * precision
        max_rate = target_rate / precision
    else:
        min_rate, max_rate = None, None

    # Convert min/max symbol rate to FFT bin range
    nfft = len(spectrum)
    min_bin = int((min_rate / sample_rate) * nfft) if min_rate else 2
    max_bin = int((max_rate / sample_rate) * nfft) if max_rate else nfft // 2

    # Filter out peaks outside the expected range
    maxima = maxima[(maxima >= min_bin) & (maxima <= max_bin)]

    if maxima.size == 0:
        return 0  # No valid frequency found

    threshold = max(spectrum[2:-1]) * 0.8
    indices_above_threshold = np.argwhere(spectrum[maxima] > threshold)

    return maxima[indices_above_threshold[0]] if indices_above_threshold.size > 0 else 0

def midpoint(a):
    """Fonction pour déterminer le milieu de segment"""
    mean_a = np.mean(a)
    mean_a_greater = np.ma.masked_greater(a, mean_a)
    high = np.ma.median(mean_a_greater)
    mean_a_less_or_equal = np.ma.masked_array(a, ~mean_a_greater.mask)
    low = np.ma.median(mean_a_less_or_equal)
    return (high + low) / 2

# whole packet clock recovery
# input: real valued NRZ-like waveform (array, tuple, or list)
#        must have at least 2 samples per symbol
#        must have at least 2 symbol transitions
# output: list of symbols
def wpcr(a, sample_rate, target_rate, tau, precision, debug):
    """Fonction principale de la méthode de démodulation"""
    if len(a) < 4:
        return []
    b = (a > midpoint(a)) * 1.0
    d = np.diff(b)**2
    if len(np.argwhere(d > 0)) < 2:
        return []
    f = np.fft.fft(d, len(a))
    p = find_clock_frequency(abs(f),sample_rate,target_rate,precision)
    if p == 0:
        return []
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.5 + np.angle(f[p])/(tau)
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i])
        clock_phase += cycles_per_sample
    clock_frequency = p * sample_rate / len(f)
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
    # binarize
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
        print("estimated baud rate: %f symbols/s" % baud_rate)
    return baud_rate[0]

# convert soft symbols into bits (assuming binary symbols)
def slice_binary(symbols):
    """Convertit les symboles en bits 0 et 1"""
    symbols_average = np.average(symbols)
    bits = symbols >= symbols_average
    return np.array(bits, dtype=np.uint8)

def slice_4fsk(symbols,mapping):
    """Convertit les symboles en bits doubles (00 à 11)"""
    # chaque symbole = 2 bits
    if len(symbols) == 0:
        return np.array([], dtype=np.uint8)
    # 4 niveaux de décision sur les symboles. la fonction attend que les symboles soient déjà normalisés
    q25, q50, q75 = -0.50, 0.00, 0.50
    # 2 mappings, sinon custom
    mappings = {
        "natural": {
            0: (0, 0),  # Lowest freq
            1: (0, 1),
            2: (1, 0),
            3: (1, 1)   # Highest freq
        },
        "gray": {
            0: (0, 0),  # Lowest freq
            1: (0, 1),
            2: (1, 1),
            3: (1, 0)   # Highest freq
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

# Emprunté : à tester
# EXPERIMENTAL
def _parabolic_interpol(mag, k):
    # Parabolic peak interpolation around bin k (on linear power or magnitude)
    # Returns fractional bin offset delta in [-0.5, 0.5] roughly
    if k <= 0 or k >= len(mag)-1:
        return 0.0
    a = mag[k-1]
    b = mag[k]
    c = mag[k+1]
    denom = (a - 2*b + c)
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (a - c) / denom
    delta = np.clip(delta, -1.0, 1.0)

    return delta

def _cluster_freqs(freqs, weights=None, merge_hz=0.0):
    if len(freqs) == 0:
        return np.array([])
    order = np.argsort(freqs)
    freqs = np.asarray(freqs, float)[order]
    w = np.ones_like(freqs) if weights is None else np.asarray(weights, float)[order]

    clusters = []
    cf, cw = freqs[0], w[0]
    for f, ww in zip(freqs[1:], w[1:]):
        if abs(f - cf) <= merge_hz:
            # merge
            cf = (cf*cw + f*ww) / (cw + ww)
            cw += ww
        else:
            clusters.append((cf, cw))
            cf, cw = f, ww
    clusters.append((cf, cw))
    clusters.sort(key=lambda x: -x[1])  # by weight
    return np.array([c[0] for c in clusters])

def detect_and_track_mfsk_auto(
    iq, fs, baud,
    num_tones=None,              # keep top-N tones after clustering (None=all)
    peak_thresh_db=8,            # per-frame relative threshold
    peak_prominence=None,        # optional: e.g. 3 (dB) -> converted internally
    win_factor=1.0,              # window length ≈ win_factor * (fs/baud)
    hop_factor=0.25,             # hop ≈ hop_factor * window
    merge_bins=1.2,              # cluster width in bins for tone dedup
    switch_penalty=0.05          # Viterbi penalty (linear power units)
):
    """
    Auto-detect baseband MFSK tones (±freq) and track them over time.
    - Uses sub-bin (parabolic) peak interpolation for tone detection
    - Clusters near-duplicate frequencies
    - Tracks via matched complex oscillators (Goertzel-style) + Viterbi
    """

    # --- Analysis window & hop (integers) ---
    sym_len = fs / float(baud)              # fractional allowed, no strict alignment needed
    N = max(16, int(round(win_factor * sym_len)))
    H = max(1, int(round(hop_factor * N)))
    nfft = N  # keeping nfft=N (power-of-two not required)
    df = fs / nfft

    # Window & frequency grid
    n = np.arange(N)
    win = 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))   # Hann
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs))

    # --- Stage 1: Tone detection over full spectrum with sub-bin interpolation ---
    obs_freqs = []
    obs_w = []

    for start in range(0, len(iq)-N+1, H):
        blk = iq[start:start+N] * win
        spec = np.fft.fftshift(np.fft.fft(blk, nfft))
        mag = np.abs(spec)
        db = 20*np.log10(mag + 1e-12)
        mx = np.max(db)

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
            delta = _parabolic_interpol(mag, k)
            f_est = freqs[k] + delta * df
            obs_freqs.append(f_est)
            # Weight: use linear power as weight
            obs_w.append(mag[k]**2)

    # Cluster close frequencies so each tone appears once
    merge_hz = max(df * merge_bins, 0.5)  # at least ~0.5 Hz to avoid crazy fragmentation
    tones_all = _cluster_freqs(obs_freqs, weights=obs_w, merge_hz=merge_hz)
    if num_tones is not None and len(tones_all) > num_tones:
        tones_all = tones_all[:num_tones]
    tone_freqs = np.sort(tones_all)
    K = len(tone_freqs)
    if K == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.empty((0,0))

    # --- Stage 2: Tracking with matched oscillators (bin-free; exact Hz ok) ---
    # Energy-normalized window
    win_e = win / np.sqrt(np.sum(win**2) + 1e-12)
    # Oscillators per tone (K x N)
    osc = np.exp(-1j * 2*np.pi * (tone_freqs[:, None] / fs) * n) * win_e

    starts = np.arange(0, len(iq)-N+1, H)
    T = len(starts)
    tone_pows = np.empty((T, K), float)
    times = np.empty(T, float)

    def agc_block(x):
        r = np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
        return x / r

    for i, s in enumerate(starts):
        blk = agc_block(iq[s:s+N])
        c = (osc * blk).sum(axis=1)
        tone_pows[i] = np.abs(c)**2
        times[i] = (s + (N-1)/2) / fs

    # Viterbi-like smoothing with simple switch penalty
    dp = np.empty_like(tone_pows)
    back = np.empty_like(tone_pows, dtype=np.int32)
    dp[0] = tone_pows[0]
    back[0] = -1
    if K > 1:
        pen = np.ones((K, K)) * switch_penalty
        np.fill_diagonal(pen, 0.0)

    for i in range(1, T):
        if K == 1:
            dp[i] = dp[i-1] + tone_pows[i]
            back[i] = 0
        else:
            prev = dp[i-1][:, None]          # (K,1)
            scores = prev - pen              # (K,K): from j→k
            j_star = np.argmax(scores, axis=0)
            best_prev = scores[j_star, np.arange(K)]
            dp[i] = best_prev + tone_pows[i]
            back[i] = j_star

    # Backtrack
    kT = int(np.argmax(dp[-1]))
    tone_idx = np.empty(T, dtype=np.int32)
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
            runs.append(i - start)   # run length in frames
            start = i
    runs.append(len(tone_idx) - start)  # last run

    if len(runs) > 0:
        avg_run = np.median(runs)       # median = robust against outliers
        measured_rate = 1.0 / (avg_run * dt)
    else:
        measured_rate = 0.0

    return tone_freqs, times, tone_idx, tone_trace, tone_pows, measured_rate