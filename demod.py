"""Basé sur les fonctions de Michael Ossmann, "clock recovery experiments" : https://github.com/mossmann/clock-recovery
Avec quelques modifications et ajouts pour compléter les cas d'usage"""
import numpy as np
import scipy.signal as signal

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
    clock_frequency = p * sample_rate // len(f)
    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("detected clock frequency: %d Hz" % (p * sample_rate // len(f)))
        print("samples per symbol: %f" % (1.0/cycles_per_sample))
        print("clock cycles per sample: %f" % (cycles_per_sample))
        print("clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase))
        print("clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2))
        print("symbol count: %d" % (len(symbols)))
    return symbols, clock_frequency

# convert soft symbols into bits (assuming binary symbols)
def slice_binary(symbols):
    """Convertit les symboles en bits 0 et 1"""
    symbols_average = np.average(symbols)
    bits = symbols >= symbols_average
    return np.array(bits, dtype=np.uint8)

def slice_4fsk(symbols,mapping="natural"):
    """Convertit les symboles en bits doubles (00 à 11)"""
    # chaque symbole = 2 bits
    if len(symbols) == 0:
        return np.array([], dtype=np.uint8)
    # 4 niveaux de décision sur percentiles
    q25, q50, q75 = np.percentile(symbols, [25, 50, 75])
    # 2 mappings (dictionnaire éventuellement à étendre)
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
    # Sélection du mapping
    if isinstance(mapping, list) and len(mapping) == 4:
        bit_map = {i: tuple(mapping[i]) for i in range(4)}
    else:
        bit_map = mappings.get(mapping, mappings["natural"])  # Default to natural
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
