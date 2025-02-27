# Fonctions DSP à ajouter si/quand nécessaire
import numpy as np

# Flattop window
def flattop_window(N):
    if N < 1:
        return np.array([])
    
    # Coefficients pour la fenêtre Flattop (source : implémentation Octave de Flattop Harris)
    a0 = 1.0
    a1 = 1.93
    a2 = 1.29
    a3 = 0.388
    a4 = 0.0322

    n = np.arange(0, N)
    term0 = a0
    term1 = -a1 * np.cos(2 * np.pi * n / (N - 1))
    term2 = a2 * np.cos(4 * np.pi * n / (N - 1))
    term3 = -a3 * np.cos(6 * np.pi * n / (N - 1))
    term4 = a4 * np.cos(8 * np.pi * n / (N - 1))

    return term0 + term1 + term2 + term3 + term4

# Fonctions pour centrer signal
def find_peaks_prominence(spectrum, proeminence=0.1): # imite scipy.signal.find_peaks
    peaks = []
    for i in range(1, len(spectrum) - 1):
        is_peak = True
        if spectrum[i] <= spectrum[i-1] or spectrum[i] <= spectrum[i+1]:
            is_peak = False

        if is_peak:
            for j in range(max(0, i-5), min(len(spectrum), i+5)):  # voisinage
                if i != j and spectrum[i] - spectrum[j] < proeminence * np.max(spectrum):
                    is_peak = False
                    break
        if is_peak:
            peaks.append(i)
    return np.array(peaks)

def center_signal(iq_wave, frame_rate, proeminence=0.1): 
    N = len(iq_wave)
    t = np.arange(N) / frame_rate

    window = np.hanning(N)
    windowed_iq = iq_wave * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed_iq))
    f = np.fft.fftfreq(N, 1/frame_rate)
    f = np.fft.fftshift(f)

    peaks_indices = find_peaks_prominence(np.abs(spectrum), proeminence)  # Get indices of peaks

    if len(peaks_indices) < 2:
        print("Moins de 2 pics trouvés pour recentrer le signal.")
        return None

    peak_freqs = f[peaks_indices]
    # Sort par mag
    sorted_indices = np.argsort(np.abs(spectrum[peaks_indices]))[::-1]
    top_2_indices = sorted_indices[:2]
    peak1_freq = peak_freqs[top_2_indices[0]]
    peak2_freq = peak_freqs[top_2_indices[1]]
    # calc & freq shift
    center_freq = (peak1_freq + peak2_freq) / 2
    iq_shifted = iq_wave * np.exp(-1j * 2 * np.pi * center_freq * t)

    return iq_shifted, center_freq