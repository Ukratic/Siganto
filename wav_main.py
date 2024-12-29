import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io.wavfile as wav
from scipy import signal
import struct
import estimate_modulation as em

## Lecture du fichier wav
filepath = sys.argv[1]

# Trouver la largeur d'échantillon (encodage en nb de bits)
def find_sample_width(file_path):
    with open(file_path,'rb') as wav_file:
        header = wav_file.read(44) # Premiers 44 bytes = réservés header
        if header[:4] != b'RIFF' or header[8:12] != b'WAVE' or header[12:16] != b'fmt ':
            raise ValueError("Invalid WAV file")
        sample_width = struct.unpack('<H', header[34:36])[0]
    
    return sample_width

try:
    n_bits = find_sample_width(filepath)
except:
    n_bits = 32

try:
    N = int(sys.argv[2])
except:
    N = 2**9

if n_bits == 32:
    n_bits = np.int32
elif n_bits == 16:
    n_bits = np.int16
elif n_bits == 8:
    n_bits = np.int8
elif n_bits == 64:
    n_bits = np.int64
else:   
    n_bits = np.int32
    print("Valeur de n_bits incorrecte, n_bits = 32 par défaut")
    print("Valeurs possibles: 8, 16, 32, 64")
print("Lecture du fichier wav au format: ", n_bits)

# Taille de fenêtre de la FFT
if len(sys.argv) < 2:
    print("Utilisation: python wav_main.py <filepath> <N>")
    print("filepath: chemin du fichier wav")
    print("N: taille de la fenêtre (optionnel)")

    ans_window = input("Modifier la taille de la fenêtre (par défaut 512) ? (y/n): ")
    if ans_window == 'y':
        N = int(input("Entrer la taille de la fenêtre: "))
    else:
        print("Taille de fenêtre : ", N)
else:
    print("Taille de fenêtre : ", N)

## Main
frame_rate,s_wave = wav.read(filepath)
# Quelques infos sur le fichier
print("Fréquence d'échantillonnage: ", frame_rate)
print("Nombre d'échantillons: ", len(s_wave))
print(len(np.shape(s_wave)))
# Récupération du signal IQ complexe
if len(np.shape(s_wave)) == 2:
    print("Canaux: ", np.shape(s_wave)[1])  
    s_wave = np.frombuffer(s_wave, dtype=n_bits)
    left, right = s_wave[0::2], s_wave[1::2]
    iq_wave = left + 1j * right
else:
    left, right = s_wave[0::2], s_wave[1::2]
    iq_wave = left[:min(len(left), len(right))] + 1j * right[:min(len(left), len(right))]
    # # handle radioscanner.ru's poorly encoded wav files
    # left, right = s_wave[0::2], s_wave[1::2]
    # iq_wave = left[:min(len(left), len(right))] + 1j * right[:min(len(left), len(right))]
    # fcenter = frame_rate//2
    # iq_wave = iq_wave * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_wave))/frame_rate)

# Compute the PSD
freqs, dsp = em.compute_dsp(iq_wave, frame_rate, N)
f_pmax = freqs[np.argmax(dsp)]
f_pmin = freqs[np.argmin(dsp)]
print(f_pmax, f_pmin)
# Plot the PSD
plt.figure(figsize=(10, 6))
plt.plot(freqs / 1e6, 20*np.log10(dsp))  # Convert frequency to MHz for plotting
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Power [dB]')
plt.grid(True)
plt.show()

plt.figure()
plt.magnitude_spectrum(iq_wave, frame_rate)
plt.show()


print("...")  
# Graphes de base
plt.figure('Spectrogramme, DSP & constellation', figsize=(12, 8))
# Spectrogramme
plot_b = plt.subplot(2, 1, 1)
freqs, times, spectrogram = em.compute_spectrogram(iq_wave, frame_rate, N)
plt.imshow(spectrogram, aspect='auto', extent=[freqs[0]/1e6, freqs[-1]/1e6, times[-1], times[0]], cmap=cm.jet)
plot_b.set_xlabel("Frequence [MHz]")
plot_b.set_ylabel("Temps [s]")

# DSP 
plot_c = plt.subplot(3, 2, 5)
plot_c.psd(iq_wave, Fs=frame_rate, NFFT=N, noverlap=N//2)
plot_c.set_xlabel('Frequence [Hz]')
plot_c.set_ylabel('Puissance')
plot_c.set_xlim(-frame_rate/2, frame_rate/2)

# Constellation
plot_a = plt.subplot(3, 2, 6)
plot_a.scatter(np.real(iq_wave), np.imag(iq_wave))
plot_a.set_xlabel('In-Phase')
plot_a.set_ylabel('Quadrature')

plt.figure('STFT, 3d plot & Pmax', figsize=(12, 8))

# STFT
freqs, times, stft_matrix = em.compute_stft(iq_wave, frame_rate, window_size=N, overlap=N//2)
plot_d = plt.subplot(2, 2, 1)
plot_d.imshow(stft_matrix, aspect='auto', extent = [frame_rate/-2/1e6, frame_rate/2/1e6, len(iq_wave)/frame_rate, 0],cmap=cm.jet)
plot_d.set_xlabel("Frequence [MHz]")
plot_d.set_ylabel("Temps [s]")

# Spectrogramme 3D 
plot_e = plt.subplot(2, 2, 2, projection='3d')
freqs, times, spectrogram = em.compute_spectrogram(iq_wave, frame_rate, N)
X, Y = np.meshgrid(freqs, times)
plot_e.plot_surface(X, Y, spectrogram, cmap=cm.coolwarm)
plot_e.set_xlabel('Frequence [Hz]')
plot_e.set_ylabel('Temps [s]')
plot_e.set_zlabel('Puissance [dB]')

# DSP avec max
wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_wave)))**2
f = np.linspace(frame_rate/-2, frame_rate/2, len(iq_wave))/1e6 # plt in MHz
plot_f = plt.subplot(3, 1, 3)
plot_f.plot(f, wav_mag)
plot_f.plot(f[np.argmax(wav_mag)], np.max(wav_mag), 'rx') # show max
plot_f.grid()
plot_f.set_xlabel('Frequence [MHz]')
plot_f.set_ylabel('Puissance [dB]')

plt.show()
 
## Modifications, calage, filtrage
# Trouver la fréquence à la puissance maximale et minimale
freq, psd = signal.welch(iq_wave, fs=frame_rate, nperseg=N, noverlap=N//2,return_onesided=False)
f_pmax = freq[np.argmax(psd)]
print("Sachant que la fréquence centrale est 0 Hz,")
print("Fréquence à la puissance maximale: ", f_pmax)
print("Niveau le plus haut: ", 10*np.log10(np.max(np.abs(iq_wave)**2)))
f_pmin = freq[np.argmin(psd)]
print("Fréquence à la puissance minimale: ", f_pmin)
print("Niveau le plus bas: ", 10*np.log10(np.min(np.abs(iq_wave)**2)))

# Corriger la fréquence centrale si nécessaire
print("...")
print("Modifier / filtrer le signal (correction de fréquence, filtre passe-bande, etc.). Pour sauter, appuyer sur Entrée.")
print("...")
ans_fcenter = input("Entrer la correction de FC en Hz (négatif pour monter sur x scale, positif pour descendre) : ")
try:
    fcenter = float(ans_fcenter)
    iq_wave = iq_wave * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_wave))/frame_rate)
    plt.figure('Spectrogramme & DSP', figsize=(12, 8))
    plot_b = plt.subplot(211)
    plot_b.specgram(iq_wave, NFFT=N, Fs=frame_rate, noverlap=N//2, cmap=cm.jet)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequence')

    plot_c = plt.subplot(212)
    plot_c.psd(iq_wave, Fs=frame_rate, NFFT=N, noverlap=N//2)
    plot_c.set_xlabel('Frequence')
    plot_c.set_ylabel('Puissance')
    plt.show()
except:
    print("Skip")

# Filter signal : retirer la bande passante inutile si nécessaire
ans_bande = input("Filtrer la bande passante inutile ? (y/n): ")
if ans_bande == 'y':
    print("...")
    ans_cut = input("Choisir le type de filtre (bas, haut, bande) : ")
    if ans_cut == 'bas':
        cutoff = float(input("Fréquence de coupure en Hz: "))
        iq_wave = em.lowpass_filter(iq_wave, cutoff, frame_rate)
    elif ans_cut == 'haut':
        cutoff = float(input("Fréquence de coupure en Hz: "))
        iq_wave = em.highpass_filter(iq_wave, cutoff, frame_rate)
    elif ans_cut == 'bande':
        lowcut = float(input("Fréquence de coupure basse en Hz: "))
        highcut = float(input("Fréquence de coupure haute en Hz: "))
        iq_wave = em.bandpass_filter(iq_wave, lowcut, highcut, frame_rate)
    else:
        print("Type de filtre inconnu")

    plt.figure('Spectrogramme & DSP', figsize=(12, 8))
    plot_b = plt.subplot(211)
    plot_b.specgram(iq_wave, NFFT=N, Fs=frame_rate, noverlap=N//2, cmap=cm.jet)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequence')

    plot_c = plt.subplot(212)
    plot_c.psd(iq_wave, Fs=frame_rate, NFFT=N, noverlap=N//2)
    plot_c.set_xlabel('Frequence')
    plot_c.set_ylabel('Puissance')
    plt.show()
else:
    print("Skip")

# Retirer une partie du signal s'il est trop long
print("...")
print("Le signal est de ", len(iq_wave), "échantillons, soit ", len(iq_wave)/frame_rate, "s.")
ans_cut = input("Retirer une partie du signal (en secondes) ? (y/n): ")
if ans_cut == 'y':
    ans_cut_start = input("Spécifier combien de secondes retirer au début: ")
    iq_wave = iq_wave[int(float(ans_cut_start)*frame_rate):]
    print("Signal raccourci de ", float(ans_cut_start), "s, soit ", len(iq_wave), "échantillons restants sur un total de ", len(iq_wave)+int(float(ans_cut_start)*frame_rate), 
          ": ", round(len(iq_wave)/(len(iq_wave)+int(float(ans_cut_start)*frame_rate))*100,3), "%")
else:
    print("Skip")

# Downsample / Upsample
print("...")
ans_downsample = input("Réduire la fréquence d'échantillonnage du signal (décimation) ? (y/n): ")
if ans_downsample == 'y':
    ans_rate = input("Entrer le taux de décimation cible: ")
    decimation_factor = int(ans_rate)
    iq_wave, frame_rate = em.downsample(iq_wave, frame_rate, decimation_factor)
    print("Fréquence d'échantillonnage réduite à ", frame_rate, "Hz")
    plt.figure('Spectrogramme & DSP', figsize=(12, 8))
    plot_b = plt.subplot(211)
    plot_b.specgram(iq_wave, NFFT=N, Fs=frame_rate, noverlap=N//2, cmap=cm.jet)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequence')

    plot_c = plt.subplot(212)
    plot_c.psd(iq_wave, Fs=frame_rate, NFFT=N, noverlap=N//2)
    plot_c.set_xlabel('Frequence')
    plot_c.set_ylabel('Puissance')
    plt.show()
else:
    print("Skip")

ans_upsample = input("Augmenter la fréquence d'échantillonnage du signal (sur-échantillonnage) ? (y/n): ")
if ans_upsample == 'y':
    ans_rate = input("Entrer le facteur de sur-échantillonnage: ")
    oversampling_factor = int(ans_rate)
    iq_wave, frame_rate = em.upsample(iq_wave, frame_rate, oversampling_factor)
    print("Fréquence d'échantillonnage augmentée à ", frame_rate, "Hz")
    plt.figure('Spectrogramme & DSP', figsize=(12, 8))
    plot_b = plt.subplot(211)
    plot_b.specgram(iq_wave, NFFT=N, Fs=frame_rate, noverlap=N//2, cmap=cm.jet)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequence')

    plot_c = plt.subplot(212)
    plot_c.psd(iq_wave, Fs=frame_rate, NFFT=N, noverlap=N//2)
    plot_c.set_xlabel('Frequence')
    plot_c.set_ylabel('Puissance')
    plt.show()
else:
    print("Skip")

# Trouver le niveau moyen du signal en dB
print("...")
iq_floor = 10*np.log10(np.mean(np.abs(iq_wave)**2))
print("Niveau moyen en dB: ", iq_floor)
# Retirer signal sous point de coupure
ans_cut = input("Relever niveau plancher à la moyenne ? (y/n): ")
if ans_cut == 'y':
    iq_wave = np.where(10*np.log10(np.abs(iq_wave)**2) < iq_floor, 0, iq_wave)

    plt.figure('Spectrogramme & DSP', figsize=(12, 8))
    plot_b = plt.subplot(211)
    plot_b.specgram(iq_wave, NFFT=N, Fs=frame_rate, noverlap=N//2, cmap=cm.jet)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequence')

    plot_c = plt.subplot(212)
    plot_c.psd(iq_wave, Fs=frame_rate, NFFT=N, noverlap=N//2)
    plot_c.set_xlabel('Frequence')
    plot_c.set_ylabel('Puissance')
    plt.show()
else:
    print("Skip")

## Mesures de caractéristiques de modulation
print("...")
print("Mesures des caractéristiques de modulation. Pour sauter, appuyer sur Entrée.")
print("...")

# Rapidité de modulation
ans_pspectrum = input("Afficher spectre de puissance ? (y/n): ")
if ans_pspectrum == 'y':
    print("...")
    print("Méthode 1 : Power spectrum")
    clock, f, peak_freq = em.power_spectrum_fft(iq_wave, frame_rate)
    plt.figure('Power spectrum', figsize=(12, 8))
    plt.title(f"Pic de puissance à {round(peak_freq)} Hz")
    plt.plot(f,np.abs(clock))
    plt.xlabel('Frequence [Hz]')
    plt.ylabel('Puissance')
    plt.show()
else:
    print("Skip")

# Alternatif
ans_speed = input("Afficher spectre de puissance alternatif ? (y/n): ")
if ans_speed == 'y':
    print("...")
    print("Méthode 1bis : Power spectrum alternatif")
    clock, f, peak_freq = em.mean_threshold_spectrum(iq_wave, frame_rate)
    plt.figure('Power spectrum bis', figsize=(12, 8))
    plt.title(f"Pic de puissance estimé à {round(peak_freq)} Hz")
    plt.plot(f,np.abs(clock))
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('Puissance')
    plt.show()
else:
    print("Skip")

# signal IQ ^2 et ^4
ans_power = input("Afficher la DSP ^2 et ^4 ? (y/n): ")
if ans_power == 'y':
    print("...")
    print("Méthode 2 : DSP power")
    plt.figure('IQ^2 & IQ^4', figsize=(12, 8))
    # signal IQ ^2
    iq_mag = iq_wave **2
    plot_a = plt.subplot(211)
    plot_a.psd(iq_mag, Fs=frame_rate, NFFT=N, noverlap=N//2)
    plot_a.set_xlabel('Frequence [Hz]')
    plot_a.set_ylabel('Puissance [dB]')

    # signal IQ ^4
    iq_quart = iq_wave**4
    plot_b = plt.subplot(212)
    plot_b.psd(iq_quart, Fs=frame_rate, NFFT=N, noverlap=N//2)
    plot_b.set_xlabel('Frequence [Hz]')
    plot_b.set_ylabel('Puissance [dB]')
    print("La rapidité de modulation peut être mesurée en observant l'écart entre les pics de puissance.")
    plt.show()
else:
    print("Skip")

# Alternatif
ans_mag = input("Afficher la DSP alternative, ^2 et ^4 ? (y/n): ")
if ans_mag == 'y':
    print("...")
    print("Méthode 2bis : DSP power alternative")
    f, mag, squared, quartic = em.power_series(iq_wave, frame_rate)
    plt.figure('Magnitude metric, square metric, quadratic metric', figsize=(12, 8))
    plot_y = plt.subplot(3, 1, 1)
    plot_y.plot(f, mag)
    plot_y.set_ylabel('Magnitude Metric')

    plot_z = plt.subplot(3, 1, 2)
    plot_z.plot(f, squared)
    plot_z.set_ylabel('Squared Metric')

    plot_w = plt.subplot(3, 1, 3)
    plot_w.plot(f, quartic)
    plot_w.set_xlabel('Frequence [Hz]')
    plot_w.set_ylabel('Quartic Metric')

    plt.show()
else:
    print("Skip")

# FI
ans_persist = input("Afficher la DSP persistante ? (y/n): ")
if ans_persist == 'y':
    print("...")
    f, min_power, max_power, persistence = em.persistance_spectrum(iq_wave, frame_rate, N)

    plt.figure(figsize=(12, 8))
    plt.imshow(
        persistence.T,
        aspect='auto',
        extent=[f[0], f[-1], min_power, max_power],
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label="Persistence (normalized)")
    plt.title("Persistence Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.show()
else:
    print("Skip")


## Cyclostationnarité
# Fonction d'autocorrélation
ans_autocorr = input("Calculer la fonction d'autocorrélation ? (y/n): ")
if ans_autocorr == 'y':
    print("Calcul de la fonction d'autocorrélation")
    print("...")
    yx, lags = em.autocorrelation(iq_wave, frame_rate)
    plt.figure('Autocorrelation', figsize=(12, 8))
    plt.plot(lags*1e3, yx/np.max(yx)) # lags en ms
    plt.xlabel('Lag (ms)')
    plt.ylabel('Autocorrelation')
    plt.show()
else:
    print("Skip")

    # peak_value, time = em.autocorrelation_peak(iq_wave, frame_rate, min_distance=10)
    # print("Pic de la fonction d'autocorrélation trouvé : ", peak_value, "à ", time, "ms")

ans_cyclo = input("Calculer l'autocorrélation cyclique ? (y/n): ")
if ans_cyclo == 'y':
    print("Calcul de la fonction d'autocorrélation cyclique")
    print("...")
    # Params cyclospectre
    alpha_max=1e3
    tau_max=50e-2
    caf_mx, alpha, tau = em.cyclic_autocorrelation(iq_wave, frame_rate, alpha_max, tau_max)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(tau, alpha / 1e3, caf_mx, shading='auto', cmap='viridis')
    plt.colorbar(label="CAF Magnitude")
    plt.xlabel("Tau (ms)")
    plt.ylabel("Cyclic Frequency (kHz)")
    plt.title("Cyclospectrum")
    plt.show()
else:
    print("Skip")

ans_autocorr_full = input("Calculer la fonction d'autocorrélation complète ? (y/n): ")
if ans_autocorr_full == 'y':
    print("Calcul de la fonction d'autocorrélation")
    print("...")
    yx, lags = em.full_autocorrelation(iq_wave)
    plt.figure('Autocorrelation complète', figsize=(12, 8))
    plt.plot(lags/frame_rate*1e3, yx/np.max(yx))
    plt.xlabel('Lag (ms)')
    plt.ylabel('Autocorrelation')
    plt.show()
else:
    print("Autocorrélation complète non calculée")

    # estim_tu = em.estimate_ofdm_symbol_duration(iq_wave, frame_rate, 0.1)
    # print("Estimation de la durée du symbole OFDM : ",estim_tu)

## EXPERIMENTAL
# Fréquence d'horloge
ans_symbol = input("Calculer la fréquence d'horloge (EXPERIMENTAL) ? (y/n): ")
if ans_symbol == 'y':
    print("Calcul de la fréquence d'horloge, du nombre de symboles, leur durée")
    print("...")
    symbols = em.clock_symbol(iq_wave)
    plt.figure('Echantillons / symboles', figsize=(12, 8))
    f = np.linspace(0, len(iq_wave)/frame_rate, len(symbols))
    plt.plot(f, symbols)
    plt.xlabel('Temps (s)')
    plt.ylabel('Packet Clock Recovery')
    plt.show()
else:
    print("Skip")


# Treillis de phase
ans_transitions = input("Afficher le treillis de phase ? (y/n): ")
if ans_transitions == 'y':
    print("...")
    modulation_type = input("Entrer le type de modulation (BPSK, QPSK, DQPSK, 8-PSK) : ")
    speed = float(input("Entrer la vitesse de modulation : "))
    num_transitions = int(input("Entrer le nombre de transitions à afficher : "))
    mapped_states, num_steps, y_positions, states = em.treillis_phase(iq_wave,frame_rate, speed, modulation_type, num_transitions)
    plt.figure(figsize=(12, 6))
    for t in range(num_steps):
        current_state = mapped_states[t]
        next_state = mapped_states[t + 1]
        plt.plot([t, t + 1], [y_positions[current_state], y_positions[next_state]], linewidth=2)
        plt.plot(t, y_positions[current_state], 'o')
        plt.title(f"Treillis de phase {modulation_type} ({num_transitions} premières transitions)")
    plt.xlabel("Time")
    plt.yticks(list(y_positions.values()), labels=states)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()
else:
    print("Skip")

# Symboles DQPSK
ans_dqpsk = input("Extraire les symboles d'un signal DQPSK (EXPERIMENTAL) ? (y/n): ")
if ans_dqpsk == 'y':
    dqpsk_symbols = em.dqpsk_symbol_extraction(iq_wave)
    bits = em.slice_bits(dqpsk_symbols)
    print(bits)
    # implémenter récupération des bits
else:
    print("Skip")

# Démodulation
ans_demod = input("Démoduler le signal (EXPERIMENTAL) ? (y/n): ")
if ans_demod == 'y':
    ans_mod = input("Choisir le type de modulation (BPSK, QPSK, DQPSK, QAM): ")
    if ans_mod == 'BSPK':
        print("Démodulation BPSK")
        symbols = em.extract_symbols_bpsk(iq_wave)
        bits = em.demodulate_bpsk(symbols)
        print(bits)
    elif ans_mod == 'QPSK':
        print("Démodulation QPSK")
        symbols = em.extract_symbols_qpsk(iq_wave)
        bits = em.demodulate_qpsk(symbols)
        print(bits)
    elif ans_mod == 'DQPSK':
        print("Démodulation DQPSK")
        symbols = em.extract_symbols_dqpsk(iq_wave)
        bits = em.demodulate_dqpsk(symbols)
        print(bits)
    elif ans_mod == 'QAM':
        print("Démodulation QAM")
        symbols = em.extract_symbols_qam(iq_wave)
        bits = em.demodulate_qam(symbols)
        print(bits)
    else:
        print("Type de modulation inconnu")

    ans_save = input("Sauvegarder les bits extraits ? (y/n): ")
    if ans_save == 'y':
        filename = filepath.split('\\')[-1]
        with open(f"demodulated_bits_{filename}.txt", 'w') as f:
            f.write(bits)
        print(f"Bits extraits enregistrés sous demodulated_bits_{filename}.txt")
else:
    print("Skip")

## OFDM
# Estimation des paramètres OFDM
ans_ofdm = input("Estimer les paramètres OFDM ? (y/n): ")
if ans_ofdm == 'y':
    print("...")
else:
    sys.exit(0)

# Input = Tu (à partir de la fonction d'autocorrélation)
print("...")
estimated_ofdm_symbol_duration = float(input("Entrer durée Tu estimée en ms: "))
print("Durée estimée de Tu: ", estimated_ofdm_symbol_duration, "ms")
print("Plot de la fonction d'autocorrélation pour une durée de symbole OFDM donnée")
alpha_peak, alpha, caf = em.estimate_alpha(iq_wave, frame_rate, estimated_ofdm_symbol_duration)
plt.figure("Fonction d'autocorrélation pour une durée Tu donnée", figsize=(12, 8))
plt.plot(alpha*1e-3, 10*np.log10(caf/np.max(caf)))
plt.xlabel('Frequence (kHz)')
plt.ylabel('Puissance (dB)')
print("...")
plt.show()

# Input = alpha_0
print("...")
try:
    print("Calcul des paramètres OFDM")
    print("...")
    alpha_0 = float(input("Entrer alpha0 estimée en Hz: "))
    print("Alpha0 : ", alpha_0, "Hz")
    bw, fmax, fmin, f, Pxx = em.estimate_bandwidth(iq_wave, frame_rate, N)
    print("BW estimée", bw)
    Tu, Tg, Ts, Df, numb = em.calc_ofdm(alpha_0, estimated_ofdm_symbol_duration,bw)
    print("Tu (Durée symbole OFDM) = ", Tu, "ms")
    print("Tg (Préfixe cyclique) = ", Tg, "ms")
    print("Ts (Tu+Tg) = ", Ts, "ms")
    print("Δf = ", Df, "Hz")
    print("Nombre de sous-porteuses = ", numb)

    ans = input("Sauvegarder les résultats ? (y/n): ")
    if ans == 'y':
        filename = filepath.split('\\')[-1]
        with open(f"estimation_results_{filename}.txt", 'w') as f:
            f.write("Tu = " + str(Tu) + " ms\n")
            f.write("Tg = " + str(Tg) + " ms\n")
            f.write("Ts = " + str(Ts) + " ms\n")
            f.write("Delta F = " + str(Df) + " Hz\n")
        print(f"Resultats enregistrés sous estimation_results_{filename}.txt")
    else:
        print("Sauvegarde annulée.")
except:
    print("Skip")

ans_cyclic_ofdm = input("Calculer la fonction d'autocorrélation cyclique 3D ? (y/n) :")
if ans_cyclic_ofdm =='y':
    # Paramètres de la fonction d'autocorrélation cyclique
    tau_max = 2  # Max delay (ms)
    alpha_max = 50  # Max frequency (kHz)
    Ncaf = len(iq_wave)
    iTauMax = min(2000, round(tau_max * frame_rate))

    alpha = (np.arange(-1/2, 1/2, 1/Ncaf)) * frame_rate
    idxsa = np.min(np.where(alpha > -alpha_max))
    idxea = np.max(np.where(alpha < alpha_max))
    caf_mx = np.zeros((idxea - idxsa + 1, iTauMax), dtype=complex)

    for iTau in range(1, iTauMax + 1):
        # Calculer autocorrelation cyclique pour chaque délai tau
        caf_tmp = np.fft.fftshift(np.fft.fft(iq_wave[:Ncaf - iTau + 1] * np.conj(iq_wave[iTau - 1:]), Ncaf))
        caf_mx[:, iTau - 1] = caf_tmp[idxsa:idxea + 1]

    # Fonction d'autocorrélation cyclique 3D
    tau = np.arange(0, iTauMax) / frame_rate * 1e3
    alpha_s = alpha[idxsa:idxea + 1] * 1e-3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T, A = np.meshgrid(tau, alpha_s)
    surface = ax.plot_surface(T, A, np.abs(caf_mx), cmap='viridis', edgecolor='none')

    ax.set_xlim(0, np.max(tau))
    ax.set_ylim(-alpha_max / 1e3, alpha_max / 1e3)
    ax.set_zlim(0, np.max(np.abs(caf_mx)) / 4)
    ax.set_xlabel('tau (ms)')
    ax.set_ylabel('alpha (kHz)')
    ax.set_zlabel('|CAF|')
    plt.colorbar(surface, ax=ax, shrink=0.5)

    ax.view_init(elev=45, azim=45)
    plt.show()
else:
    print("Skip")

print("Fin.")