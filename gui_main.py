"""
SigAnTo - Signal Analysis Toolbox - GUI
Author: Arnaud Barraquand (@Ukratic)
Date: 2024-11-21

Description: GUI for signal analysis using Python. 3 libraries are used: tkinter, matplotlib, and scipy.
The GUI allows the user to load a WAV file, display its spectrogram, DSP, and other graphs.
Also, some signal modification functions : apply filters, cut the signal, move center frequency.
And some estimations, mostly through graphs : Modulation, symbol rate, bandwidth, OFDM parameters.

Many thanks to Dr Marc Lichtman - University of Maryland. Author of PySDR.
"""
import tkinter as tk
from threading import Thread, Event

# Activation option audio
with_sound = True
# Messages de debug
debug = True
# Drag & drop
drag_drop = True
if drag_drop is True:
    from tkinterdnd2 import TkinterDnD, DND_FILES

# Initialisation : chargement des librairies et de la langue dans une fonction pour pouvoir l'intégrer dans un thread
# Ce thread permet de charger les librairies en arrière-plan avec une fenêtre de chargement, en attendant que tout soit prêt
loading_end = Event()
def loading_libs():
    # Librairies
    print("Chargement des dépendances...")
    global struct, gc, FigureCanvasTkAgg, NavigationToolbar2Tk, plt, cm, np, wav, ll, em, lang, mg, sm, df, dm, scrolledtext, ttk, sd, string
    import struct
    import gc
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    import scipy.io.wavfile as wav
    from tkinter import scrolledtext
    import language_lib as ll
    import estimate_modulation as em
    import main_graphs as mg
    import signal_modification as sm
    import dsp_funcs as df
    import demod as dm
    import string

    if with_sound:
        from tkinter import ttk
        import sounddevice as sd
    # Langue, noms des fonctions, etc.
    lang = ll.get_fra_lib()
    # loading_end.wait() à la fin du script
    loading_end.set()
    
# Fenêtre de chargement
t = Thread(target=loading_libs, daemon=True)
t.start()
if drag_drop is True:
    root = TkinterDnD.Tk()
else:
    root = tk.Tk()
root.withdraw()
loading_screen = tk.Toplevel(root)
loading_screen.title("SigAnTo")
loading_screen.geometry("250x100")
loading_label = tk.Label(loading_screen, text="Loading...")
loading_label.pack()

while t.is_alive():
    root.update()

# Init fenêtre principale
print("Chargement de l'application...")
# Fenêtre principale
root.title(lang["siganto"])
root.geometry("1024x768")
menu_bar = tk.Menu(root)
print("Chargement des fonctions...")
# Init des variables
filepath = None
s_rate = None
iq_sig = None
bw = None
mono_real = True
convert_button = False
N = 512 # taille de la fenêtre FFT par défaut
overlap_value = 4 # valeur de recouvrement par défaut
overlap = N//overlap_value # recouvrement de la STFT
toolbar = None # toolbar matplotlib
diff_window = 0 # fenêtre de lissage des transitions
hist_bins = 1000 # bins pour les histogrammes
persistance_bins = 50 # bins pour le spectre de persistance
window_choice = "hamming" # fenêtre par défaut pour STFT
# Variables pour curseurs, lignes, et on/off
cursor_points = []
cursor_lines = []
distance_text = None # distance entre les curseurs
cursor_mode = False  # on/off
click_event_id = None
peak_indices = []
# Audio
is_playing = False
is_paused = False
audio_thread = None
audio_stream = None
stream_position = 0
# Autres params
tau_modifier = 2 # multiplicateur de tau pour WPCR
tau = np.pi * tau_modifier # constante tau pour WPCR
precision = 0.9 # précision par défaut de recherche de rapidité de modulation
filter_order = 4 # ordre des filtres par défaut (Butterworth)
peak_prominence = 0.1 # proéminence des pics pour le centrage du signal
morlet_fc = 6.0  # param de fréquence centrale par défaut pour ondelette de Morlet
morlet_nfreq = 96  # nombre de fréquences par défaut pour CWT Morlet
acf_min_distance = 25  # distance min par défaut pour ACF
scf_alpha_step = 10  # pas alpha par défaut pour SCF
costas_damping = 0.707  # facteur d'amortissement par défaut pour Costas
costas_bw_factor = 0.01  # facteur de boucle bande passante par défaut pour Costas
eye_num_traces = 500  # nombre de traces par défaut pour diagramme de l'oeil
eye_channel = 'I'  # canal par défaut pour diagramme de l'oeil
eye_symbols = 2 # nombre de symboles par défaut pour diagramme de l'oeil
mfsk_tresh_db = 8  # seuil en dB par défaut pour MFSK
mfsk_peak_prom_db = None  # proéminence des pics en dB par défaut pour MFSK
mfsk_win_factor = 1.0  # facteur de taille de fenêtre par défaut pour MFSK
mfsk_hop_factor = 0.25  # facteur de saut par défaut pour MFSK
mfsk_bin_width_cluster_factor = 1.2 # facteur de largeur de bin par défaut pour MFSK
mfsk_viterbi_penalty = 0.05  # pénalité Viterbi par défaut pour MFSK

# Frame pour les graphes
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Fonctions de chargement de fichier WAV
def find_sample_width(file_path):
    # Fonc de détermination d'encodage WAV
    with open(file_path,'rb') as wav_file:
        header = wav_file.read(44) # Premiers 44 bytes = réservés header
        if header[:4] != b'RIFF' or header[8:12] != b'WAVE' or header[12:16] != b'fmt ': # vérifie si c'est un fichier WAV
            tk.messagebox.showerror(lang["error"], lang["invalid_wav"], parent=root)
            raise ValueError(lang["invalid_wav"])
        bits_per_sample  = struct.unpack('<H', header[34:36])[0]
        audio_format = struct.unpack('<H', header[20:22])[0]
        if bits_per_sample not in [8, 16, 24, 32, 64]:
            tk.messagebox.showerror(lang["error"], lang["unsupported_bits"], parent=root)
            raise ValueError(lang["unsupported_bits"])
        if audio_format not in [1, 3]:  # PCM ou IEEE float
            tk.messagebox.showerror(lang["error"], lang["unsupported_format"], parent=root)
            raise ValueError(lang["unsupported_format"])
        audio_format = "PCM" if audio_format == 1 else "IEEE float"
    return bits_per_sample , audio_format

def load_wav():
    global filepath, s_rate, iq_sig, N, overlap, corr, convert_button
    if filepath is None:
        filepath = tk.filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not filepath:
        return

    s_rate, s_wave = wav.read(filepath)

    try:
        if s_wave.ndim == 2 and s_wave.shape[1] == 2:
            # Fichier stéréo → on suppose I/Q sur 2 canaux
            left = s_wave[:, 0].astype(np.float32)
            right = s_wave[:, 1].astype(np.float32)
            # retire quelques échantillons pour éviter les erreurs si nécessaire
            min_len = min(len(left), len(right))
            iq_sig = left[:min_len] + 1j * right[:min_len]
            # Vérification de corrélation entre les canaux gauche et droit
            corr = np.corrcoef(left, right)[0,1]
            if abs(corr) > 0.99:
                if debug is True:
                    print("Attention: canaux droite & gauche identiques. On force en mono.")
                mono_signal = left
                analytic_signal = sm.hilbert(mono_signal)
                iq_sig = analytic_signal
            else:
                iq_sig = left + 1j*right
        elif s_wave.ndim == 1:
            # Mono → 2 hypothèses
            if not convert_button:
                load_mono_button.pack(side=tk.LEFT, padx=5)
                convert_button = True
            if mono_real is True:
                iq_sig = sm.hilbert(s_wave)
                iq_sig = iq_sig * np.exp(-1j*2*np.pi*(s_rate//4)*np.arange(len(iq_sig))/s_rate)
                iq_sig, s_rate = sm.downsample(iq_sig, s_rate, 2)
            else:
                left = s_wave[0::2].astype(np.float32)
                right = s_wave[1::2].astype(np.float32)
                min_len = min(len(left), len(right))
                iq_sig = left[:min_len] + 1j * right[:min_len]

        else:
            raise ValueError("Format WAV inattendu")
    except:
        if debug is True:
            print("Erreur de conversion IQ")
        tk.messagebox.showerror(lang["error"], lang["wav_conversion"], parent=root)
    if len(iq_sig) > 1e6: # si plus d'un million d'échantillons
        N = 4096
    elif 1e5 < len(iq_sig) < 1e6 :
        N = 1024
    elif len(iq_sig) < s_rate: # si moins d'une seconde
        N = (s_rate//25)*(len(iq_sig)/s_rate) # base de résolution = 25 Hz par défaut, proportionnellement à la durée si inférieur à 1 seconde
        N = (int(N/2))*2 # N pair de préférence
        if N < 4: # taille minimum de 4 échantillons
            N = 4
    else:
        N = 512 # taille de fenêtre FFT par défaut
    overlap = N//overlap_value
        
    display_file_info()
    # Plot graphes initiaux après chargement du fichier
    plot_initial_graphs()

def on_file_drop(event):
    global filepath
    files = root.tk.splitlist(event.data)  # handles spaces in file names
    if files:
        filepath = files[0]  # take first dropped file
        load_wav()    

def load_real():
    global mono_real
    if mono_real is True:
        mono_real = False
        load_mono_button.config(text=lang["mono_iq"])
    else:
        mono_real = True
        load_mono_button.config(text=lang["mono_real"])

    load_wav()

# fonc de nettoyage graphe
def clear_plot():
    try:
        if cursor_mode:
            clear_cursors()
            toggle_cursor_mode()
        for widget in plot_frame.winfo_children():
            widget.destroy() # détruit les widgets tkinter
        plt.cla()
        plt.clf()
        plt.close('all') # ferme les figures matplotlib
        if toolbar:
            toolbar.destroy() # détruit la toolbar matplotlib
    except:
        pass
        if debug is True:
            print("Erreur de nettoyage du graphe")

# fonc fermeture du fichier, nettoie tout
def close_wav():
    global filepath, iq_sig
    filepath = None
    iq_sig = None
    clear_plot()
    # nettoie la mémoire
    gc.collect()
    if debug is True:
        print("Fermeture du fichier")
        print("Mémoire nettoyée:" , gc.get_stats())
    display_file_info()

# Fonction des graphes de base
def plot_initial_graphs():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text, bw
    # Spectrogramme et DSP. 1er sur 2 lignes, 2eme sur 1 ligne
    clear_plot()
    fig= plt.figure()
    spec = fig.add_gridspec(3, 2)
    fig.suptitle(lang["spec_dsp"])
    fig.tight_layout()
    a0 = fig.add_subplot(spec[0:2, :])
    a1 = fig.add_subplot(spec[2, :])
    ax = (a0, a1)

    # Spectrogramme 
    print(lang["spec_dsp"])
    if not filepath:
        print(lang["no_file"])
        return
    freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice)
    if freqs is None:
        print(lang["error_stft"])
        # message d'erreur si la STFT n'a pas pu être calculée
        tk.messagebox.showerror(lang["error"], lang["error_stft"], parent=root)
        return
    ax[0].imshow(stft_matrix, aspect='auto', extent = [s_rate/-2, s_rate/2, len(iq_sig)/s_rate, 0], cmap='jet')
    ax[0].set_ylabel(f"{lang['time_xy']} [s]")
    ax[0].set_title(f"{lang['window']} {window_choice}")

    # DSP
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_sig, s_rate, N, overlap, window_choice)
    Pxx_shifted = np.fft.fftshift(Pxx) 
    f_centered = np.linspace(-s_rate/2, s_rate/2, len(f)) # même échelle x que le spectrogramme
    ax[1].plot(f_centered, Pxx_shifted)
    ax[1].set_xlim(-s_rate/2, s_rate/2)
    ax[1].set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax[1].set_ylabel(lang["power_scale"])
    ax[1].axvline(x=fmax, color='r', linestyle='--')
    ax[1].axvline(x=fmin, color='r', linestyle='--')
    if debug is True:
        print("Bande passante estimée: ", bw, " Hz")
        print("Fréquence max BW: ", fmax, " Hz")
        print("Fréquence min BW: ", fmin, " Hz")
    
    ax[0].sharex(ax[1]) # zoom des 2 graphes en même temps

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    # del var pour libérer la mémoire
    del stft_matrix, canvas, spec, a0, a1, freqs, times, fmin, fmax, f, Pxx

# Fonc pour changer la taille de fenêtre FFT
def define_N():
    global N, overlap_value, overlap
    N = int(tk.simpledialog.askstring("N", lang["define_n"], parent=root))
    if N is None:
        if debug is True:
            print("Taille de fenêtre FFT non définie")
        return
    N = (int(N/2))*2 # N doit être pair
    overlap = N//overlap_value
    print(lang["fft_window"], N)
    plot_initial_graphs()
    display_file_info()

# # Autres groupe de graphes de base et flèches pour ajuster la fréquence centrale
def plot_other_graphs():
    global toolbar, ax, fig, canvas, iq_sig, original_iq_sig, fcenter, freq_label
    # Figure avec 3 sous-graphes. Le premier est sur deux lignes, les deux autres se partagent la 3eme ligne
    original_iq_sig = iq_sig.copy() # copie du signal original pour les modifications
    fcenter = 0  # Init de l'offset FC
    clear_plot()
    fig = plt.figure()
    spec = fig.add_gridspec(3, 2)
    fig.suptitle(lang["spec_const"])
    fig.tight_layout()
    a0 = fig.add_subplot(spec[0:2, :])
    a1 = fig.add_subplot(spec[2, 0])
    a2 = fig.add_subplot(spec[2, 1])
    ax = (a0, a1, a2)
    if not filepath:
        print(lang["no_file"])
        return
    # STFT
    freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice)
    stft = ax[0].imshow(stft_matrix, aspect='auto', extent=[s_rate / -2, s_rate / 2, len(iq_sig) / s_rate, 0], cmap=cm.jet)
    ax[0].set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax[0].set_ylabel(f"{lang['time_xy']} [s]")
    ax[0].set_title(f"{lang['window']} {window_choice}")

    # Constellation
    iq_constel = iq_sig/np.max(np.abs(iq_sig))
    line_constellation = ax[1].scatter(np.real(iq_constel), np.imag(iq_constel), s=1)
    ax[1].set_xlabel("In-Phase")
    ax[1].set_ylabel("Quadrature")

    # DSP avec max
    wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_sig)))**2
    wav_mag = wav_mag / np.max(wav_mag)
    f = np.linspace(s_rate / -2, s_rate / 2, len(iq_sig)) # freq en Hz
    line_spectrum, = ax[2].plot(f, wav_mag)
    ax[2].plot(f[np.argmax(wav_mag)], np.max(wav_mag), 'rx') # point max
    ax[2].grid()
    ax[2].set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax[2].set_ylabel(lang["norm_power"])

    # Label pour afficher l'offset FC
    freq_label = tk.Label(plot_frame, text=f"{lang['offset_freq']}: {fcenter} Hz")
    freq_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_graph():
        global iq_sig, fcenter
        # Recompute iq_sig avec l'offset de fréquence à partir de l'iq_sig original
        iq_sig = original_iq_sig * np.exp(-1j * 2 * np.pi * fcenter * np.arange(len(original_iq_sig)) / s_rate)
        # Màj label
        freq_label.config(text=f"{lang['offset_freq']}: {fcenter} Hz")
        # Màj STFT
        freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice)
        stft.set_data(stft_matrix)
        # Màj constellation
        iq_constel = iq_sig/np.max(np.abs(iq_sig))
        line_constellation.set_offsets(np.c_[np.real(iq_constel), np.imag(iq_constel)])
        # Màj DSP
        wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_sig)))**2
        wav_mag = wav_mag / np.max(wav_mag)
        line_spectrum.set_ydata(wav_mag)

        canvas.draw()

    def change_freq(step, event=None):
        global fcenter
        fcenter += step
        update_graph()

    # Flèches du clavier pour déplacer la fréquence centrale grossièrement
    root.bind("<Left>", lambda e: change_freq(-1, e))
    root.bind("<Right>", lambda e: change_freq(1, e))
    # Sinon boutons sur l'interface pour déplacer finement
    button_frame = tk.Frame(plot_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)
    left_button = tk.Button(button_frame, text="←", command=lambda: change_freq(-0.01))
    left_button.pack(side=tk.LEFT, padx=10)
    right_button = tk.Button(button_frame, text="→", command=lambda: change_freq(0.01))
    right_button.pack(side=tk.RIGHT, padx=10)
    
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del spec, a0, a1, a2, f, stft_matrix, freqs, times, wav_mag, iq_constel

# Fonc de changement de fenêtre FFT pour STFT
def set_window():
    global window_choice
    #dropdown pour choisir la fenêtre
    popup = tk.Toplevel()
    place_relative(popup, root, 300, 300)
    popup.title(lang["window_choice"])
    window_list = tk.StringVar()
    window_list.set(window_choice)
    tk.Radiobutton(popup, text="Hann", variable=window_list, value="hann").pack()
    tk.Radiobutton(popup, text="Kaiser", variable=window_list, value="kaiser").pack()
    tk.Radiobutton(popup, text="Hamming", variable=window_list, value="hamming").pack()
    tk.Radiobutton(popup, text="Blackman", variable=window_list, value="blackman").pack()
    tk.Radiobutton(popup, text="Bartlett", variable=window_list, value="bartlett").pack()
    tk.Radiobutton(popup, text="Flat Top", variable=window_list, value="flattop").pack()
    tk.Radiobutton(popup, text="Blackman-Harris 7-term", variable=window_list, value="blackmanharris7term").pack()
    tk.Radiobutton(popup, text="Rectangular", variable=window_list, value="rectangular").pack()
    tk.Radiobutton(popup, text="Gaussian", variable=window_list, value="gaussian").pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    window_choice = window_list.get()

    if window_choice is None or window_choice == "":
        window_choice = "rectangular" # Valeur par défaut
        if debug is True:
            print("Pas de fenêtre définie pour la STFT")
        return
    print("Fenêtre définie pour la STFT: ", window_choice)
    plot_initial_graphs()

# Fonc du spectrogramme 3D
def plot_3d_spectrogram():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # 3D
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["spec_3d"])
    print(lang["spec_3d"])
    if not filepath:
        print(lang["no_file"])
        return
    # Génération du spectrogramme 3D 
    freqs, times, spectrogram = mg.compute_spectrogram(iq_sig, s_rate, N, window_func=window_choice)
    spectrogram = spectrogram / np.max(spectrogram)  # normalisation
    X, Y = np.meshgrid(freqs, times)
    ax = plt.subplot(projection='3d')
    ax.plot_surface(X, Y, spectrogram, cmap=cm.coolwarm)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(f"{lang['time_xy']} [s]")
    ax.set_zlabel(lang["norm_amplitude"])
    ax.set_title(f"{lang['window']} {window_choice}")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    # del var pour libérer la mémoire
    del freqs, times, spectrogram, canvas

def time_amplitude():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # Amplitude en fonction du temps
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["time_amp"])
    print(lang["time_amp"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    time = np.arange(len(iq_sig)) / s_rate
    iq_norm = iq_sig / np.max(np.abs(iq_sig))
    ax.plot(time, iq_norm)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel(lang["norm_amplitude"])
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas, time, iq_norm

def spectre_persistance():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # Spectre de persistance : carte de chaleur de la persistance sur le spectre
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["persist_spectrum"])
    print(lang["persist_spectrum"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    f, min_power, max_power, persistence = em.persistance_spectrum(iq_sig, s_rate, N, persistance_bins, window_choice, overlap)
    ax.imshow(persistence.T, aspect='auto', extent=[f[0], f[-1], 0, 1], origin='lower', cmap='jet')
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas, f, min_power, max_power, persistence

# paramètres pour le spectre de persistance : nombre de bins
def param_spectre_persistance():
    global persistance_bins
    persistance_bins = tk.simpledialog.askstring(lang["params"], lang["pers_bins"], parent=root)
    if persistance_bins is None:
        persistance_bins = 50
        if debug is True:
            print("Pas de nombre de bins pour le spectre de persistance défini")
        return
    persistance_bins = int(persistance_bins)
    if debug is True:
        print("Nombre de bins pour le spectre de persistance défini à ", persistance_bins)
    spectre_persistance()

# params pour les fonctions de transitions de phase et de fréquence : lissage optionnel
def set_diff_params():
    global diff_window
    diff_window = tk.simpledialog.askstring(lang["params"], lang["smoothing_val"], parent=root)
    if diff_window is None:
        diff_window = 0
        if debug is True:
            print("Pas de fenêtre de lissage des transitions définie")
        return
    diff_window = int(diff_window)
    if debug is True:
        print("Fenêtre de lissage des transitions définie à ", diff_window)
    phase_difference()

# params pour les fonctions de transitions de phase et de fréquence : lissage optionnel
def set_hist_bins():
    global hist_bins
    hist_bins = tk.simpledialog.askstring(lang["params"], lang["hist_bins"], parent=root)
    if hist_bins is None:
        hist_bins = 1000
        if debug is True:
            print("Pas de nombre de bins défini pour les distributions")
        return
    hist_bins = int(hist_bins)
    if debug is True:
        print("Nombre de bins pour les histogrammes défini à ", hist_bins)
    frequency_cumulative()

def set_overlap():
    global overlap, overlap_value
    enter_overlap = tk.simpledialog.askstring(lang["params"], lang["overlap_val"], parent=root)
    if enter_overlap is None or enter_overlap == "" or int(enter_overlap) < 2:
        tk.messagebox.showinfo(lang["error"], lang["overlap_valid"], parent=root)
        overlap = N//overlap_value
        if debug is True:
            print("Pas de recouvrement valable défini pour la STFT. Recouvrement par défaut à ", overlap)
        return
    overlap_value = int(enter_overlap)
    overlap = N//overlap_value
    if debug is True:
        print("Recouvrement défini à ", overlap)
    display_file_info()
    plot_initial_graphs()

def stft_solo():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # STFT seul (hors groupe)
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["spectrogram"])
    print(lang["spectrogram"])
    if not filepath:
        print(lang["no_file"])
        return

    ax = plt.subplot()
    freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice)
    if freqs is None:
        print(lang["error_stft"])
        # message d'erreur si la STFT n'a pas pu être calculée
        tk.messagebox.showerror(lang["error"], lang["error_stft"], parent=root)
        return
    ax.imshow(stft_matrix, aspect='auto', extent = [s_rate/-2, s_rate/2, len(iq_sig)/s_rate, 0],cmap=cm.jet)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(f"{lang['time_xy']} [s]")
    ax.set_title(f"{lang['window']} {window_choice}")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del stft_matrix, canvas, freqs, times

# Affichage infos supplémentaires sur le signal : Mesures de puissance. Estimations de largeur de bande, rapidité de modulation & ACF.
def display_frq_info():
    print(lang["frq_info"])
    if iq_sig is None :
        if debug is True:
            print("Fichier non chargé")
        return
    wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_sig)))**2
    f = np.linspace(s_rate/-2, s_rate/2, len(iq_sig))
    f_pmax = f[np.argmax(wav_mag)]
    f_pmin = f[np.argmin(wav_mag)]
    _, psd = mg.compute_dsp(iq_sig, s_rate, N, overlap, window_choice)
    max_lvl = 10*np.log10(np.max(psd))
    low_lvl = 10*np.log10(np.min(psd))
    mean_lvl = 10*np.log10(np.mean(psd))
    estim_speed_2 = round(np.abs(em.mean_threshold_spectrum(iq_sig, s_rate)[2]),2)
    estim_speed = round(np.abs(em.power_spectrum_envelope(iq_sig, s_rate)[2]),2)
    _,_,_, peak_squared_freq, peak_quartic_freq = em.power_series(iq_sig, s_rate)
    estim_speed_3 = [round(abs(peak_squared_freq),2),round(abs(peak_quartic_freq),2)]
    _, freq_diff = em.frequency_transitions(iq_sig, s_rate, diff_window, window_choice)
    freq_diff /= np.max(np.abs(freq_diff))
    estim_speed_4 = round(float(dm.estimate_baud_rate(freq_diff, s_rate)),2)
    estim_speed_5 = round(np.abs(em.envelope_spectrum(iq_sig, s_rate)[2]),2)
    acf_peak = round(np.abs(em.autocorrelation_peak(iq_sig, s_rate, min_distance=acf_min_distance)[1]),2)

    # estimation de rapidité de modulation via différentes méthodes et indicateur de confiance
    confidence = 1
    if abs(estim_speed - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    if abs(estim_speed_2 - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    if abs(estim_speed_3[0]*2 - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    if abs(estim_speed_3[1]*2 - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    if abs(estim_speed_5 - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1

    if confidence < 2:
        confidence_level = lang['low_confidence']
        estim_speed_ag = f"{estim_speed_4} Bds ({confidence_level})"
    elif confidence >= 2 and confidence < 3:
        confidence_level = lang['medium_confidence']
        estim_speed_ag = f"{estim_speed_4} Bds ({confidence_level})"
    elif confidence >= 3:
        confidence_level = lang['high_confidence']
        estim_speed_ag = f"{estim_speed_4} Bds ({confidence_level})"

    popup = tk.Toplevel()
    popup.title(lang["frq_info"])
    place_relative(popup, root, 600, 200)
    tk.Label(popup, text=f"{lang['high_freq']}: {f_pmax:.2f} Hz. {lang['low_freq']}: {f_pmin:.2f} Hz\n \
                    {lang['high_level']} {max_lvl:.2f}. {lang['low_level']} {low_lvl:.2f}.\n \
                    {lang['mean_level']} {mean_lvl:.2f} dB ").pack()
    tk.Label(popup, text=f"{lang['estim_bw']} {round(bw,2)} Hz").pack()
    tk.Label(popup, text=f"{lang['estim_speed']} {estim_speed_ag}").pack()
    tk.Label(popup, text=f"{lang['acf_peak_txt']} {acf_peak} ms" ).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()

    if debug is True:
        print("Fréquence à la puissance maximale: ", f_pmax)
        print("Niveau le plus haut: ", max_lvl)
        print("Fréquence à la puissance minimale: ", f_pmin)
        print("Niveau le plus bas: ", low_lvl)
        print("Niveau moyen en dB: ", mean_lvl)
        print("Largeur de bande estimée: ", bw, " Hz")
        print("Rapidité de modulation estimée avec la fonction mts: ", estim_speed_2, " Bauds")
        print("Rapidité de modulation avec la FFT de puissance classique: ", estim_speed, " Bauds")
        print("Rapidité de modulation estimée par signal puissance: ", estim_speed_3[0]*2, "/", estim_speed_3[1]*2, "Bauds")
        print("Rapidité de modulation estimée par transitions de fréquence: ", estim_speed_4, " Bauds")
        print("Confiance dans l'estimation de rapidité de modulation: ", confidence, "/5")
        print("Autocorrélation estimée: ", acf_peak, " ms")
    del _,estim_speed,estim_speed_2, wav_mag, f, f_pmax, f_pmin, max_lvl, low_lvl, mean_lvl, freq_diff, acf_peak

# Affichage des informations du fichier : nom, encodage, durée, fréquence d'échantillonnage, taille de la fenêtre FFT
def display_file_info():
    if filepath is None:
        info_label.config(text=lang["no_file"])
        return
    info_label.config(text=f"{filepath}. {lang['encoding']} {find_sample_width(filepath)[0]} bits. Format {find_sample_width(filepath)[1]}. \
                      \n{lang['samples']} {len(iq_sig)}. {lang['sampling_frq']} {s_rate} Hz. {lang['duree']}: {len(iq_sig)/s_rate:.2f} sec.\
                      \n {lang['fft_window']} {N}. Overlap : {overlap}. {lang['f_resol']} {s_rate/N:.2f} Hz.")
    if debug is True:
        print("Affichage des informations du fichier")
        print("Chargé: ", filepath)
        print("Encodage: ", find_sample_width(filepath)[0], " bits")
        print("Format: ", find_sample_width(filepath)[1])
        print("Echantillons: ", len(iq_sig))
        print("Fréquence d'échantillonnage: ", s_rate, " Hz")
        print("Durée: ", len(iq_sig)/s_rate, " secondes")
        print("Taille fenêtre FFT: ", N)
        print("Recouvrement: ", overlap)

# Fonctions de traitement du signal (filtres, déplacement de fréquence, sous-échantillonnage, sur-échantillonnage, coupure)
def move_frequency():
    # déplacement de la fréquence centrale (valeur entrée par l'utilisateur)
    global iq_sig, s_rate
    fcenter = float(tk.simpledialog.askstring(lang["fc"], lang["move_txt"], parent=root))
    if fcenter is None:
        if debug is True:
            print("Modification de fréquence centrale annulée, valeur non définie")
        return
    iq_sig = iq_sig * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_sig))/s_rate)
    if debug is True:
        print("Fréquence centrale déplacée de ", fcenter, " Hz")
    plot_initial_graphs()

def move_frequency_cursors():
    # déplacement de la fréquence centrale sur le curseur (inactif si 2 curseurs)
    global cursor_points, iq_sig, s_rate
    if len(cursor_points) != 1 and (cursor_points[0][0] != cursor_points[1][0]):
        tk.messagebox.showinfo(lang["error"], lang["1pt_cursors"], parent=root)
        return
    fcenter = cursor_points[0][0]
    iq_sig = iq_sig * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_sig))/s_rate)
    if debug is True:
        print("Fréquence centrale déplacée de ", fcenter, " Hz")
    plot_initial_graphs()

def apply_filter_high_low():
    # passage d'un filtre passe-haut ou passe-bas
    global iq_sig, s_rate
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    popup.title(lang["high_low"])
    place_relative(popup, root, 300, 200)
    filter_type = tk.StringVar()
    filter_type.set(lang["low_val"])
    tk.Radiobutton(popup, text=lang["low_val"], variable=filter_type, value=lang["low_val"]).pack()
    tk.Radiobutton(popup, text=lang["high_val"], variable=filter_type, value=lang["high_val"]).pack()
    cutoff = tk.StringVar()
    tk.Label(popup, text=lang["freq_pass"]).pack()
    tk.Entry(popup, textvariable=cutoff).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if cutoff.get() == "":
        if debug is True:
            print("Filtre passe-", filter_type.get(), " non appliqué. Fréquence de coupure non définie")
        return
    if filter_type.get() == lang["low_val"]:
        iq_sig = sm.lowpass_filter(iq_sig, float(cutoff.get()), s_rate, filter_order)
    elif filter_type.get() == lang["high_val"]:
        iq_sig = sm.highpass_filter(iq_sig, float(cutoff.get()), s_rate, filter_order)
    if debug is True:
        print("Filtre passe-", filter_type.get(), " appliqué. Fréquence de coupure: ", cutoff.get(), "Hz")
    plot_initial_graphs()

def apply_filter_band():
    # passage d'un filtre passe-bande
    global iq_sig, s_rate
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy()) 
    popup.title(lang["bandpass"])
    place_relative(popup, root, 300, 200)
    lowcut = tk.StringVar()
    highcut = tk.StringVar()
    tk.Label(popup, text=lang["freq_low"]).pack()
    tk.Entry(popup, textvariable=lowcut).pack()
    tk.Label(popup, text=lang["freq_high"]).pack()
    tk.Entry(popup, textvariable=highcut).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if (lowcut.get() == "" and highcut.get() != "") or (lowcut.get() != "" and highcut.get() == ""):
        tk.messagebox.showinfo(lang["error"], lang["freq_valid"], parent=root)
        return
    elif lowcut.get() == "" and highcut.get() == "":
        return
    else:
        iq_sig = sm.bandpass_filter(iq_sig, float(lowcut.get()), float(highcut.get()), s_rate, filter_order)
    if debug is True:
        print("Filtre passe-bande appliqué. Fréquence de coupure basse: ", lowcut.get(), "Hz. Fréquence de coupure haute: ", highcut.get(), "Hz")
    plot_initial_graphs()

def mean_filter():
    # filtre moyenneur
    global iq_sig
    # popup pour choisir entre appliquer ou définir le seuil
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy()) 
    popup.title(lang["mean"])
    place_relative(popup, root, 300, 200)
    mean_filter = tk.StringVar()
    mean_filter.set(lang["not_apply"])
    # Afficher sur la popup la valeur de la variable
    _, psd = mg.compute_dsp(iq_sig, s_rate, N, overlap, window_choice)
    iq_floor = 10*np.log10(np.mean(psd))
    iq_sig_db = 10*np.log10(np.abs(iq_sig)**2 / (s_rate * N))
    tk.Label(popup, text=lang["mean_level"] + str(iq_floor)).pack()
    tk.Radiobutton(popup, text=lang["not_apply"], variable=mean_filter, value=lang["not_apply"]).pack()
    tk.Radiobutton(popup, text=lang["apply_mean"], variable=mean_filter, value=lang["apply_mean"]).pack()
    tk.Radiobutton(popup, text=lang["def_level"], variable=mean_filter, value=lang["def_level"]).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if mean_filter.get() == lang["def_level"]:
        iq_floor = float(tk.simpledialog.askstring(lang["level"], lang["enter_level"], parent=root))
        iq_sig = np.where(iq_sig_db < iq_floor, 0, iq_sig)
        if iq_floor is None:
            if debug is True:
                print("Seuil de moyennage non défini")
            return
        if debug is True:
            print("Signal moyenné avec un seuil de ", iq_floor, " dB")
    elif mean_filter.get() == lang["apply_mean"]:
        iq_sig = np.where(iq_sig_db < iq_floor, 0, iq_sig)
        if debug is True:
            print("Signal moyenné avec un seuil de ", iq_floor, " dB")
    else:
        return
    
    print(lang["mean"])
    plot_initial_graphs()
    del _, psd, iq_floor, iq_sig_db

def downsample_signal():
    # sous-échantillonnage
    global iq_sig, s_rate
    rate = tk.simpledialog.askstring(lang["downsample"], lang["down_value"], parent=root)
    if rate is None:
        if debug is True:
            print("Taux de sous-échantillonnage non défini")
        return
    decimation_factor = int(rate)
    iq_sig, s_rate = sm.downsample(iq_sig, s_rate, decimation_factor)
    print(lang["sampling_frq"], s_rate, "Hz")
    plot_initial_graphs()
    display_file_info()

def upsample_signal():
    # sur-échantillonnage
    global iq_sig, s_rate
    rate = tk.simpledialog.askstring(lang["upsample"], lang["up_value"], parent=root)
    if rate is None:
        if debug is True:
            print("Taux de sur-échantillonnage non défini")
        return
    oversampling_factor = int(rate)
    iq_sig, s_rate = sm.upsample(iq_sig, s_rate, oversampling_factor)
    print(lang["sampling_frq"], s_rate, "Hz")
    plot_initial_graphs()
    display_file_info()

def polyphase_resample():
    # rééchantillonnage par méthode polyphasée
    global iq_sig, s_rate
    new_fs = tk.simpledialog.askstring(lang["resample_poly"], lang["resample_value"], parent=root)
    if new_fs is None:
        if debug is True:
            print("Taux de rééchantillonnage non défini")
        return
    new_fs = int(new_fs)
    iq_sig, s_rate = sm.resample_polyphase(iq_sig, s_rate, new_fs)
    if debug is True:
        print("Signal rééchantillonné à ", new_fs, " Hz via méthode polyphasée")
    plot_initial_graphs()
    display_file_info()

def cut_signal():
    # coupure du signal : entrer les points de début et de fin (en secondes)
    global iq_sig, s_rate
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 300, 300)
    popup.title(lang["cut_val"])
    start = tk.StringVar()
    end = tk.StringVar()
    tk.Label(popup, text=lang["start_cut"]).pack()
    tk.Entry(popup, textvariable=start).pack()
    tk.Label(popup, text=lang["end_cut"]).pack()
    tk.Entry(popup, textvariable=end).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if start.get() =="" and end.get() =="":
        # tk.messagebox.showinfo(lang["error"], lang["valid_cut"], parent=root)
        return
    if start.get() =="":
        start = 0
    else:
        start = int(float(start.get())*s_rate)
    if end.get() =="":
        end = int((len(iq_sig)/s_rate)*s_rate)
    else:
        end = int(float(end.get())*s_rate)
    iq_sig = iq_sig[start:end]
    if debug is True:
        print("Signal coupé de ", start, "échantillons à ", end, "échantillons, soit ", end-start, "échantillons restants")
        print("Nouvelle durée du signal : ", len(iq_sig)/s_rate, " secondes")
    plot_initial_graphs()
    display_file_info()

def cut_signal_cursors():
    # coupure du signal entre les 2 curseurs (ne prend en compte que la durée, pas l'écart en fréquence)
    global iq_sig, s_rate, cursor_points
    if len(cursor_points) < 2:
        tk.messagebox.showinfo(lang["error"], lang["2pt_cursors"], parent=root)
        return
    # signal coupé entre les 2 points. On ne sait pas quel point est le début et lequel est la fin, donc on prend les valeurs y les plus petites et les plus grandes
    start = int(cursor_points[0][1]*s_rate)
    end = int(cursor_points[1][1]*s_rate)
    print(cursor_points[1][1],cursor_points[0][1])   
    if cursor_points[1][1] < cursor_points[0][1]:
        iq_sig = iq_sig[end:start]
        if debug is True:
            print("Signal coupé de ", end, "échantillons à ", start, "échantillons, soit ", start-end, "échantillons restants")
            print("Nouvelle durée du signal : ", len(iq_sig)/s_rate, " secondes")
    else:
        iq_sig = iq_sig[start:end]
        if debug is True:
            print("Signal coupé de ", start, "échantillons à ", end, "échantillons, soit ", end-start, "échantillons restants")
            print("Nouvelle durée du signal : ", len(iq_sig)/s_rate, " secondes")
    plot_initial_graphs()
    display_file_info()

def center_signal_coarse():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # param de proéminence à ajuster
    iq_sig, center = df.center_signal(iq_sig, s_rate, prominence=peak_prominence)
    if debug is True:
        print(f"Signal centré grossièrement par déplacement de {round(center, 2)} Hz.")

    plot_initial_graphs()

def center_signal_fine():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    iq_sig, center = df.estimate_carrier_weighted(iq_sig, s_rate)
    if debug is True:
        print(f"Signal centré finement par déplacement de {round(center, 2)} Hz.")

    plot_initial_graphs()

def apply_doppler_correction():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # entrée de la vitesse en m/s
    freq_offset = tk.simpledialog.askstring(lang["doppler"], lang["doppler_txt"], parent=root)
    if freq_offset is None:
        if debug is True:
            print("Correction Doppler non appliquée, valeur non définie")
        return
    freq_offset = float(freq_offset)
    iq_sig,_ = sm.doppler_lin_shift(iq_sig, s_rate, 0, freq_offset)
    if debug is True:
        print(f"Doppler appliqué avec une correction de {freq_offset} Hz.")
    plot_initial_graphs()

# Fonctions de mesure de la rapidité de modulation
def psf():
    # Mesure de la rapidité de modulation par la spectre d'enveloppe du puissance. Polyvalente.
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["psf"])
    print("Génération de la FFT de puissance")
    if not filepath:
        print(lang["no_file"])
        return
    clock, f, peak_freq = em.power_spectrum_envelope(iq_sig, s_rate)
    clock = clock / np.max(clock)
    ax = plt.subplot()
    ax.plot(f,np.abs(clock))
    if abs(peak_freq) > 25:
        ax.axvline(x=-peak_freq, color='r', linestyle='--')
        ax.axvline(x=peak_freq, color='r', linestyle='--')
        ax.set_title(f"{lang['estim_peak']} {round(peak_freq,2)} Hz")
    else:
        ax.set_title(lang['estim_failed'])
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del clock, f, peak_freq, canvas

def mts():
    # Variation de la fonction précédente, plus efficace sur certains signaux. En général performant sur les signaux de modulation de phase
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["mts"])
    print(lang["mts"])
    if not filepath:
        print(lang["no_file"])
        return
    clock, f, peak_freq = em.mean_threshold_spectrum(iq_sig, s_rate)
    clock = clock / np.max(clock)
    ax = plt.subplot()
    ax.plot(f,np.abs(clock))
    # ligne en rouge du pic de puissance estimé sur le graphe à -peak_freq et peak_freq, sauf si < 25 Hz
    if abs(peak_freq) > 25:
        ax.axvline(x=-peak_freq, color='r', linestyle='--')
        ax.axvline(x=peak_freq, color='r', linestyle='--')
        ax.set_title(f"{lang['estim_peak']} {round(peak_freq,2)} Hz")
    else:
        ax.set_title(lang['estim_failed'])
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del clock, f, peak_freq, canvas

def pseries():
    # mesure de la rapidité de modulation par les ordres de puissance, efficace sur les signaux de modulation d'amplitude et de fréquence
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["pseries"])
    print(lang["pseries"])
    if not filepath:
        print(lang["no_file"])
        return
    spec = fig.add_gridspec(2, 1)
    fig.tight_layout()
    a0 = fig.add_subplot(spec[0, :])
    a1 = fig.add_subplot(spec[1, :])
    ax = (a0, a1)
    f, squared, quartic, peak_squared_freq, peak_quartic_freq = em.power_series(iq_sig, s_rate)
    squared = squared / np.max(squared)
    quartic = quartic / np.max(quartic)
    ax[0].plot(f, squared)
    ax[0].set_ylabel(f"{lang["norm_power"]} ^2")
    if abs(peak_squared_freq) > 25:
        ax[0].axvline(x=-peak_squared_freq, color='r', linestyle='--')
        ax[0].axvline(x=peak_squared_freq, color='r', linestyle='--')
        ax[0].set_title(f"{lang['estim_peak']} {round(peak_squared_freq,2)} Hz")
        if debug is True:
            print("Ecart estimé : ", round(np.abs(peak_squared_freq),2))
    ax[1].plot(f, quartic)
    ax[1].set_ylabel(f"{lang["norm_power"]} ^4")
    ax[1].set_xlabel(f"{lang['freq_xy']} [Hz]")
    if abs(peak_quartic_freq) > 25:
        ax[1].axvline(x=-peak_quartic_freq, color='r', linestyle='--')
        ax[1].axvline(x=peak_quartic_freq, color='r', linestyle='--')
        ax[1].set_title(f"{lang['estim_peak']} {round(peak_quartic_freq,2)} Hz")
        if debug is True:
            print("Ecart estimé : ", round(np.abs(peak_quartic_freq),2))

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del f, squared, quartic, canvas

def cyclospectrum():
    # Cyclospectre
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["cyclospectrum"])
    print(lang["cyclospectrum"])
    if not filepath:
        print(lang["no_file"])
        return
    f, cyclic_corr_avg, peak_freq = em.cyclic_spectrum_sliding_fft(iq_sig, s_rate, window=window_choice, frame_len=N, step=overlap)
    f = np.linspace(s_rate/-2, s_rate/2, len(cyclic_corr_avg))
    cyclic_corr_avg = cyclic_corr_avg / np.max(cyclic_corr_avg)
    ax = plt.subplot()
    ax.plot(f,cyclic_corr_avg)
    # on retire les pics de puissance autour de 0 Hz pour déterminer la rapidité de modulation
    if abs(peak_freq) > 25:
        # on affiche les pics de puissance à -peak_freq et peak_freq
        ax.axvline(x=-peak_freq, color='r', linestyle='--')
        ax.axvline(x=peak_freq, color='r', linestyle='--')
        ax.set_title(f"{lang['estim_peak']} {round(peak_freq,2)} Hz")
    else:
        ax.set_title(lang['estim_failed'])
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del cyclic_corr_avg, f, peak_freq, canvas

def envelope_spectrum():
    # Spectre de l'enveloppe du signal
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["envelope_spectrum"])
    print(lang["envelope_spectrum"])
    if not filepath:
        print(lang["no_file"])
        return
    envelope, f, peak_freq = em.envelope_spectrum(iq_sig, s_rate)
    envelope = envelope / np.max(envelope)
    print(peak_freq)
    ax = plt.subplot()
    ax.plot(f,envelope)
    if abs(peak_freq) > 25:
        ax.axvline(x=-peak_freq, color='r', linestyle='--')
        ax.axvline(x=peak_freq, color='r', linestyle='--')
        ax.set_title(f"{lang['estim_peak']} {round(peak_freq,2)} Hz")
    else:
        ax.set_title(lang['estim_failed'])
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del envelope, f, peak_freq, canvas

def dsp():
    # affichage de la densité spectrale de puissance et de la bande passante estimée
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text, bw
    print(lang["dsp"])
    print(lang["bandwidth"])
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["dsp"])
    ax = fig.add_subplot()
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_sig, s_rate, N,overlap,window_choice)
    Pxx_shifted = np.fft.fftshift(Pxx) 
    f_centered = np.linspace(-s_rate/2, s_rate/2, len(f)) # pas de point en dehors de la bande passante, donc évite les artefacts de bords
    ax.plot(f_centered, Pxx_shifted)
    ax.set_xlim(-s_rate/2, s_rate/2)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["power_scale"])
    ax.axvline(x=fmax, color='r', linestyle='--')
    ax.axvline(x=fmin, color='r', linestyle='--')
    ax.set_title(f"{lang['bandwidth']} : {round(bw,2)} Hz")
    if debug is True:
        print("Bande passante estimée: ", bw, " Hz")
        print("Fréquence max BW: ", fmax, " Hz")
        print("Fréquence min BW: ", fmin, " Hz")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del fmax, fmin, canvas, f, Pxx

# fonc dsp max
def dsp_max():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["dsp_max"])
    print(lang["dsp_max"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_sig)))**2
    wav_mag = wav_mag / np.max(wav_mag)
    f = np.linspace(s_rate/-2, s_rate/2, len(iq_sig)) # frq en Hz
    ax.plot(f, wav_mag)
    ax.plot(f[np.argmax(wav_mag)], np.max(wav_mag), 'rx') # show max
    ax.grid()
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas, wav_mag, f
    
def constellation():
    # affichage de la constellation du signal. Fortement dépendant d'un bon calibrage de la fréquence centrale
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["constellation"])
    print(lang["constellation"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    iq_constel = iq_sig/np.max(np.abs(iq_sig))
    ax.scatter(np.real(iq_constel), np.imag(iq_constel), s=1)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas

# Fonctions d'autocorrélation
def autocorr():
    # Autocorrélation du signal sur la FFT : rapide, mais moins précis. En général, satisfaisant
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["autocorr"])
    print(lang["autocorr"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    yx, lags = em.autocorrelation(iq_sig, s_rate)
    ax.plot(lags*1e3, yx/np.max(yx)) # lags en ms
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel(lang["autocorr"])
    peak, time = em.autocorrelation_peak_from_acf(yx, lags, min_distance=acf_min_distance)
    ax.axvline(x=time, color='r', linestyle='--')
    ax.set_title(f"{lang['acf_peak_txt']} {round(time,4)} ms")
    if debug is True:
        print("Pic d'autocorrélation trouvé à ", time, " ms, ", peak, " Hz")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del yx, lags, canvas

def autocorr_full():
    # Autocorrélation complète du signal : plus précis, mais plus lent
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["autocorr_full"])
    print(lang["autocorr_full"])
    if not filepath:
        print(lang["no_file"])
        return
    if not tk.messagebox.askokcancel(lang["autocorr_full"], lang["confirm_wait"], parent=root):
        return
    ax = plt.subplot()
    # Params cyclospectre
    if debug is True:
        print("Peut générer des ralentissements. Patienter")
    yx, lags = em.full_autocorrelation(iq_sig)
    ax.plot(lags/s_rate*1e3, yx/np.max(yx)) # lags en ms
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel(lang["autocorr"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del yx, lags, canvas

#SCF
def scf():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["scf"])
    print(lang["scf"])
    if not filepath:
        print(lang["no_file"])
        return
    if not tk.messagebox.askokcancel(lang["scf"], lang["confirm_wait"], parent=root):
        return
    ax = plt.subplot()
    if debug is True:
        print("Peut générer des ralentissements. Patienter")
    scf, faxis, alphas = em.scf_tsm(iq_sig, s_rate, N, window_choice, overlap, alpha_step_hz=scf_alpha_step)
    extent = (faxis[0], faxis[-1], alphas[-1], alphas[0])
    ax.imshow(scf, aspect='auto', extent=extent, cmap='jet', origin='upper')
    ax.set_xlabel(f"{lang["freq_xy"]} [Hz]")
    ax.set_ylabel(f"{lang["cyclic_f"]} [Hz]")
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del scf, faxis, alphas, extent, canvas

# Fonctions de mesures de transitions de phase et de fréquence
def phase_difference():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["diff_phase"])
    print(lang["diff_phase"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    time, phase_diff = em.phase_time_angle(iq_sig, s_rate, diff_window, window_choice)
    ax.plot(time, phase_diff)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel(f"{lang['diff_phase']} [rad]")
    ax.set_title(f"{lang['smoothing']} {diff_window}")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del time, phase_diff, canvas

def freq_difference():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["diff_freq"])
    print(lang["diff_freq"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    time, freq_diff = em.frequency_transitions(iq_sig, s_rate, diff_window, window_choice)
    ax.plot(time, freq_diff)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel(f"{lang['freq_xy']} [Hz]") 
    ax.set_title(f"{lang['smoothing']} {diff_window}")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del time, freq_diff, canvas

def phase_cumulative():
    # fonc expérimentale de distribution de phase. A évaluer/améliorer
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(f"{lang['distrib_phase']} ")
    print(f"{lang['distrib_phase']} ")
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    hist, bins = em.phase_cumulative_distribution(iq_sig, num_bins=hist_bins)
    ax.plot(bins, hist)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel(lang["density"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del hist, bins, canvas

def frequency_cumulative():
    # fonc expérimentale de distribution fréquence instantanée. A évaluer/améliorer
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(f"{lang['distrib_freq']} ")
    print(f"{lang['distrib_freq']} ")
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    hist, bins = em.frequency_cumulative_distribution(iq_sig, s_rate, num_bins=hist_bins, window_size=diff_window, window_type=window_choice)
    ax.plot(bins, hist)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["density"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del hist, bins, canvas

# params pour les fonctions de différence de phase et de fréquence
def set_diff_params():
    global diff_window
    diff_window = tk.simpledialog.askstring(lang["params"], lang["smoothing_val"], parent=root)
    if diff_window is None:
        diff_window = 0
        if debug is True:
            print("Pas de fenêtre de lissage des transitions définie")
        return
    diff_window = int(diff_window)
    if debug is True:
        print("Fenêtre de lissage des transitions définie à ", diff_window)

def morlet_wavelet():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["morlet_cwt"])
    if not filepath:
        print(lang["no_file"])
        return
    if not tk.messagebox.askokcancel(lang["morlet_cwt"], lang["confirm_wait"], parent=root):
        return
    if debug is True:
        print("Génération de la transformée en ondelettes de Morlet")
        print("Peut générer des ralentissements. Patienter")
    ax = plt.subplot()
    coefs, center_freqs = df.morlet_cwt(iq_sig, fs=s_rate, nfreq=morlet_nfreq, w=morlet_fc)
    times = np.arange(coefs.shape[1]) / s_rate
    power = np.abs(coefs)
    power = power / np.max(power)
    im = ax.imshow(power, extent=[times[0], times[-1], center_freqs[0], center_freqs[-1]], aspect='auto', cmap='jet', origin='lower')
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel("Morlet  freq [π rad/sample]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(lang["norm_power"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del coefs, center_freqs, times, canvas


# OFDM
def alpha_from_symbol():
    global Tu, toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # calcul de alpha frq à partir de la durée symbole OFDM (Tu), estimée par l'utilisateur avec la fonction d'autocorrélation
    # demander à l'utilisateur de rentrer la durée estimée de Tu
    Tu = tk.simpledialog.askstring(lang["alpha"], lang["estim_tu"], parent=root)
    if Tu is None:
        if debug is True:
            print("Durée symbole OFDM non définie")
        return
    clear_plot()
    Tu = float(Tu)
    peak, alpha, caf = em.estimate_alpha(iq_sig, s_rate, Tu)
    fig = plt.figure()
    fig.suptitle(lang['alpha'])
    ax = plt.subplot()
    ax.plot(alpha, 10*np.log10(caf/np.max(caf)))
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["mag"])
    if peak != 0:
        ax.axvline(x=peak, color='r', linestyle='--')
        ax.set_title(f"{lang['alpha_peak']} {round(peak,4)} Hz")
    if debug is True:
        print("Calcul de alpha à partir de la durée symbole OFDM estimée par l'utilisateur")
        print("Durée estimée de Tu: ", Tu, "ms")
        print("Fréquence alpha estimée: ", peak, "Hz")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del peak, alpha, caf, canvas

def ofdm_results():
    global Tu, toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # Calcul des paramètres OFDM à partir de la fréquence alpha0 estimée par l'utilisateur
    # Demander à l'utilisateur de rentrer la fréquence alpha0 estimée
    alpha_0 = tk.simpledialog.askstring(lang["alpha"], lang["alpha0"], parent=root)
    if alpha_0 is None:
        if debug is True:
            print("Fréquence alpha0 non définie")
        return
    alpha_0 = float(alpha_0)
    dsp()
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_sig, s_rate, N,overlap,window_choice)
    # affiche BW estimée et demande de valider ou de redéfinir la bande passante
    popup = tk.Toplevel()
    place_relative(popup, root, 600, 100)
    popup.bind("<Return>", lambda event: popup.destroy())
    popup.title(lang["bandwidth"])
    tk.Label(popup, text=f"{lang['estim_bw']} :{round(bw,2)} Hz").pack()
    new_bw = tk.StringVar()
    tk.Label(popup, text=f"{lang['unreliable_sscarrier']}\n{lang['redef_bw'] }").pack()
    tk.Entry(popup, textvariable=new_bw).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()

    if new_bw.get() == "":
        if debug is True:
            print("Bande passante non définie")
        new_bw = bw
    else:
        new_bw = float(new_bw.get())
    if debug is True:
        print("Bande passante redéfinie à ", new_bw, " Hz")
        
    print(lang["ofdm_results"])
    Tu, Tg, Ts, Df, num = em.calc_ofdm(alpha_0, Tu, new_bw)
    if debug is True:
        print("Paramètres OFDM calculés")
        print("Tu = ", Tu, "ms")
        print("Tg = ", Tg, "ms")
        print("Ts = ", Ts, "ms")
        print("Δf = ", Df, "Hz")
        print("Nombre de sous-porteuses: ", num)
    # afficher les résultats dans une fenêtre
    popup = tk.Toplevel()
    place_relative(popup, root, 400, 200)
    popup.title(lang["ofdm_results"])
    tk.Label(popup, text=f"{lang['tu']} = {Tu:.6f} ms").pack()
    tk.Label(popup, text=f"{lang['tg']} = {Tg:.6f} ms").pack()
    tk.Label(popup, text=f"{lang['ts']} = {Ts:.6f} ms").pack()
    tk.Label(popup, text=f"{lang['df']} = {Df:.6f} Hz").pack()
    tk.Label(popup, text=f"{lang['num_ssp']} = {num}").pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    
    clear_plot()
    # afficher les sous-porteuses OFDM sur la DSP du signal avec la bande passante estimée, qui sert de fenêtre dans laquelle les sous-porteuses sont affichées
    fig = plt.figure()
    fig.suptitle(lang["dsp"])
    ax = fig.add_subplot()
    Pxx_shifted = np.fft.fftshift(Pxx) 
    f_centered = np.linspace(-s_rate/2, s_rate/2, len(f)) # même échelle x que le spectrogramme
    ax.plot(f_centered, Pxx_shifted)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["power_scale"])
    fmin, fmax = -new_bw/2, new_bw/2
    ax.axvline(x=fmax, color='r', linestyle='--')
    ax.axvline(x=fmin, color='r', linestyle='--')
    ax.set_title(f"{lang['bandwidth']} : {round(new_bw,2)} Hz")
    for i in range(num):
        ax.axvline(x=fmin + (i+1)*Df, color='g', linestyle='--')
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del fmax, fmin, f, Pxx, alpha_0, popup, canvas

## Fonctions de gestion des curseurs
# Calcule distance entre 2 points
def calculate_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx**2 + dy**2), dx, dy

# Cherche les pics dans les données
def find_peaks(data):
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i - 1] < data[i] > data[i + 1]:  # condition
            peaks.append(i)
    return peaks

# Récupère les données x,y du graphe actuel
def get_current_graph_data(ax):
    line = ax.lines[0]  # Les pics à chercher sont sur X et on suppose que c'est sur la 1ere ligne
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    return x_data, y_data

# Déplace le curseur sur le pic le plus proche
def move_cursor_to_nearest_peak():
    global cursor_points, cursor_lines, distance_text
    if not cursor_points:
        print("Pas de curseur défini.")
        return
    # Recupère position du curseur en x
    cursor_x = cursor_points[-1][0]
    # donnees du graphe
    x_data, y_data = get_current_graph_data(ax)
    # func cherche pics, puis pic le plus proche
    peaks = find_peaks(y_data)
    if not peaks:
        print("Pas de pic trouvé.")
        return
    distances = np.abs(x_data[peaks] - cursor_x)
    nearest_peak_idx = np.argmin(distances)
    nearest_peak_x = x_data[peaks][nearest_peak_idx]
    nearest_peak_y = y_data[peaks][nearest_peak_idx]
    # Dernier curseur sur le pic le plus proche
    cursor_points[-1] = (nearest_peak_x, nearest_peak_y)
    h_line, v_line = cursor_lines.pop()  # Retire le dernier curseur pour le placer sur le pic
    h_line.remove()
    v_line.remove()
    h_line = ax.axhline(nearest_peak_y, color='red', linestyle='--', linewidth=0.8)
    v_line = ax.axvline(nearest_peak_x, color='red', linestyle='--', linewidth=0.8)
    cursor_lines.append((h_line, v_line))
    # Calcul et affiche distance si 2 points
    if len(cursor_points) == 2:
        dist, dx, dy = calculate_distance(cursor_points[0], cursor_points[1])
        if distance_text:
            distance_text.remove()
        distance_text = ax.text(0.05, 0.95, f"ΔX: {dx:.6f}, ΔY: {dy:.6f}",
                                transform=ax.transAxes, ha='left', va='top', fontsize=10, color='blue')
    # Màj graphe
    fig.canvas.draw()
    if debug:
        print(f"Curseur déplacé à ({nearest_peak_x:.6f}, {nearest_peak_y:.6f})")
        if len(cursor_points) == 2:
            print(f"Delta_x: {dx}, Delta_y: {dy}")

# active/désactive curseurs
def toggle_cursor_mode():
    global cursor_mode, click_event_id
    cursor_mode = not cursor_mode
    mode_button.config(text=lang["cursors_on"] if cursor_mode else lang["cursors_off"])
    
    if cursor_mode:
        # active si option = on
        click_event_id = fig.canvas.mpl_connect('button_press_event', on_click)
        if debug is True:
            print("Curseurs activés")
    else:
        # désactive si option = off
        if click_event_id is not None:
            fig.canvas.mpl_disconnect(click_event_id)
            click_event_id = None
            if debug is True:
                print("Curseurs désactivés")

# Gère curseurs sur clic
def on_click(event):
    global cursor_points, cursor_lines, distance_text  
    if event.inaxes:  # Verif si dans le graphe
        # Ajoute un pt
        cursor_points.append((event.xdata, event.ydata))
        # Dessine croix rouge
        if type(ax) is not tuple:
            h_line = ax.axhline(event.ydata, color='red', linestyle='--', linewidth=0.8)
            v_line = ax.axvline(event.xdata, color='red', linestyle='--', linewidth=0.8)
            cursor_lines.append((h_line, v_line))
        else:
            # check quel subplot est cliqué
            for a_num in ax:
                if a_num == event.inaxes:
                    h_line = a_num.axhline(event.ydata, color='red', linestyle='--', linewidth=0.8)
                    v_line = a_num.axvline(event.xdata, color='red', linestyle='--', linewidth=0.8)
                    cursor_lines.append((h_line, v_line))
                    break
        # Garde uniquement les 2 derniers points
        if len(cursor_points) > 2:
            cursor_points.pop(0)
            old_h_line, old_v_line = cursor_lines.pop(0)
            old_h_line.remove()
            old_v_line.remove()
        # Calcul et affiche distance si 2 points
        if len(cursor_points) == 2:
            dist, dx, dy = calculate_distance(cursor_points[0], cursor_points[1])
            # Màj ou crée texte distances
            if distance_text:
                distance_text.remove()
            if type(ax) is not tuple:
                distance_text = ax.text(0.05, 0.95, f"ΔX: {dx:.6f}, ΔY: {dy:.6f}", # ignore "dist" : distance en diagonale
                                transform=ax.transAxes, ha='left', va='top', fontsize=10, color='blue')
            else:
                for a_num in ax:
                    if a_num == event.inaxes:
                        distance_text = a_num.text(0.05, 0.95, f"ΔX: {dx:.6f}, ΔY: {dy:.6f}",
                                        transform=a_num.transAxes, ha='left', va='top', fontsize=10, color='blue')
        # Màj
        fig.canvas.draw()
        if debug is True:
            print("Curseur ajouté à ", event.xdata, event.ydata)
            if len(cursor_points) == 2:
                print("Delta_x: ", dx, " Delta_y: ", dy)

# Nettoie curseurs
def clear_cursors():
    global cursor_points, cursor_lines, distance_text
    # Retire lignes
    for h_line, v_line in cursor_lines:
        h_line.remove()
        v_line.remove()
    cursor_lines.clear()
    cursor_points.clear()
    # Retire texte
    if distance_text:
        distance_text.remove()
        distance_text = None
    if debug is True:
        print("Curseurs retirés")
    # Redessine
    fig.canvas.draw()

def place_relative(popup, parent, w=300, h=200):
    popup.withdraw()  # cache avant de positionner
    parent.update_idletasks()

    # coordonnées de la fenetre principale
    x = parent.winfo_rootx()
    y = parent.winfo_rooty()
    ph = parent.winfo_height()

    # centre la popup sur la gauche de la fenetre principale
    xpos = x
    ypos = y + (ph - h) // 2

    popup.geometry(f"{w}x{h}+{xpos}+{ypos}")
    popup.transient(parent)
    popup.deiconify()  # affiche la popup
    popup.lift()

# sauvegarde le signal (modifié ou non) en nouveau fichier wav
def save_as_wav():
    global iq_sig, s_rate
    filename = tk.filedialog.asksaveasfilename(title=lang["save_wav"],defaultextension=".wav", filetypes=[("Waveform Audio File", "*.wav"), ("All Files", "*.*")])
    # Normalise les données IQ pour éviter le clipping
    max_amplitude = np.max(np.abs(iq_sig))
    if max_amplitude > 0:
        iq_data_normalized = iq_sig / max_amplitude
    else:
        iq_data_normalized = iq_sig
    # Formate en 16 bits et sépare reel/imaginaire
    if debug is True:
        print("Conversion en 2 voies 16 bits")
    real_part = (iq_data_normalized.real * 32767).astype(np.int16)
    imag_part = (iq_data_normalized.imag * 32767).astype(np.int16)
    # transforme reel+imag en 2 canaux pour le wav
    stereo_data = np.column_stack((real_part, imag_part))
    wav.write(filename, s_rate, stereo_data)
    if debug is True:
        print("Ecriture du nouveau wav")

# Démodulation FSK(CPM NRZ)/PSK 2 et 4
def demod_cpm_psk():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # vars pour fonctions de démod
    target_rate = None
    order = None
    mapping = None
    mod_type = None
    # on demande rapidité, ordre et mapping
    def toggle_mapping():
        # mapping seulement si ordre 4
        if param_order.get() == lang["param_order4"]:
            mapping_nat.config(state=tk.NORMAL)
            mapping_gray.config(state=tk.NORMAL)
            # mapping_custom.config(state=tk.NORMAL)
            param_mapping.set(lang["mapping_nat"])
        else:
            mapping_nat.config(state=tk.DISABLED)
            mapping_gray.config(state=tk.DISABLED)
            # mapping_custom.config(state=tk.DISABLED)
    def toggle_diff_offset():
        # seulement si PSK
        if modulation_type.get() == lang["demod_psk"]:
            param_diff.config(state=tk.NORMAL)
            param_offset.config(state=tk.NORMAL)
            # ajouter pi/4 : déjà prêt dans la fonction de demod
        else:
            param_diff.config(state=tk.DISABLED)
            param_offset.config(state=tk.DISABLED)

    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 350, 350)
    popup.title(lang["demod_param"])
    modulation_type = tk.StringVar()
    tk.Label(popup, text=lang["demod_type"]).pack()
    tk.Radiobutton(popup, text=lang["demod_fsk"], variable=modulation_type, value=lang["demod_fsk"], command=toggle_diff_offset).pack()
    tk.Radiobutton(popup, text=lang["demod_psk"], variable=modulation_type, value=lang["demod_psk"], command=toggle_diff_offset).pack()
    modulation_type.set(lang["demod_fsk"])

    param_order = tk.StringVar()
    tk.Label(popup, text=lang["param_order"]).pack()
    tk.Radiobutton(popup, text=lang["param_order2"], variable=param_order, value=lang["param_order2"], command=toggle_mapping).pack()
    tk.Radiobutton(popup, text=lang["param_order4"], variable=param_order, value=lang["param_order4"], command=toggle_mapping).pack()
    param_order.set(lang["param_order2"])
    target_rate = tk.StringVar()
    tk.Label(popup, text=lang["demod_speed"]).pack()
    tk.Entry(popup, textvariable=target_rate).pack()
    param_mapping = tk.StringVar()
    param_mapping.set(lang["mapping"])
    mapping_nat = tk.Radiobutton(popup, text=lang["mapping_nat"], variable=param_mapping, value=lang["mapping_nat"], state=tk.DISABLED)
    mapping_nat.pack()
    mapping_gray = tk.Radiobutton(popup, text=lang["mapping_gray"], variable=param_mapping, value=lang["mapping_gray"], state=tk.DISABLED)
    mapping_gray.pack()
    # mapping_custom = tk.Radiobutton(popup, text=lang["mapping_custom"], variable=param_mapping, value=lang["mapping_custom"], state=tk.DISABLED)
    # mapping_custom.pack()
    use_diff = tk.BooleanVar(value=False)
    use_offset = tk.BooleanVar(value=False)
    param_diff = tk.Checkbutton(popup, text=lang["param_diff"], variable=use_diff, state=tk.DISABLED)
    param_diff.pack()
    param_offset = tk.Checkbutton(popup, text=lang["param_offset"], variable=use_offset, state=tk.DISABLED)
    param_offset.pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()    
    popup.wait_window()

    if target_rate.get() == "":
        if debug is True:
            print("Pas de choix de rapidité de démodulation : Echec")
        return
    elif float(target_rate.get()) < 1:
        if debug is True:
            print("Pas de rapidité de démodulation définie. Essai aveugle")
        target_rate = None
    else:
        target_rate = float(target_rate.get())
    if param_order.get() == lang["param_order2"]:
        order = 2
    elif param_order.get() == lang["param_order4"]:
        order = 4
    if param_mapping.get() == lang["mapping_nat"]:
        mapping = "natural"
    elif param_mapping.get() == lang["mapping_gray"]:
        mapping = "gray"
    elif param_mapping.get() == lang["mapping_custom"]:
        mapping = tk.simpledialog.askstring(lang["mapping_custom"], lang["mapping_custom_desc"], parent=root)
        # non utilisé. à modifier dans demod.py, accepte actuellement chaîne de type "(0,0),(0,1),(0,2),(0,3)"
        mapping = list(eval(mapping))

    if modulation_type.get() == lang["demod_psk"]:
        time, diff = em.phase_time_angle(iq_sig, s_rate, diff_window, window_choice)
        mod_type = "PSK"
        if debug is True:
            print("Démodulation PSK sélectionnée")
        if order != 2 and order != 4:
            if debug is True:
                print("Ordre de modulation PSK non supporté")
            return
        if mapping == "gray":
            gray = True
        else:
            gray = False
        differential = use_diff.get()
        offset = use_offset.get()
        try:
            if target_rate is not None:
                clock = target_rate
                bits = dm.psk_demodulate(iq_sig, s_rate, clock, order, gray=gray, differential=differential, offset=offset, costas_damping=costas_damping, costas_bw_factor=costas_bw_factor)
            else:
                clock = dm.estimate_baud_rate(diff, s_rate)
                bits = dm.psk_demodulate(iq_sig, s_rate, clock, order, gray=gray, differential=differential, offset=offset, costas_damping=costas_damping, costas_bw_factor=costas_bw_factor)
            if differential and not offset:
                alt_psk = "D"
            elif offset and not differential:
                alt_psk = "O"
            elif differential and offset:
                alt_psk = "DO"
            else:
                alt_psk = ""
            if debug is True :
                print(f"Démodulation {alt_psk}{mod_type}{order} réalisée avec mapping {mapping}, rapidité {clock} bauds, bits: {len(bits)}")
        except:
            if debug is True:
                print("Echec de démodulation PSK")
            bits = 0

    else:
        time, diff = em.frequency_transitions(iq_sig, s_rate, diff_window, window_choice)
        diff /= np.max(np.abs(diff))
        mod_type = "CPM/FSK"
        if debug is True:
            print("Démodulation FSK sélectionnée")
        # fonction de démod et slice bits en fonction de l'ordre
        try:
            symbols, clock = dm.wpcr(diff, s_rate, target_rate, tau, precision, debug)
            if len(symbols) > 2 and order == 2:
                bits=dm.slice_binary(symbols)
                if debug is True:
                    print("Démodulation FSK 2 réalisée, bits: ", len(bits))
            elif len(symbols) > 2 and order == 4:
                bits = dm.slice_4ary(symbols,mapping)
                if debug is True :
                    print(f"Démodulation {mod_type} {order} réalisée avec mapping {mapping}, rapidité {clock} bauds, bits: {len(bits)}") 
        except:
                bits=0
        # plot des bits demodulés
    try:
        clear_plot()
        fig = plt.figure()
        fig.suptitle(lang["estim_bits"])
        if not filepath:
            print(lang["no_file"])
            return
        ax = plt.subplot()
        # regroupe bits en symboles
        num_bits = int(np.ceil(np.log2(order)))
        symbols = np.reshape(bits[: len(bits) // num_bits * num_bits], (-1, num_bits))
        bits_plot = np.array([int("".join(map(str, s)), 2) for s in symbols])
        if len(bits) > 5000: # graphe allégé si signal long
            bits_plot = bits_plot[:5000]
            fig.suptitle(f"{lang['estim_bits']} {lang["short_bits"]}")

        ax.plot(bits_plot, "o-")
        ax.set_xlabel("Bits")
        ax.set_ylabel(lang["bits_value"])
        if order == 2:
            charset = ["0", "1"]
        elif order == 4:
            charset = ["00", "01", "10", "11"]
        ax.set_yticks(range(len(charset)))
        ax.set_yticklabels(charset)
    except :
        clear_plot()
        fig = plt.figure()
        fig.suptitle(lang["bits_fail"])
        ax = plt.subplot()
        ax.plot(0)
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    # text box pour les bits démodulés sous format txt qui pourront être copiés
    text_box = scrolledtext.ScrolledText(plot_frame,height=6, wrap="char")
    text_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    text_box.config(state=tk.NORMAL)  # Temporarily enable text box
    try:
        if mod_type == "CPM/FSK" and order == 4:
            text_output= f"{mod_type}{order}. {lang['clock_frequency']} {clock} Hz, {mapping} mapping. {lang['estim_bits']} : {len(bits)}. \n"
        elif mod_type == "PSK" and order == 4:
            text_output= f"{alt_psk}{mod_type}{order} {clock} bauds, {mapping} mapping. {lang['estim_bits']} : {len(bits)}. \n"
        elif mod_type == "CPM/FSK" and order == 2:
            text_output= f"{mod_type}{order}. {lang['clock_frequency']} {clock} Hz. {lang['estim_bits']} : {len(bits)}. \n"
        elif mod_type == "PSK" and order == 2:
            text_output= f"{alt_psk}{mod_type}{order} {clock} bauds. {lang['estim_bits']} : {len(bits)}. \n"
        # lignes de bits
        max_display = 50000  # nombre max de bits à afficher
        display_bits = bits[:max_display]
        formatted_bits = "".join(map(str, display_bits))
        if len(bits) >= max_display:
            formatted_bits += f"\n... ({len(bits) - max_display} {lang['more_bits']})"
        text_output += formatted_bits
    except:
        display_bits, bits,bits_plot,formatted_bits = "","","", ""
        text_output = lang["bits_fail"]
    text_box.insert(tk.END, text_output)
    text_box.config(state=tk.DISABLED)
    del canvas, time, diff, bits, display_bits, bits_plot, formatted_bits, text_output, text_box

def demod_fm():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # demod fm
    try:
        iq_sig = dm.fm_demodulate(iq_sig,s_rate)
        if debug is True:
            print("Démodulation FM réalisée")
    except:
        if debug is True:
            print("Echec de démodulation FM")
            return
    plot_other_graphs()

def demod_am():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # demod am
    try:
        iq_sig = dm.am_demodulate(iq_sig)
        if debug is True:
            print("Démodulation AM réalisée")
    except:
        if debug is True:
            print("Echec de démodulation AM")
            return
    plot_other_graphs()

# EXPERIMENTAL
def demod_mfsk():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    time, freq_diff = em.frequency_transitions(iq_sig, s_rate, diff_window, window_choice)
    freq_diff /= np.max(np.abs(freq_diff))
    # vars pour fonctions de démod
    target_rate = None
    mapping = None
    spacing = None # espacement entre les symboles. Pour l'instant pas utilisé
    # on demande rapidité, ordre, espacement et mapping    
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 500, 250)
    popup.title(lang["demod_param"])
    # popup.geometry("350x250")
    param_method = tk.StringVar()
    tk.Radiobutton(popup, text=lang["mfsk_discrete_diff"], variable=param_method, value="main").pack()
    tk.Radiobutton(popup, text=lang["mfsk_tone_detection"], variable=param_method, value="alt").pack()
    param_method.set("alt")
    param_order = tk.StringVar()
    tk.Label(popup, text=lang["param_order"]).pack()
    tk.Entry(popup, textvariable=param_order).pack()
    target_rate = tk.StringVar()
    tk.Label(popup, text=lang["demod_speed"]).pack()
    tk.Entry(popup, textvariable=target_rate).pack()
    param_mapping = tk.StringVar()
    param_mapping.set(lang["mapping"])
    mapping_nat = tk.Radiobutton(popup, text=lang["mapping_nat"], variable=param_mapping, value=lang["mapping_nat"])
    mapping_nat.pack()
    mapping_gray = tk.Radiobutton(popup, text=lang["mapping_gray"], variable=param_mapping, value=lang["mapping_gray"])
    mapping_gray.pack()
    mapping_custom = tk.Radiobutton(popup, text=lang["mapping_non-binary"], variable=param_mapping, value=lang["mapping_non-binary"])
    mapping_custom.pack()
    param_mapping.set(lang["mapping_nat"])
    tk.Button(popup, text="OK", command=popup.destroy).pack()

    popup.wait_window()
    if target_rate.get() == "":
        if debug is True:
            print("Rapidité de démodulation non définie.")
        return
    elif float(target_rate.get()) < 1:
        print("Pas de rapidité de démodulation définie. Essai aveugle")
        target_rate = None
        clock = dm.estimate_baud_rate(freq_diff, s_rate, target_rate, precision, debug)
    else:
        clock = float(target_rate.get())
    order = int(param_order.get())
    if param_mapping.get() == lang["mapping_nat"]:
        return_format = "binary"
        mapping = "natural"
    elif param_mapping.get() == lang["mapping_gray"]:
        return_format = "binary"
        mapping = "gray"
    elif param_mapping.get() == lang["mapping_non-binary"]:
        mapping = "non-binary"
        return_format = "char"

    try:
        if debug is True:
            print("Démodulation MFSK en cours...")
        if param_method.get() == "main":
            symbols, clock = dm.wpcr(freq_diff, s_rate, clock, tau, precision, debug)
        elif param_method.get() == "alt":
        # EXPERIMENTAL
            tone_freqs, t, tone_idx, tone_freq, tone_powers, clock = dm.detect_and_track_mfsk_auto(iq_sig, s_rate, clock, num_tones=order, peak_thresh_db=mfsk_tresh_db, peak_prominence=mfsk_peak_prom_db, win_factor=mfsk_win_factor, hop_factor=mfsk_hop_factor, merge_bins=mfsk_bin_width_cluster_factor, switch_penalty=mfsk_viterbi_penalty)       
            tone_freq /= np.max(np.abs(tone_freq))
            symbols, _ = dm.wpcr(tone_freq, s_rate, target_rate=None, tau=tau, precision=precision, debug=debug)
        if len(symbols) > 2:
            bits = dm.slice_mfsk(symbols, int(param_order.get()), mapping, return_format)
            if debug is True:
                print(f"Démodulation MFSK réalisée avec mapping {mapping}, rapidité {clock} bauds, bits: {len(bits)}")
        else:
            bits = 0
        # plot des bits demodulés
        clear_plot()
        fig = plt.figure()
        fig.suptitle(lang["estim_bits"])
        if not filepath:
            print(lang["no_file"])
            return
        ax = plt.subplot()
        if return_format == "binary":
            # group bits into symbols
            num_bits = int(np.ceil(np.log2(order)))
            symbols = np.reshape(bits[: len(bits) // num_bits * num_bits], (-1, num_bits))
            bits_plot = np.array([int("".join(map(str, s)), 2) for s in symbols])
            # build charset dynamically for any order
            charset = [format(i, f"0{num_bits}b") for i in range(order)]
            ax.set_yticks(range(order))
            ax.set_yticklabels(charset)
        elif return_format == "char":
            charset = (string.digits + string.ascii_uppercase + string.ascii_lowercase + string.punctuation) # charset récupéré de la fonction demod.slice_mfsk
            charmap = {ch: i for i, ch in enumerate(charset)}
            bits_plot = np.array([charmap[ch] for ch in bits if ch in charmap])
            ax.set_yticks(range(len(charset)))
            ax.set_yticklabels(list(charset))
        else:
            bits_plot = bits
            ax.set_yticks(range(order))
            ax.set_yticklabels(range(order))

        if len(bits) > 5000:
            bits_plot = bits_plot[:5000]
            fig.suptitle(f"{lang['estim_bits']} {lang['short_bits']}")
        ax.plot(bits_plot, "o-")
        ax.set_xlabel("Bits")
        ax.set_ylabel(lang["bits_value"])
    except Exception as e:
        clear_plot()
        fig = plt.figure()
        fig.suptitle(lang["bits_fail"])
        ax = plt.subplot()
        ax.plot(0)
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    # text box pour les bits démodulés sous format txt qui pourront être copiés
    text_box = scrolledtext.ScrolledText(plot_frame, height=6, wrap="char")
    text_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    text_box.config(state=tk.NORMAL)
    try:
        if param_method.get() == "main":
            text_output = f"MFSK{order}. {lang['clock_frequency']} {clock} Hz, {mapping} mapping. {lang['estim_bits']} : {len(bits)}. \n"
        elif param_method.get() == "alt":
            text_output= f"MFSK{order} {clock} bauds, {mapping} mapping. {lang['estim_bits']} : {len(bits)}. \n"
        # lignes de bits
        if return_format == "int" or return_format == "char":
            # si format int, on sépare par une virgule chaque symbole
            formatted_bits = ",".join(map(str, bits))
            text_output += formatted_bits
        else:
            formatted_bits = "".join(map(str, bits))
            text_output += formatted_bits
    except:
        bits, bits_plot, formatted_bits = "", "", ""
        text_output = lang["bits_fail"]
    text_box.insert(tk.END, text_output)
    text_box.config(state=tk.DISABLED)
    del canvas, time, freq_diff, bits, bits_plot, formatted_bits, text_output, text_box

# Filtres supplémentaires
def apply_median_filter():
    global iq_sig, s_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Demande la taille du filtre. 4 options (Léger pour 3, Modéré pour 5, Agressif pour 9, sinon personnalisé) avec liste dans la fenêtre
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 300, 150)
    popup.title(lang["median_filter"])
    tk.Label(popup, text=lang["kernel_select"]).pack()
    kernel_size = tk.StringVar()
    options = [lang["light_filter"], lang["medium_filter"], lang["aggressive_filter"], lang["dynamic_filter"], "Custom"]
    kernel_size.set("Custom")  # Valeur par défaut
    kernel_size_menu = tk.OptionMenu(popup, kernel_size, *options)
    kernel_size_menu.pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()  # Attendre que l'utilisateur ferme la fenêtre
    if kernel_size.get() == "Custom":
        # Si l'utilisateur a choisi "Custom", on demande la taille du noyau
        kernel_size = tk.simpledialog.askinteger(lang["median_filter"], lang["kernel_size"], minvalue=1, parent=root)
    else:
        # Si l'utilisateur a choisi une option, on définit la taille du noyau en fonction de l'option choisie
        if kernel_size.get() == lang["light_filter"]:
            kernel_size = 3
        elif kernel_size.get() == lang["medium_filter"]:
            kernel_size = 5
        elif kernel_size.get() == lang["aggressive_filter"]:
            kernel_size = 9
        elif kernel_size.get() == lang["dynamic_filter"]:
            symbol_rate = tk.simpledialog.askinteger(lang["median_filter"], "Bauds", minvalue=1, parent=root)
            kernel_size = int(s_rate / (symbol_rate * 2)) | 1
        else:
            # Si l'utilisateur a choisi une option non reconnue, on affiche un message d'erreur
            tk.messagebox.showerror(lang["error"], lang["kernel_invalid"], parent=root)
            if debug is True:
                print("Taille du noyau invalide sélectionnée.")
            return
        if kernel_size is None:
            return  # Annulation
        elif kernel_size % 2 == 0:
            # Si la taille du noyau est paire, on l'incrémente de 1 pour qu'elle soit impaire
            kernel_size += 1
            if debug is True:
                print(f"La taille du noyau doit être un entier positif impair. Taille ajustée à {kernel_size}")
        elif kernel_size < 1:
            # afficher un message d'erreur dans l'interface
            tk.messagebox.showerror(lang["error"], lang["kernel_error"])
            if debug is True:
                print("La taille du noyau doit être un entier positif impair.")
            return
    try:
        iq_sig = sm.median_filter(iq_sig, kernel_size)
        if debug is True:
            print(f"Filtre médian de taille {kernel_size} appliqué")
    except Exception as e:
        if debug:
            print(f"Erreur lors de l'application du filtre médian: {e}")
    plot_initial_graphs()

def apply_moving_average():
    global iq_sig, s_rate
    if not filepath:
        print(lang["no_file"])
        return
    
    # Demande la taille du filtre. 4 options (Léger pour 3, Modéré pour 5, Agressif pour 9, sinon personnalisé) avec liste dans la fenêtre
    popup = tk.Toplevel()
    popup.title(lang["moving_average"])
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 300, 150)
    tk.Label(popup, text=lang["window_size_select"]).pack()
    window_size = tk.StringVar()
    options = [lang["light_filter"], lang["medium_filter"], lang["aggressive_filter"], lang["dynamic_filter"], "Custom"]
    window_size.set("Custom")  # Valeur par défaut
    window_size_menu = tk.OptionMenu(popup, window_size, *options)
    window_size_menu.pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()  # Attendre que l'utilisateur ferme la fenêtre
    if window_size.get() == "Custom":
        # Si l'utilisateur a choisi "Custom", on demande la taille de fenêtre
        window_size = tk.simpledialog.askinteger(lang["moving_average"], lang["filter_window"], minvalue=1, parent=root)
    else:
        # Si l'utilisateur a choisi une option, on définit la taille de fenêtre en fonction de l'option choisie
        if window_size.get() == lang["light_filter"]:
            window_size = 5
        elif window_size.get() == lang["medium_filter"]:
            window_size = 11
        elif window_size.get() == lang["aggressive_filter"]:
            window_size = 21
        elif window_size.get() == lang["dynamic_filter"]:
            symbol_rate = tk.simpledialog.askinteger(lang["moving_average"], "Bauds", minvalue=1, parent=root)
            window_size = int(s_rate / (symbol_rate)) | 1
        else:
            # Si l'utilisateur a choisi une option non reconnue, on affiche un message d'erreur
            tk.messagebox.showerror(lang["error"], lang["window_invalid"], parent=root)
            if debug is True:
                print("Taille de fenêtre invalide sélectionnée.")
            return
        if window_size is None:
            return  # Annulation
        elif window_size < 1:
            # afficher un message d'erreur dans l'interface
            tk.messagebox.showerror(lang["error"], lang["window_size_error"])
            if debug is True:
                print("La taille de la fenêtre doit être un entier positif.")
            return
    try:
        iq_sig = sm.moving_average(iq_sig, window_size)
        if debug is True:
            print(f"Moyenne mobile de taille {window_size} appliquée")
    except Exception as e:
        if debug:
            print(f"Erreur lors de l'application de la moyenne mobile: {e}")
    plot_initial_graphs()

def apply_fir_filter():
    global iq_sig, s_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Demande la fréquence de coupure et le nombre de taps
    cutoff_freq = tk.simpledialog.askfloat(lang["fir_filter"], lang["freq_pass"], minvalue=1, maxvalue=s_rate/2, parent=root)
    num_taps = tk.simpledialog.askinteger(lang["fir_filter"], lang["fir_taps"], minvalue=1, parent=root)
    if cutoff_freq is None or num_taps is None:
        return
    try:
        iq_sig = sm.fir_filter(iq_sig, s_rate, cutoff_freq, 'lowpass', num_taps)
        if debug is True:
            print(f"Filtre FIR avec fréquence de coupure {cutoff_freq} Hz et {num_taps} taps appliqué")
    except Exception as e:
        print(f"Erreur lors de l'application du filtre FIR: {e}")
    plot_initial_graphs()

def apply_wiener_filter():
    global iq_sig, s_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Demande la taille du filtre Wiener et le bruit
    # Note: le bruit est estimé par l'utilisateur, pas calculé
    size = tk.simpledialog.askinteger(lang["wiener_filter"], lang["size"], minvalue=1, parent=root)
    noise = tk.simpledialog.askfloat(lang["wiener_filter"], lang["noise_variance"], minvalue=0.0, parent=root)
    if noise is None and size is None:
        return # L'utilisateur a annulé. Une des 2 valeurs peut être nulle
    try:
        iq_sig = sm.wiener_filter(iq_sig, size, noise)
        if debug is True:
            print(f"Filtre Wiener de taille {size} appliqué avec variance de bruit {noise}")
    except Exception as e:
        if debug:
            print(f"Erreur lors de l'application du filtre Wiener: {e}")
    plot_initial_graphs()

# Fonction pour appliquer un filtre adapté
def apply_matched_filter():
    global iq_sig, s_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Popup pour sélectionner le filtre
    popup = tk.Toplevel()
    popup.title(lang["matched_filter"])
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 300, 150)
    tk.Label(popup, text=lang["pulse_shape"]).pack()
    pulse_shape_var = tk.StringVar()
    options = ['rectangular', 'gaussian', 'raised_cosine', 'root_raised_cosine', 'sinc', 'rsinc']
    pulse_shape_var.set('root_raised_cosine')  # default
    pulse_shape_menu = tk.OptionMenu(popup, pulse_shape_var, *options)
    pulse_shape_menu.pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    pulse_shape = pulse_shape_var.get()
    # Demande la rapidité de modulation
    symbol_rate = tk.simpledialog.askfloat(lang["matched_filter"], lang["symbol_rate"], minvalue=0.1, parent=root)
    if symbol_rate is None:
        return  # Annulation
    # Demande du facteur (optionnel)
    if pulse_shape in ('raised_cosine', 'root_raised_cosine', 'gaussian'):
        factor = tk.simpledialog.askfloat(lang["matched_filter"], lang["filter_factor"], minvalue=0.0, maxvalue=1.0, parent=root)
        if factor is None:
            return  # Annulation
    else:
        factor = None

    try:
        iq_sig = sm.matched_filter(iq_sig, s_rate, symbol_rate, factor=factor, pulse_shape=pulse_shape)
        if debug:
            print(f"Filtre adapté appliqué: {pulse_shape}, facteur={factor}, rapidité={symbol_rate}")
    except Exception as e:
        tk.messagebox.showerror(lang["error"], f"{lang["error_matched_filter"]}: {e}", parent=root)
        if debug:
            print(f"Erreur lors de l'application du filtre adapté : {e}")
    plot_initial_graphs()

def eye_diagram():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    if not filepath:
        print(lang["no_file"])
        return
    # Demande la rapidité de modulation
    baud_rate = tk.simpledialog.askfloat(lang["eye_diagram"], lang["demod_speed"], parent=root)
    if baud_rate is None or baud_rate < 0:
        if debug is True:
            print("Rapidité de modulation non définie ou invalide.")
        return
    elif baud_rate == 0:
        print("Pas de rapidité de modulation définie. Essai aveugle")
        symbol_rate = None
        try:
            freq_diff = em.frequency_transitions(iq_sig, s_rate, window_size=diff_window, window_type=window_choice)[1]
            symbol_rate = dm.estimate_baud_rate(freq_diff, s_rate, target_rate=baud_rate, precision=precision, debug=debug)
            if symbol_rate is None or symbol_rate <= 0:
                symbol_rate = baud_rate
            if debug is True:
                print(f"Rapidité estimée pour diagramme de l'oeil: {symbol_rate} bauds")
        except Exception as e:
            if debug is True:
                print(f"Echec de l'estimation de la rapidité: {e}")
                return
    else:
        symbol_rate = baud_rate
    try:
        time, traces, metrics = dm.eye_diagram_with_metrics(iq_sig, s_rate, symbol_rate, eye_channel, eye_num_traces, eye_symbols)
        if debug is True:
            print("Diagramme de l'oeil calculé")
            print("Metriques: ", metrics)
        # plot
        clear_plot()
        fig = plt.figure()
        fig.suptitle(f"{lang['eye_diagram']}. \n{lang['channel']} {eye_channel}, {lang['symbol_rate']} : {symbol_rate}")
        ax = plt.subplot()
        for trace in traces:
            ax.plot(time, trace, color='blue', alpha=0.18)
        ax.set_xlabel(lang["bits_value"])
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        # annotation des métriques
        textstr = (
            f"{lang['eye_height']} : {metrics['eye_height']}\n"
            f"{lang['eye_width']} : {metrics['eye_width']} {lang['bits_value']}\n"
            f"{lang['eye_opening_ratio']} : {metrics['eye_opening_ratio']}"
        )
        fig.text(0.02, 0.02, textstr, ha="left", va="bottom", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    except Exception as e:
        clear_plot()
        time, traces, metrics = None, None, None
        fig = plt.figure()
        fig.suptitle(lang["eye_fail"])
        ax = plt.subplot()
        ax.plot(0)
        if debug is True:
            print(f"Echec du diagramme de l'oeil: {e}")
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas, traces, time, metrics

def shift_frequency():
    global iq_sig, s_rate
    # décale la fréquence centrale de la moitié de fréquence d'échantillonnage : prend en compte les fichiers encodés "à l'envers"
    if not filepath:
        return
    iq_sig = iq_sig * ((-1) ** np.arange(len(iq_sig)))
    if debug is True:
        print("Décalage de fréquence Fe/2 appliqué")
    plot_initial_graphs()

def prepare_audio(sig):
    if sig.ndim > 1: # stereo to mono
        sig = sig[:, 0]
    max_val = np.max(np.abs(sig))
    if max_val > 1: # normalisation
        sig = sig / max_val
    return np.asarray(sig, dtype=np.float32)

# Play/Pause
def toggle_playback():
    global is_playing, is_paused
    if not is_playing:
        start_playback()
    else:
        # Pause si en cours de playback
        is_paused = not is_paused
        play_button.config(text="Resume" if is_paused else "Pause")

def start_playback():
    global is_playing, is_paused, audio_thread
    if not is_playing:
        is_playing = True
        is_paused = False
        play_button.config(text="Pause")
        audio_thread = Thread(target=play_audio)
        audio_thread.start()

def stop_playback():
    global is_playing, is_paused, audio_stream, stream_position
    if audio_stream and stream_position < total_samples :
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
    is_playing = False
    is_paused = False
    stream_position = 0
    play_button.config(text="Play")

def play_audio():
    global audio_stream, stream_position, is_playing, is_paused, total_samples

    try:
        prepared_audio = prepare_audio(iq_sig) # suppose demodulation deja faite, on convertit en réel
        total_samples = len(prepared_audio)

        def audio_callback(outdata, frames, time, status):
            global stream_position, is_paused, is_playing

            if status:
                print(status)

            if not is_playing:
                raise sd.CallbackStop()

            if is_paused:
                outdata[:] = np.zeros((frames, 1))
                return

            end_pos = stream_position + frames
            if end_pos > total_samples:
                end_pos = total_samples

            chunk = prepared_audio[stream_position:end_pos]

            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)))

            outdata[:] = chunk.reshape(-1, 1)
            stream_position = end_pos

            if stream_position >= total_samples:
                stop_playback()

        audio_stream = sd.OutputStream(callback=audio_callback, samplerate=s_rate, channels=1, dtype='float32')
        audio_stream.start()

    except:
        return

def update_progress_bar():
    global stream_position, progress
    if is_playing and not is_paused and iq_sig is not None:
        progress_value = (stream_position / len(iq_sig)) * 100
        progress["value"] = min(progress_value, 100)
    root.after(100, update_progress_bar)  # Update 100 ms

def audio_output():
    global play_button, stop_button, progress
    # Controles playback dans la meme frame que les graphes
    clear_plot()

    play_button = tk.Button(plot_frame, text="Play", command=toggle_playback)
    play_button.pack(side=tk.LEFT, padx=5, pady=5)

    stop_button = tk.Button(plot_frame, text="Stop", command=stop_playback)
    stop_button.pack(side=tk.LEFT, padx=5, pady=5)

    progress = ttk.Progressbar(plot_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress.pack(side=tk.LEFT, padx=5, pady=5)
    update_progress_bar()

def advanced_settings():
    # Distance min pour fc et nfreq pour CWT Morlet, 
    # (db_thresh,win_factor,hop_factor,peak_prominence,cluster_bin_width_factor,viteri_penalty) pour détection de tons MFSK,
    # eye diagram (num_traces, channel, symbols_per_trace), costas loop (zeta damping_factor,loop_bw_factor),
    global filter_order, peak_prominence, acf_min_distance, tau_modifier, precision, morlet_fc, morlet_nfreq
    global scf_alpha_step, costas_damping, costas_bw_factor, eye_channel, eye_num_traces, eye_symbols
    global mfsk_tresh_db, mfsk_peak_prom_db, mfsk_win_factor, mfsk_viterbi_penalty, mfsk_bin_width_cluster_factor, mfsk_hop_factor
    # Popup pour les paramètres avancés
    popup = tk.Toplevel()
    popup.title(lang["advanced_settings"])
    place_relative(popup, root, 600, 750)
    tk.Label(popup, text=lang["filter_order"]).pack()
    filter_order_var = tk.IntVar(value=filter_order) # valeur initiale
    tk.Entry(popup, textvariable=filter_order_var).pack() # ordre du filtre Butterworth
    tk.Label(popup, text=lang["peak_prominence"]).pack()
    peak_prominence_var = tk.DoubleVar(value=peak_prominence)
    tk.Entry(popup, textvariable=peak_prominence_var).pack() # proéminence des pics pour centrage
    tk.Label(popup, text=lang["acf_min_distance"]).pack()
    acf_min_distance_var = tk.IntVar(value=acf_min_distance)
    tk.Entry(popup, textvariable=acf_min_distance_var).pack() # distance min pour considérer pic ACF
    tk.Label(popup, text=lang["tau_modifier"]).pack()
    tau_modifier_var = tk.DoubleVar(value=tau_modifier)
    tk.Entry(popup, textvariable=tau_modifier_var).pack() # modificateur de tau pour wpcr
    tk.Label(popup, text=lang["precision"]).pack()
    precision_var = tk.DoubleVar(value=precision)
    tk.Entry(popup, textvariable=precision_var).pack() # précision pour recherche de rapidité
    tk.Label(popup,text=lang["morlet_params"]).pack()
    morlet_fc_var = tk.DoubleVar(value=morlet_fc)
    morlet_nfreq_var = tk.IntVar(value=morlet_nfreq)
    tk.Entry(popup, textvariable=morlet_fc_var).pack() # param fréquence centrale CWT Morlet
    tk.Entry(popup, textvariable=morlet_nfreq_var).pack() # nombre de fréquences CWT Morlet
    tk.Label(popup, text=lang["scf_alpha_step"]).pack()
    scf_alpha_step_var = tk.IntVar(value=scf_alpha_step)
    tk.Entry(popup, textvariable=scf_alpha_step_var).pack() # pas alpha pour SCF
    tk.Label(popup, text=lang["costas_damping"]).pack()
    costas_damping_var = tk.DoubleVar(value=costas_damping)
    tk.Entry(popup, textvariable=costas_damping_var).pack() # facteur d'amortissement pour Costas
    tk.Label(popup, text=lang["costas_bw_factor"]).pack()
    costas_bw_factor_var = tk.DoubleVar(value=costas_bw_factor)
    tk.Entry(popup, textvariable=costas_bw_factor_var).pack() # facteur de boucle bande passante pour Costas
    tk.Label(popup, text=lang["eye_diagram_params"]).pack()
    eye_channel_var = tk.StringVar(value=eye_channel)
    eye_num_traces_var = tk.IntVar(value=eye_num_traces)
    eye_symbols_var = tk.IntVar(value=eye_symbols)
    tk.Entry(popup, textvariable=eye_channel_var).pack() # canal pour diagramme de l'oeil
    tk.Entry(popup, textvariable=eye_num_traces_var).pack() # nombre de traces pour diagramme de l'oeil
    tk.Entry(popup, textvariable=eye_symbols_var).pack() # nombre de symboles par trace pour diagramme de l'oeil
    tk.Label(popup, text=lang["mfsk_tone_detection_params"]).pack()
    tk.Label(popup, text=lang["mfsk_tresh_db"]).pack()
    mfsk_tresh_db_var = tk.DoubleVar(value=mfsk_tresh_db)
    tk.Entry(popup, textvariable=mfsk_tresh_db_var).pack() # seuil en dB pour démodulation MFSK
    tk.Label(popup, text=lang["mfsk_peak_prom_db"]).pack()
    mfsk_peak_prom_db_var = tk.DoubleVar(value=mfsk_peak_prom_db)
    tk.Entry(popup, textvariable=mfsk_peak_prom_db_var).pack() # proéminence des pics pour démodulation MFSK
    tk.Label(popup, text=lang["mfsk_win_factor"]).pack()
    mfsk_win_factor_var = tk.DoubleVar(value=mfsk_win_factor)
    tk.Entry(popup, textvariable=mfsk_win_factor_var).pack() # facteur de taille de fenêtre pour démodulation MFSK
    tk.Label(popup, text=lang["mfsk_hop_factor"]).pack()
    mfsk_hop_factor_var = tk.DoubleVar(value=mfsk_hop_factor)
    tk.Entry(popup, textvariable=mfsk_hop_factor_var).pack() # facteur de saut pour démodulation MFSK
    tk.Label(popup, text=lang["mfsk_cluster_bin_width_factor"]).pack()
    mfsk_cluster_bin_width_factor_var = tk.DoubleVar(value=mfsk_bin_width_cluster_factor)
    tk.Entry(popup, textvariable=mfsk_cluster_bin_width_factor_var).pack() # facteur de largeur de bin pour clustering dans démodulation MFSK
    tk.Label(popup, text=lang["mfsk_viterbi_penalty"]).pack()
    mfsk_viterbi_penalty_var = tk.DoubleVar(value=mfsk_viterbi_penalty)
    tk.Entry(popup, textvariable=mfsk_viterbi_penalty_var).pack() # pénalité Viterbi pour démodulation MFSK

    def save_settings():
        nonlocal filter_order_var, peak_prominence_var, acf_min_distance_var, tau_modifier_var, precision_var
        nonlocal morlet_fc_var, morlet_nfreq_var, scf_alpha_step_var, costas_damping_var, costas_bw_factor_var
        nonlocal eye_channel_var, eye_num_traces_var, eye_symbols_var
        nonlocal mfsk_tresh_db_var, mfsk_peak_prom_db_var, mfsk_win_factor_var, mfsk_viterbi_penalty_var
        nonlocal mfsk_cluster_bin_width_factor_var, mfsk_hop_factor_var
        global filter_order, peak_prominence, acf_min_distance, tau_modifier, precision, morlet_fc, morlet_nfreq
        global scf_alpha_step, costas_damping, costas_bw_factor, eye_channel, eye_num_traces, eye_symbols
        global mfsk_tresh_db, mfsk_peak_prom_db, mfsk_win_factor, mfsk_viterbi_penalty, mfsk_bin_width_cluster_factor, mfsk_hop_factor
        filter_order = filter_order_var.get()
        peak_prominence = peak_prominence_var.get()
        acf_min_distance = acf_min_distance_var.get()
        tau_modifier = tau_modifier_var.get()
        precision = precision_var.get()
        morlet_fc = morlet_fc_var.get()
        morlet_nfreq = morlet_nfreq_var.get()
        scf_alpha_step = scf_alpha_step_var.get()
        costas_damping = costas_damping_var.get()
        costas_bw_factor = costas_bw_factor_var.get()
        eye_channel = eye_channel_var.get()
        eye_num_traces = eye_num_traces_var.get()
        eye_symbols = eye_symbols_var.get()
        mfsk_tresh_db = mfsk_tresh_db_var.get()
        mfsk_peak_prom_db = mfsk_peak_prom_db_var.get()
        mfsk_win_factor = mfsk_win_factor_var.get()
        mfsk_hop_factor = mfsk_hop_factor_var.get()
        mfsk_bin_width_cluster_factor = mfsk_cluster_bin_width_factor_var.get()
        mfsk_viterbi_penalty = mfsk_viterbi_penalty_var.get()
        if debug is True:
            print(f"Paramètres avancés mis à jour : "
                  f"filter_order={filter_order}, peak_prominence={peak_prominence}, acf_min_distance={acf_min_distance}, "
                  f"tau_modifier={tau_modifier}, precision={precision}, morlet_fc={morlet_fc}, morlet_nfreq={morlet_nfreq}, "
                  f"scf_alpha_step={scf_alpha_step}, costas_damping={costas_damping}, costas_bw_factor={costas_bw_factor}, "
                  f"eye_channel={eye_channel}, eye_num_traces={eye_num_traces}, eye_symbols={eye_symbols}, "
                  f"mfsk_tresh_db={mfsk_tresh_db}, mfsk_peak_prom_db={mfsk_peak_prom_db}, mfsk_win_factor={mfsk_win_factor}, "
                  f"mfsk_hop_factor={mfsk_hop_factor}, mfsk_bin_width_cluster_factor={mfsk_bin_width_cluster_factor}, "
                  f"mfsk_viterbi_penalty={mfsk_viterbi_penalty}")
        popup.destroy()
    tk.Button(popup, text="OK", command=save_settings).pack()

# Fonc de changement de langue. Recharge les labels des boutons et des menus. Couvre FR et EN uniquement
def change_lang():
    global lang
    # clic : fr -> en, en -> fr
    lang = ll.get_eng_lib() if lang['lang'] == "Français" else ll.get_fra_lib()
    if debug is True:
        print("Dict: ", lang["lang"])
    # reload txt
    load_lang_changes()

def load_lang_changes():
    global lang
    # supprime cascade & menu items
    menu_bar.delete(0, tk.END)
    # Barre de menu
    graphs_menu = tk.Menu(menu_bar, tearoff=0)
    info_menu = tk.Menu(menu_bar, tearoff=0)
    mod_menu = tk.Menu(menu_bar, tearoff=0)
    filter_menu = tk.Menu(menu_bar, tearoff=0)
    power_menu = tk.Menu(menu_bar, tearoff=0)
    diff_menu = tk.Menu(menu_bar, tearoff=0)
    speed_menu = tk.Menu(menu_bar, tearoff=0)
    acf_menu = tk.Menu(menu_bar, tearoff=0)
    ofdm_menu = tk.Menu(menu_bar, tearoff=0)
    freq_menu = tk.Menu(menu_bar, tearoff=0)
    demod_menu = tk.Menu(menu_bar, tearoff=0)
    if with_sound:
        audio_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label=lang["display"], menu=info_menu)
    menu_bar.add_cascade(label=lang["modify"], menu=mod_menu)
    menu_bar.add_cascade(label=lang["filtrer"], menu=filter_menu)
    menu_bar.add_cascade(label=lang["main_viz"], menu=graphs_menu)
    menu_bar.add_cascade(label=lang["power_estimate"], menu=power_menu)
    menu_bar.add_cascade(label=lang["freq_estimate"], menu=freq_menu)
    menu_bar.add_cascade(label=lang["phase_estimate"], menu=diff_menu)
    menu_bar.add_cascade(label=lang["speed_estimate"], menu=speed_menu)
    menu_bar.add_cascade(label=lang["cyclo_estimate"], menu=acf_menu)
    menu_bar.add_cascade(label=lang["ofdm"], menu=ofdm_menu)
    menu_bar.add_cascade(label=lang["demod"], menu=demod_menu)
    if with_sound:
        menu_bar.add_cascade(label="Audio", menu=audio_menu)
    # Redéfinit les labels des boutons
    load_button.config(text=lang["load"])
    close_button.config(text=lang["close"])
    mode_button.config(text=lang["cursors_off"])
    clear_button.config(text=lang["clear_cursors"])
    info_label.config(text=lang["load_msg"])
    peak_button.config(text=lang["peak_find"])
    # Fonctions des menus cascade
    # Graphes : spectrogramme, STFT, 3D
    graphs_menu.add_command(label=lang["group_spec"], command=plot_initial_graphs)
    graphs_menu.add_command(label=lang["group_const"], command=plot_other_graphs)
    graphs_menu.add_command(label=lang["spec_3d"], command=plot_3d_spectrogram)
    graphs_menu.add_command(label=lang["spectrogram"], command=stft_solo)
    # Quelques infos sur le signal & résultats automatiques
    info_menu.add_command(label=lang["frq_info"], command=display_frq_info)
    # Submenu params graphes
    param_submenu = tk.Menu(info_menu,tearoff=0)
    param_submenu.add_command(label=lang["param_spectre_persistance"], command=param_spectre_persistance)
    param_submenu.add_command(label=lang["param_hist_bins"],command=set_hist_bins)
    param_submenu.add_command(label=lang["advanced_settings"], command=advanced_settings)
    info_menu.add_cascade(label=lang["params"], menu=param_submenu)
    # Changer langue. Label "English" si langue = fr, "Français" si langue = en
    lang_switch = "Switch language: English" if lang['lang'] == "Français" else "Changer langue: Français"
    info_menu.add_command(label=lang_switch, command=change_lang)
    # Modifications du signal
    # Submenu FC
    move_submenu = tk.Menu(mod_menu,tearoff=0)
    move_submenu.add_command(label=lang["move_frq"], command=move_frequency)
    move_submenu.add_command(label=lang["move_freq_cursors"], command=move_frequency_cursors)
    move_submenu.add_command(label=lang["auto_center_coarse"],command=center_signal_coarse)
    move_submenu.add_command(label=lang["auto_center_fine"],command=center_signal_fine)
    move_submenu.add_command(label=lang["doppler"], command=apply_doppler_correction)
    mod_menu.add_cascade(label=lang["center_frq"],menu=move_submenu)
    # Echantillonnage
    sample_submenu = tk.Menu(mod_menu,tearoff=0)
    sample_submenu.add_command(label=lang["downsample"], command=downsample_signal)
    sample_submenu.add_command(label=lang["upsample"], command=upsample_signal)
    sample_submenu.add_command(label=lang["resample_poly"], command=polyphase_resample)
    mod_menu.add_cascade(label=lang["resample"],menu=sample_submenu)
    # Découpage : réduire durée
    cut_submenu = tk.Menu(mod_menu,tearoff=0)
    cut_submenu.add_command(label=lang["cut_val"], command=cut_signal)
    cut_submenu.add_command(label=lang["cut_cursors"], command=cut_signal_cursors)
    mod_menu.add_cascade(label=lang["cut_signal"],menu=cut_submenu)
    # Submenu fenêtrage
    window_submenu = tk.Menu(mod_menu,tearoff=0)
    window_submenu.add_command(label=lang["fft_size"], command=define_N)
    window_submenu.add_command(label=lang["set_window"], command=set_window)
    window_submenu.add_command(label=lang["set_overlap"], command=set_overlap)
    mod_menu.add_cascade(label=lang["window_options"], menu=window_submenu)
    # Lissage
    mod_menu.add_command(label=lang["param_phase_freq"], command=set_diff_params)
    # Enregistrer nouveau wav
    mod_menu.add_command(label=lang["save_wav"], command=save_as_wav)
    # Menu Filtre
    filter_menu.add_command(label=lang["filter_high_low"], command=apply_filter_high_low)
    filter_menu.add_command(label=lang["filter_band"], command=apply_filter_band)
    filter_menu.add_command(label=lang["mean"], command=mean_filter)
    filter_menu.add_command(label=lang["median_filter"], command=apply_median_filter)
    filter_menu.add_command(label=lang["moving_average"], command=apply_moving_average)
    filter_menu.add_command(label=lang["wiener_filter"], command=apply_wiener_filter)
    filter_menu.add_command(label=lang["fir_filter"], command=apply_fir_filter)
    filter_menu.add_command(label=lang["matched_filter"], command=apply_matched_filter)
    # Analyse du signal
    # Puissance
    power_menu.add_command(label=lang["dsp"], command=dsp)
    power_menu.add_command(label=lang["dsp_max"], command=dsp_max)
    power_menu.add_command(label=lang["time_amp"], command=time_amplitude)
    # Phase
    diff_menu.add_command(label=lang["constellation"], command=constellation)
    diff_menu.add_command(label=lang["distrib_phase"], command=phase_cumulative)
    diff_menu.add_command(label=lang["diff_phase"], command=phase_difference)
    diff_menu.add_command(label=lang["eye_diagram"], command=eye_diagram)
    # Frequence
    freq_menu.add_command(label=lang["diff_freq"], command=freq_difference)
    freq_menu.add_command(label=lang["distrib_freq"], command=frequency_cumulative)
    freq_menu.add_command(label=lang["persist_spectrum"], command=spectre_persistance)
    freq_menu.add_command(label=lang["scalogram"], command=morlet_wavelet)
    # Rapidité de modulation
    speed_menu.add_command(label=lang["envelope_spectrum"], command=envelope_spectrum)
    speed_menu.add_command(label=lang["psf"], command=psf)
    speed_menu.add_command(label=lang["mts"], command=mts)
    speed_menu.add_command(label=lang["pseries"], command=pseries)
    speed_menu.add_command(label=lang["cyclospectrum"], command=cyclospectrum)
    # ACF
    acf_menu.add_command(label=lang["autocorr"], command=autocorr)
    acf_menu.add_command(label=lang["autocorr_full"], command=autocorr_full)
    acf_menu.add_command(label=lang["scf"], command=scf)
    # OFDM
    ofdm_menu.add_command(label=lang["ofdm_symbol"], command=alpha_from_symbol)
    ofdm_menu.add_command(label=lang["ofdm_results"], command=ofdm_results)
    # Demod
    demod_menu.add_command(label=lang["demod_cpm_psk"], command=demod_cpm_psk)
    demod_menu.add_command(label=lang["demod_fm"], command=demod_fm)
    demod_menu.add_command(label=lang["demod_am"], command=demod_am)
    demod_menu.add_command(label=lang["demod_mfsk"], command=demod_mfsk)
    # Audio
    if with_sound:
        audio_menu.add_command(label=lang["audio"], command=audio_output)
    # Decod
    # decod_menu.add_command(label=lang[])


# Label d'infos au départ
info_label = tk.Label(root, text=lang["load_msg"])
info_label.pack()

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, padx=5, pady=5)

# Charge fichier
load_button = tk.Button(button_frame, text=lang["load"], command=load_wav)
load_button.pack(side=tk.LEFT)
close_button = tk.Button(button_frame, text=lang["close"], command=close_wav)
close_button.pack(side=tk.LEFT, padx=5)
shift_button = tk.Button(button_frame, text=lang["shift_frq"], command=shift_frequency)
shift_button.pack(side=tk.LEFT, padx=5)
load_mono_button = tk.Button(button_frame, text=lang["mono_real"], command=load_real)
# Active/désactive curseurs
mode_button = tk.Button(button_frame, text=lang["cursors_off"], command=toggle_cursor_mode)
mode_button.pack(side=tk.RIGHT)
clear_button = tk.Button(button_frame, text=lang["clear_cursors"], command=clear_cursors)
clear_button.pack(side=tk.RIGHT, padx=5)
peak_button = tk.Button(button_frame, text=lang["peak_find"], command=move_cursor_to_nearest_peak)
peak_button.pack(side=tk.RIGHT, padx=5)

# Création des menus dans la langue choisie
load_lang_changes()

def on_close():
    root.quit()  # Stop mainloop
    root.destroy()  # Détruit la fenêtre, libère les ressources

if drag_drop is True :
    root.drop_target_register(DND_FILES)
    root.dnd_bind("<<Drop>>", on_file_drop)

# Menu
root.config(menu=menu_bar)
# Config pour le resizing
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
plot_frame.pack(fill=tk.BOTH, expand=True)

root.protocol("WM_DELETE_WINDOW", on_close)

# Affichage du titre ascii si console activee
print(r"   _____ _                   _______    ")
print(r"  / ____(_)         /\      |__   __|   ")
print(r" | (___  _  __ _   /  \   _ __ | | ___  ")
print(r"  \___ \| |/ _` | / /\ \ | '_ \| |/ _ \ ")
print(r"  ____) | | (_| |/ ____ \| | | | | (_) |")
print(r" |_____/|_|\__, /_/    \_\_| |_|_|\___/ ")
print(r"            __/ |                       ")
print(r"           |___/                        ") 

print("Application démarrée")
print(lang["load_msg"])

# A l'ouverture, affiche le logo
fig = plt.figure()
ax = plt.subplot()
ax.text(0.5, 0.5, "SigAnTo", ha='center', va='center', fontsize=50, color='black', font='Brush Script MT')
ax.axis('off')
canvas = FigureCanvasTkAgg(fig, plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Lancement de la boucle principale de l'interface graphique
# Détruit la fenêtre de chargement et affiche la principale
loading_end.wait()
loading_screen.destroy()
root.deiconify()
root.focus_force()
root.mainloop()

# Fin
