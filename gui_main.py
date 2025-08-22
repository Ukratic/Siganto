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
frame_rate = None
iq_wave = None
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

# Frame pour les graphes
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Fonctions de chargement de fichier WAV
def find_sample_width(file_path):
    # Fonc de détermination d'encodage WAV
    with open(file_path,'rb') as wav_file:
        header = wav_file.read(44) # Premiers 44 bytes = réservés header
        if header[:4] != b'RIFF' or header[8:12] != b'WAVE' or header[12:16] != b'fmt ': # vérifie si c'est un fichier WAV
            tk.messagebox.showerror(lang["error"], lang["invalid_wav"])
            raise ValueError(lang["invalid_wav"])
        bits_per_sample  = struct.unpack('<H', header[34:36])[0]
        audio_format = struct.unpack('<H', header[20:22])[0]
        if bits_per_sample not in [8, 16, 24, 32, 64]:
            tk.messagebox.showerror(lang["error"], lang["unsupported_bits"])
            raise ValueError(lang["unsupported_bits"])
        if audio_format not in [1, 3]:  # PCM ou IEEE float
            tk.messagebox.showerror(lang["error"], lang["unsupported_format"])
            raise ValueError(lang["unsupported_format"])
        audio_format = "PCM" if audio_format == 1 else "IEEE float"
    return bits_per_sample , audio_format

def load_wav():
    global filepath, frame_rate, iq_wave, N, overlap, corr, convert_button
    if filepath is None:
        filepath = tk.filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not filepath:
        return

    frame_rate, s_wave = wav.read(filepath)

    try:
        if s_wave.ndim == 2 and s_wave.shape[1] == 2:
            # Fichier stéréo → on suppose I/Q sur 2 canaux
            left = s_wave[:, 0].astype(np.float32)
            right = s_wave[:, 1].astype(np.float32)
            # retire quelques échantillons pour éviter les erreurs si nécessaire
            min_len = min(len(left), len(right))
            iq_wave = left[:min_len] + 1j * right[:min_len]
            # Vérification de corrélation entre les canaux gauche et droit
            corr = np.corrcoef(left, right)[0,1]
            if abs(corr) > 0.99:
                if debug is True:
                    print("Attention: canaux droite & gauche identiques. On force en mono.")
                mono_signal = left
                analytic_signal = sm.hilbert(mono_signal)
                iq_wave = analytic_signal
            else:
                iq_wave = left + 1j*right
        elif s_wave.ndim == 1:
            # Mono → 2 hypothèses
            if not convert_button:
                load_mono_button.pack(side=tk.LEFT, padx=5)
                convert_button = True
            if mono_real is True:
                iq_wave = sm.hilbert(s_wave)
                iq_wave = iq_wave * np.exp(-1j*2*np.pi*(frame_rate//4)*np.arange(len(iq_wave))/frame_rate)
                iq_wave, frame_rate = sm.downsample(iq_wave, frame_rate, 2)
            else:
                left = s_wave[0::2].astype(np.float32)
                right = s_wave[1::2].astype(np.float32)
                min_len = min(len(left), len(right))
                iq_wave = left[:min_len] + 1j * right[:min_len]

        else:
            raise ValueError("Format WAV inattendu")
    except:
        if debug is True:
            print("Erreur de conversion IQ")
        tk.messagebox.showerror(lang["error"], lang["wav_conversion"])
    if len(iq_wave) > 1e6: # si plus d'un million d'échantillons
        N = 4096
    elif 1e5 < len(iq_wave) < 1e6 :
        N = 1024
    elif len(iq_wave) < frame_rate: # si moins d'une seconde
        N = (frame_rate//25)*(len(iq_wave)/frame_rate) # base de résolution = 25 Hz par défaut, proportionnellement à la durée si inférieur à 1 seconde
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
    global filepath, iq_wave
    filepath = None
    iq_wave = None
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
    freqs, times, stft_matrix = mg.compute_stft(iq_wave, frame_rate, window_size=N, overlap=overlap, window_func=window_choice)
    if freqs is None:
        print(lang["error_stft"])
        # message d'erreur si la STFT n'a pas pu être calculée
        tk.messagebox.showerror(lang["error"], lang["error_stft"])
        return
    ax[0].imshow(stft_matrix, aspect='auto', extent = [frame_rate/-2, frame_rate/2, len(iq_wave)/frame_rate, 0], cmap='jet')
    ax[0].set_ylabel(f"{lang['time_xy']} [s]")
    ax[0].set_title(f"{lang['window']} {window_choice}")

    # DSP
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_wave, frame_rate, N, overlap, window_choice)
    Pxx_shifted = np.fft.fftshift(Pxx) 
    f_shifted = np.fft.fftshift(f)
    ax[1].plot(f_shifted, Pxx_shifted)
    ax[1].set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax[1].set_ylabel("Amplitude")
    ax[1].axvline(x=fmax, color='r', linestyle='--')
    ax[1].axvline(x=fmin, color='r', linestyle='--')
    if debug is True:
        print("Bande passante estimée: ", bw, " Hz")
        print("Fréquence max BW: ", fmax, " Hz")
        print("Fréquence min BW: ", fmin, " Hz")

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
    global toolbar, ax, fig, canvas, iq_wave, original_iq_wave, fcenter, freq_label
    # Figure avec 3 sous-graphes. Le premier est sur deux lignes, les deux autres se partagent la 3eme ligne
    original_iq_wave = iq_wave.copy() # copie du signal original pour les modifications
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
    freqs, times, stft_matrix = mg.compute_stft(iq_wave, frame_rate, window_size=N, overlap=overlap, window_func=window_choice)
    stft = ax[0].imshow(stft_matrix, aspect='auto', extent=[frame_rate / -2, frame_rate / 2, len(iq_wave) / frame_rate, 0], cmap=cm.jet)
    ax[0].set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax[0].set_ylabel(f"{lang['time_xy']} [s]")
    ax[0].set_title(f"{lang['window']} {window_choice}")

    # Constellation
    line_constellation = ax[1].scatter(np.real(iq_wave), np.imag(iq_wave), s=1)
    ax[1].set_xlabel("In-Phase")
    ax[1].set_ylabel("Quadrature")

    # DSP avec max
    wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_wave)))**2
    f = np.linspace(frame_rate / -2, frame_rate / 2, len(iq_wave)) # freq en Hz
    line_spectrum, = ax[2].plot(f, wav_mag)
    ax[2].plot(f[np.argmax(wav_mag)], np.max(wav_mag), 'rx') # point max
    ax[2].grid()
    ax[2].set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax[2].set_ylabel("Amplitude")

    # Label pour afficher l'offset FC
    freq_label = tk.Label(plot_frame, text=f"{lang['offset_freq']}: {fcenter} Hz")
    freq_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_graph():
        global iq_wave, fcenter
        # Recompute iq_wave avec l'offset de fréquence à partir de l'iq_wave original
        iq_wave = original_iq_wave * np.exp(-1j * 2 * np.pi * fcenter * np.arange(len(original_iq_wave)) / frame_rate)
        # Màj label
        freq_label.config(text=f"{lang['offset_freq']}: {fcenter} Hz")
        # Màj STFT
        freqs, times, stft_matrix = mg.compute_stft(iq_wave, frame_rate, window_size=N, overlap=overlap, window_func=window_choice)
        stft.set_data(stft_matrix)
        # Màj constellation
        line_constellation.set_offsets(np.c_[np.real(iq_wave), np.imag(iq_wave)])
        # Màj DSP
        wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_wave)))**2
        line_spectrum.set_ydata(wav_mag)

        canvas.draw()

    def move_left(event):
        global fcenter
        fcenter -= 1  # Freq - 1 Hz
        update_graph()

    def move_right(event):
        global fcenter
        fcenter += 1  # Freq + 1 Hz
        update_graph()

    # Flèches du clavier pour déplacer la fréquence centrale
    root.bind("<Left>", move_left)
    root.bind("<Right>", move_right)
    # Sinon boutons sur l'interface
    button_frame = tk.Frame(plot_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)
    left_button = tk.Button(button_frame, text="←", command=move_left)
    left_button.pack(side=tk.LEFT, padx=10)
    right_button = tk.Button(button_frame, text="→", command=move_right)
    right_button.pack(side=tk.RIGHT, padx=10)
    
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del spec, a0, a1, a2, f, stft_matrix, freqs, times, wav_mag

# Fonc de changement de fenêtre FFT pour STFT
def set_window():
    global window_choice
    #dropdown pour choisir la fenêtre
    popup = tk.Toplevel()
    popup.title(lang["window_choice"])
    popup.geometry("300x300")
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
    freqs, times, spectrogram = mg.compute_spectrogram(iq_wave, frame_rate, N, window_func=window_choice)
    X, Y = np.meshgrid(freqs, times)
    ax = plt.subplot(projection='3d')
    ax.plot_surface(X, Y, spectrogram, cmap=cm.coolwarm)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(f"{lang['time_xy']} [s]")
    ax.set_zlabel("Amplitude")
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
    time = np.arange(len(iq_wave)) / frame_rate
    ax.plot(time, iq_wave)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas, time

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
    f, min_power, max_power, persistence = em.persistance_spectrum(iq_wave, frame_rate, N, persistance_bins)
    ax.imshow(persistence.T, aspect='auto', extent=[f[0], f[-1], min_power, max_power], origin='lower', cmap='jet')
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel("Amplitude")
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
        tk.messagebox.showinfo(lang["error"], lang["overlap_valid"])
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
    freqs, times, stft_matrix = mg.compute_stft(iq_wave, frame_rate, window_size=N, overlap=overlap, window_func=window_choice)
    if freqs is None:
        print(lang["error_stft"])
        # message d'erreur si la STFT n'a pas pu être calculée
        tk.messagebox.showerror(lang["error"], lang["error_stft"])
        return
    ax.imshow(stft_matrix, aspect='auto', extent = [frame_rate/-2, frame_rate/2, len(iq_wave)/frame_rate, 0],cmap=cm.jet)
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
    if iq_wave is None :
        if debug is True:
            print("Fichier non chargé")
        return
    wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_wave)))**2
    f = np.linspace(frame_rate/-2, frame_rate/2, len(iq_wave))
    f_pmax = f[np.argmax(wav_mag)]
    f_pmin = f[np.argmin(wav_mag)] 
    max_lvl = 10*np.log10(np.max(np.abs(iq_wave)**2))
    low_lvl = 10*np.log10(np.min(np.abs(iq_wave)**2))
    mean_lvl = 10*np.log10(np.mean(np.abs(iq_wave)**2))
    estim_speed_2 = round(np.abs(em.mean_threshold_spectrum(iq_wave, frame_rate)[2]),2)
    estim_speed = round(np.abs(em.power_spectrum_fft(iq_wave, frame_rate)[2]),2)
    _,_,_, peak_squared_freq, peak_quartic_freq = em.power_series(iq_wave, frame_rate)
    estim_speed_3 = [round(abs(peak_squared_freq),2),round(abs(peak_quartic_freq),2)]
    _, freq_diff = em.frequency_transitions(iq_wave, frame_rate, diff_window)
    freq_diff /= np.max(np.abs(freq_diff))
    estim_speed_4 = round(float(dm.estimate_baud_rate(freq_diff, frame_rate)),2)
    acf_peak = round(np.abs(em.autocorrelation_peak(iq_wave, frame_rate, min_distance=25)[1]),2)

    popup = tk.Toplevel()
    popup.title(lang["frq_info"])
    popup.geometry("600x200")
    tk.Label(popup, text=f"{lang['high_freq']}: {f_pmax:.2f} Hz. {lang['low_freq']}: {f_pmin:.2f} Hz\n \
                    {lang['high_level']} {max_lvl:.2f}. {lang['low_level']} {low_lvl:.2f}.\n \
                    {lang['mean_level']} {mean_lvl:.2f} dB ").pack()
    tk.Label(popup, text=f"{lang['estim_bw']} {round(bw,2)} Hz").pack()
    tk.Label(popup, text=f"{lang['estim_speed']} {estim_speed} Bds\n \
                    {lang['estim_speed_2']} {estim_speed_2} Bds\n \
                    {lang['estim_speed_3']} {estim_speed_3[0]*2} / {estim_speed_3[1]*2} Bds\n \
                    {lang['estim_speed_4']} {estim_speed_4} Bds").pack()
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
        print("Autocorrélation estimée: ", acf_peak, " ms")
    del _,estim_speed,estim_speed_2, wav_mag

# Affichage des informations du fichier : nom, encodage, durée, fréquence d'échantillonnage, taille de la fenêtre FFT
def display_file_info():
    if filepath is None:
        info_label.config(text=lang["no_file"])
        return
    info_label.config(text=f"{filepath}. {lang['encoding']} {find_sample_width(filepath)[0]} bits. Format {find_sample_width(filepath)[1]}. \
                      \n{lang['samples']} {len(iq_wave)}. {lang['sampling_frq']} {frame_rate} Hz. {lang['duree']}: {len(iq_wave)/frame_rate:.2f} sec.\
                      \n {lang['fft_window']} {N}. Overlap : {overlap}. {lang['f_resol']} {frame_rate/N:.2f} Hz.")
    if debug is True:
        print("Affichage des informations du fichier")
        print("Chargé: ", filepath)
        print("Encodage: ", find_sample_width(filepath)[0], " bits")
        print("Format: ", find_sample_width(filepath)[1])
        print("Echantillons: ", len(iq_wave))
        print("Fréquence d'échantillonnage: ", frame_rate, " Hz")
        print("Durée: ", len(iq_wave)/frame_rate, " secondes")
        print("Taille fenêtre FFT: ", N)
        print("Recouvrement: ", overlap)

# Fonctions de traitement du signal (filtres, déplacement de fréquence, sous-échantillonnage, sur-échantillonnage, coupure)
def move_frequency():
    # déplacement de la fréquence centrale (valeur entrée par l'utilisateur)
    global iq_wave, frame_rate
    fcenter = float(tk.simpledialog.askstring(lang["fc"], lang["move_txt"], parent=root))
    if fcenter is None:
        if debug is True:
            print("Modification de fréquence centrale annulée, valeur non définie")
        return
    iq_wave = iq_wave * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_wave))/frame_rate)
    if debug is True:
        print("Fréquence centrale déplacée de ", fcenter, " Hz")
    plot_initial_graphs()

def move_frequency_cursors():
    # déplacement de la fréquence centrale sur le curseur (inactif si 2 curseurs)
    global cursor_points, iq_wave, frame_rate
    if len(cursor_points) != 1 and (cursor_points[0][0] != cursor_points[1][0]):
        tk.messagebox.showinfo(lang["error"], lang["1pt_cursors"])
        return
    fcenter = cursor_points[0][0]
    iq_wave = iq_wave * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_wave))/frame_rate)
    if debug is True:
        print("Fréquence centrale déplacée de ", fcenter, " Hz")
    plot_initial_graphs()

def apply_filter_high_low():
    # passage d'un filtre passe-haut ou passe-bas
    global iq_wave, frame_rate
    popup = tk.Toplevel()
    popup.title(lang["high_low"])
    popup.geometry("300x200")
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
        iq_wave = sm.lowpass_filter(iq_wave, float(cutoff.get()), frame_rate)
    elif filter_type.get() == lang["high_val"]:
        iq_wave = sm.highpass_filter(iq_wave, float(cutoff.get()), frame_rate)
    if debug is True:
        print("Filtre passe-", filter_type.get(), " appliqué. Fréquence de coupure: ", cutoff.get(), "Hz")
    plot_initial_graphs()

def apply_filter_band():
    # passage d'un filtre passe-bande
    global iq_wave, frame_rate
    popup = tk.Toplevel()
    popup.title(lang["bandpass"])
    popup.geometry("300x200")
    lowcut = tk.StringVar()
    highcut = tk.StringVar()
    tk.Label(popup, text=lang["freq_low"]).pack()
    tk.Entry(popup, textvariable=lowcut).pack()
    tk.Label(popup, text=lang["freq_high"]).pack()
    tk.Entry(popup, textvariable=highcut).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if (lowcut.get() == "" and highcut.get() != "") or (lowcut.get() != "" and highcut.get() == ""):
        tk.messagebox.showinfo(lang["error"], lang["freq_valid"])
        return
    elif lowcut.get() == "" and highcut.get() == "":
        return
    else:
        iq_wave = sm.bandpass_filter(iq_wave, float(lowcut.get()), float(highcut.get()), frame_rate)
    if debug is True:
        print("Filtre passe-bande appliqué. Fréquence de coupure basse: ", lowcut.get(), "Hz. Fréquence de coupure haute: ", highcut.get(), "Hz")
    plot_initial_graphs()

def mean_filter():
    # filtre moyenneur
    global iq_wave
    # popup pour choisir entre appliquer ou définir le seuil
    popup = tk.Toplevel()
    popup.title(lang["mean"])
    popup.geometry("300x150")
    mean_filter = tk.StringVar()
    mean_filter.set(lang["not_apply"])
    # Afficher sur la popup la valeur de la variable
    iq_floor = 10*np.log10(np.mean(np.abs(iq_wave)**2))
    tk.Label(popup, text=lang["mean_level"] + str(iq_floor)).pack()
    tk.Radiobutton(popup, text=lang["not_apply"], variable=mean_filter, value=lang["not_apply"]).pack()
    tk.Radiobutton(popup, text=lang["apply_mean"], variable=mean_filter, value=lang["apply_mean"]).pack()
    tk.Radiobutton(popup, text=lang["def_level"], variable=mean_filter, value=lang["def_level"]).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if mean_filter.get() == lang["def_level"]:
        iq_floor = float(tk.simpledialog.askstring(lang["level"], lang["enter_level"], parent=root))
        iq_wave = np.where(10*np.log10(np.abs(iq_wave)**2) < iq_floor, 0, iq_wave)
        if iq_floor is None:
            if debug is True:
                print("Seuil de moyennage non défini")
            return
        if debug is True:
            print("Signal moyenné avec un seuil de ", iq_floor, " dB")
    elif mean_filter.get() == lang["apply_mean"]:
        iq_wave = np.where(10*np.log10(np.abs(iq_wave)**2) < iq_floor, 0, iq_wave)
        if debug is True:
            print("Signal moyenné avec un seuil de ", iq_floor, " dB")
    else:
        return
    
    print(lang["mean"])
    plot_initial_graphs()

def downsample_signal():
    # sous-échantillonnage
    global iq_wave, frame_rate
    rate = tk.simpledialog.askstring(lang["downsample"], lang["down_value"], parent=root)
    if rate is None:
        if debug is True:
            print("Taux de sous-échantillonnage non défini")
        return
    decimation_factor = int(rate)
    iq_wave, frame_rate = sm.downsample(iq_wave, frame_rate, decimation_factor)
    print(lang["sampling_frq"], frame_rate, "Hz")
    plot_initial_graphs()
    display_file_info()

def upsample_signal():
    # sur-échantillonnage
    global iq_wave, frame_rate
    rate = tk.simpledialog.askstring(lang["upsample"], lang["up_value"], parent=root)
    if rate is None:
        if debug is True:
            print("Taux de sur-échantillonnage non défini")
        return
    oversampling_factor = int(rate)
    iq_wave, frame_rate = sm.upsample(iq_wave, frame_rate, oversampling_factor)
    print(lang["sampling_frq"], frame_rate, "Hz")
    plot_initial_graphs()
    display_file_info()

def cut_signal():
    # coupure du signal : entrer les points de début et de fin (en secondes)
    global iq_wave, frame_rate
    popup = tk.Toplevel()
    popup.title(lang["cut_val"])
    popup.geometry("300x200")
    start = tk.StringVar()
    end = tk.StringVar()
    tk.Label(popup, text=lang["start_cut"]).pack()
    tk.Entry(popup, textvariable=start).pack()
    tk.Label(popup, text=lang["end_cut"]).pack()
    tk.Entry(popup, textvariable=end).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if start.get() =="" and end.get() =="":
        # tk.messagebox.showinfo(lang["error"], lang["valid_cut"])
        return
    if start.get() =="":
        start = 0
    else:
        start = int(float(start.get())*frame_rate)
    if end.get() =="":
        end = int((len(iq_wave)/frame_rate)*frame_rate)
    else:
        end = int(float(end.get())*frame_rate)
    iq_wave = iq_wave[start:end]
    if debug is True:
        print("Signal coupé de ", start, "échantillons à ", end, "échantillons, soit ", end-start, "échantillons restants")
        print("Nouvelle durée du signal : ", len(iq_wave)/frame_rate, " secondes")
    plot_initial_graphs()
    display_file_info()

def cut_signal_cursors():
    # coupure du signal entre les 2 curseurs (ne prend en compte que la durée, pas l'écart en fréquence)
    global iq_wave, frame_rate, cursor_points
    if len(cursor_points) < 2:
        tk.messagebox.showinfo(lang["error"], lang["2pt_cursors"])
        return
    # signal coupé entre les 2 points. On ne sait pas quel point est le début et lequel est la fin, donc on prend les valeurs y les plus petites et les plus grandes
    start = int(cursor_points[0][1]*frame_rate)
    end = int(cursor_points[1][1]*frame_rate)
    print(cursor_points[1][1],cursor_points[0][1])   
    if cursor_points[1][1] < cursor_points[0][1]:
        iq_wave = iq_wave[end:start]
        if debug is True:
            print("Signal coupé de ", end, "échantillons à ", start, "échantillons, soit ", start-end, "échantillons restants")
            print("Nouvelle durée du signal : ", len(iq_wave)/frame_rate, " secondes")
    else:
        iq_wave = iq_wave[start:end]
        if debug is True:
            print("Signal coupé de ", start, "échantillons à ", end, "échantillons, soit ", end-start, "échantillons restants")
            print("Nouvelle durée du signal : ", len(iq_wave)/frame_rate, " secondes")
    plot_initial_graphs()
    display_file_info()

def center_signal():
    global iq_wave
    if not filepath:
        print(lang["no_file"])
        return
    # param de proeminence à ajuster
    iq_wave, center = df.center_signal(iq_wave, frame_rate, prominence=0.1)
    if debug is True:
        print(f"Signal centré par déplacement de {round(center, 2)} Hz.")

    plot_initial_graphs()

# Fonctions de mesure de la rapidité de modulation
def psf():
    # Mesure de la rapidité de modulation par la FFT de puissance. Polyvalente.
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["psf"])
    print("Génération de la FFT de puissance")
    if not filepath:
        print(lang["no_file"])
        return
    clock, f, peak_freq = em.power_spectrum_fft(iq_wave, frame_rate)
    ax = plt.subplot()
    ax.plot(f,np.abs(clock))
    if abs(peak_freq) > 25:
        ax.axvline(x=-peak_freq, color='r', linestyle='--')
        ax.axvline(x=peak_freq, color='r', linestyle='--')
        ax.set_title(f"{lang['estim_peak']} {round(peak_freq,2)} Hz")
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["mag"])

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
    clock, f, peak_freq = em.mean_threshold_spectrum(iq_wave, frame_rate)
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
    ax.set_ylabel(lang["mag"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del clock, f, peak_freq, canvas

def pseries():
    # mesure de la rapidité de modulation par la série de puissance, efficace sur les signaux de modulation d'amplitude et de fréquence
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
    f, squared, quartic, peak_squared_freq, peak_quartic_freq = em.power_series(iq_wave, frame_rate)
    ax[0].plot(f, squared)
    ax[0].set_ylabel(f"{lang['mag']} ^2")
    if abs(peak_squared_freq) > 25:
        ax[0].axvline(x=-peak_squared_freq, color='r', linestyle='--')
        ax[0].axvline(x=peak_squared_freq, color='r', linestyle='--')
        ax[0].set_title(f"{lang['estim_peak']} {round(peak_squared_freq,2)} Hz")
        if debug is True:
            print("Ecart estimé : ", round(np.abs(peak_squared_freq),2))
    ax[1].plot(f, quartic)
    ax[1].set_ylabel(f"{lang['mag']} ^4")
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
    f, cyclic_corr_avg, peak_freq = em.cyclic_spectrum_sliding_fft(iq_wave, frame_rate, window=window_choice, frame_len=N, step=overlap)
    f = np.linspace(frame_rate/-2, frame_rate/2, len(cyclic_corr_avg))
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
    ax.set_ylabel(lang["mag"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del cyclic_corr_avg, f, peak_freq, canvas

def dsp():
    # affichage de la densité spectrale de puissance et de la bande passante estimée
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text, bw
    print(lang["dsp"])
    print(lang["bandwidth"])
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["dsp"])
    ax = fig.add_subplot()
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_wave, frame_rate, N,overlap,window_choice)
    line, = ax.plot(f, Pxx)
    # Suppression du segment le plus long qui affiche une ligne inutile
    # Cherche le plus long segment
    distances = np.sqrt(np.diff(f)**2 + np.diff(Pxx)**2)
    longest_segment_index = np.argmax(distances)
    # Coordonnées du segment le plus long
    x1, y1 = f[longest_segment_index], Pxx[longest_segment_index]
    x2, y2 = f[longest_segment_index + 1], Pxx[longest_segment_index + 1]
    # Retire le segment le plus long du graphe en le remplaçant par une ligne blanche
    ax.plot([x1, x2], [y1, y2], color="white", linewidth=3) # width = 3 pour être sûr de bien couvrir le segment même si la ligne est inclinée
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel("Amplitude")
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
    del fmax, fmin, canvas, f, Pxx, line, distances, longest_segment_index, x1, y1, x2, y2

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
    wav_mag = np.abs(np.fft.fftshift(np.fft.fft(iq_wave)))**2
    f = np.linspace(frame_rate/-2, frame_rate/2, len(iq_wave)) # frq en Hz
    ax.plot(f, wav_mag)
    ax.plot(f[np.argmax(wav_mag)], np.max(wav_mag), 'rx') # show max
    ax.grid()
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel("Amplitude")

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
    ax.scatter(np.real(iq_wave), np.imag(iq_wave), s=1)
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
    yx, lags = em.autocorrelation(iq_wave, frame_rate)
    ax.plot(lags*1e3, yx/np.max(yx)) # lags en ms
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel(lang["autocorr"])
    peak, time = em.autocorrelation_peak_from_acf(yx, lags, min_distance=25)
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
    if not tk.messagebox.askokcancel(lang["autocorr_full"], lang["confirm_wait"]):
        return
    ax = plt.subplot()
    # Params cyclospectre
    if debug is True:
        print("Peut générer des ralentissements. Patienter")
    yx, lags = em.full_autocorrelation(iq_wave)
    ax.plot(lags/frame_rate*1e3, yx/np.max(yx)) # lags en ms
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel(lang["autocorr"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del yx, lags, canvas

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
    time, phase_diff = em.phase_time_angle(iq_wave, frame_rate, diff_window)
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
    time, freq_diff = em.frequency_transitions(iq_wave, frame_rate, diff_window)
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
    hist, bins = em.phase_cumulative_distribution(iq_wave, num_bins=hist_bins)
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
    hist, bins = em.frequency_cumulative_distribution(iq_wave, frame_rate, num_bins=hist_bins)
    ax.plot(bins, hist)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["density"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del hist, bins, canvas

def phase_spectrum():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["phase_spectrum"])
    print(lang["phase_spectrum"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    ax.phase_spectrum(iq_wave, Fs=frame_rate)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["insta_phase"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas

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
    peak, alpha, caf = em.estimate_alpha(iq_wave, frame_rate, Tu)
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
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_wave, frame_rate, N,overlap,window_choice)
    # affiche BW estimée et demande de valider ou de redéfinir la bande passante
    popup = tk.Toplevel()
    popup.title(lang["bandwidth"])
    popup.geometry("600x100")
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
    popup.title(lang["ofdm_results"])
    popup.geometry("400x200")
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
    line, = ax.plot(f, Pxx)
    # Suppression du segment le plus long qui affiche une ligne inutile
    # Cherche le plus long segment
    distances = np.sqrt(np.diff(f)**2 + np.diff(Pxx)**2)
    longest_segment_index = np.argmax(distances)
    # Coordonnées du segment le plus long
    x1, y1 = f[longest_segment_index], Pxx[longest_segment_index]
    x2, y2 = f[longest_segment_index + 1], Pxx[longest_segment_index + 1]
    # Retire le segment le plus long du graphe en le remplaçant par une ligne blanche
    ax.plot([x1, x2], [y1, y2], color="white", linewidth=3) # width = 3 pour être sûr de bien couvrir le segment même si la ligne est inclinée
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel("Amplitude")
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
    del fmax, fmin, f, Pxx, alpha_0, popup, canvas, distances, longest_segment_index, x1, y1, x2, y2

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

# sauvegarde le signal (modifié ou non) en nouveau fichier wav
def save_as_wav():
    global iq_wave, frame_rate
    filename = tk.filedialog.asksaveasfilename(title=lang["save_wav"],defaultextension=".wav", filetypes=[("Waveform Audio File", "*.wav"), ("All Files", "*.*")])
    # Normalise les données IQ pour éviter le clipping
    max_amplitude = np.max(np.abs(iq_wave))
    if max_amplitude > 0:
        iq_data_normalized = iq_wave / max_amplitude
    else:
        iq_data_normalized = iq_wave
    # Formate en 16 bits et sépare reel/imaginaire
    if debug is True:
        print("Conversion en 2 voies 16 bits")
    real_part = (iq_data_normalized.real * 32767).astype(np.int16)
    imag_part = (iq_data_normalized.imag * 32767).astype(np.int16)
    # transforme reel+imag en 2 canaux pour le wav
    stereo_data = np.column_stack((real_part, imag_part))
    wav.write(filename, frame_rate, stereo_data)
    if debug is True:
        print("Ecriture du nouveau wav")

# Démodulation FSK 2 et 4
def demod_fsk():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # transitions de frq
    time, freq_diff = em.frequency_transitions(iq_wave, frame_rate, diff_window)
    freq_diff /= np.max(np.abs(freq_diff))
    # vars pour fonctions de démod
    target_rate = None
    precision = 0.9
    order = None
    mapping = None
    tau = np.pi * 2
    # on demande rapidité, ordre et mapping
    def toggle_mapping():
        # mapping seulement si ordre 4
        if param_order.get() == lang["param_order4"]:
            mapping_nat.config(state=tk.NORMAL)
            mapping_gray.config(state=tk.NORMAL)
            mapping_custom.config(state=tk.NORMAL)
            param_mapping.set(lang["mapping_nat"])
        else:
            mapping_nat.config(state=tk.DISABLED)
            mapping_gray.config(state=tk.DISABLED)
            mapping_custom.config(state=tk.DISABLED)
    popup = tk.Toplevel()
    popup.title(lang["demod_param"])
    popup.geometry("350x250")
    param_order = tk.StringVar()
    param_order.set(lang["param_order"])
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
    mapping_custom = tk.Radiobutton(popup, text=lang["mapping_custom"], variable=param_mapping, value=lang["mapping_custom"], state=tk.DISABLED)
    mapping_custom.pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    
    popup.wait_window()
    if target_rate.get() == "":
        if debug is True:
            print("Rapidité de démodulation non définie.")
        return
    elif float(target_rate.get()) < 1:
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
        # actuellement un string sous forme de "0,1,2,3" ou "00,01,10,11" pour 4fsk. A transformer en liste
        mapping = list(eval(mapping))

    # fonction de démod et slice bits en fonction de l'ordre
    try:
        symbols, clock = dm.wpcr(freq_diff, frame_rate, target_rate, tau, precision, debug)
        if len(symbols) > 2 and order == 2:
            bits=dm.slice_binary(symbols)
            if debug is True:
                print("Démodulation FSK 2 réalisée, bits: ", len(bits))
        elif len(symbols) > 2 and order == 4:
            bits = dm.slice_4fsk(symbols,mapping)
            if debug is True :
                print(f"Démodulation FSK {order} réalisée avec mapping {mapping}, rapidité {clock} bauds, bits: {len(bits)}") 
        else:
            bits=0
        # plot des bits demodulés
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
        text_output= f"{lang['estim_bits']} ({len(bits)}). {lang['clock_frequency']} {clock} Hz : \n"
        # lignes de bits
        formatted_bits = "".join(map(str, bits))
        text_output += formatted_bits
    except:
        bits,bits_plot,formatted_bits = "","",""
        text_output = lang["bits_fail"]
    text_box.insert(tk.END, text_output)
    text_box.config(state=tk.DISABLED)
    del canvas, time, freq_diff, bits, bits_plot, formatted_bits, text_output, text_box

def demod_fm():
    global iq_wave
    if not filepath:
        print(lang["no_file"])
        return
    # demod fm
    try:
        iq_wave = dm.fm_demodulate(iq_wave,frame_rate)
        if debug is True:
            print("Démodulation FM réalisée")
    except:
        if debug is True:
            print("Echec de démodulation FM")
            return
    plot_other_graphs()

def demod_am():
    global iq_wave
    if not filepath:
        print(lang["no_file"])
        return
    # demod am
    try:
        iq_wave = dm.am_demodulate(iq_wave)
        if debug is True:
            print("Démodulation AM réalisée")
    except:
        if debug is True:
            print("Echec de démodulation AM")
            return
    plot_other_graphs()

# def demod_psk():
#     global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
#     # vars & input
#     # mapping
#     # popups
#     # demod func
#     # graphe & bits output
#     #

# EXPERIMENTAL
def demod_mfsk():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    time, freq_diff = em.frequency_transitions(iq_wave, frame_rate, diff_window)
    freq_diff /= np.max(np.abs(freq_diff))
    # vars pour fonctions de démod
    target_rate = None
    precision = 0.9
    mapping = None
    return_format = "binary"  # par défaut, on renvoie les bits en binaire
    spacing = None # espacement entre les symboles. Pour l'instant pas utilisé
    tau = np.pi
    # on demande rapidité, ordre, espacement et mapping    
    popup = tk.Toplevel()
    popup.title(lang["demod_param"])
    # popup.geometry("350x250")
    param_method = tk.StringVar()
    tk.Radiobutton(popup, text=lang["mfsk_discrete_diff"], variable=param_method, value="main").pack()
    tk.Radiobutton(popup, text=lang["mfsk_tone_detection"], variable=param_method, value="alt").pack()
    param_method.set("main")
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
        clock = dm.estimate_baud_rate(freq_diff, frame_rate, target_rate, precision, debug)
    else:
        clock = float(target_rate.get())
    order = int(param_order.get())
    if param_mapping.get() == lang["mapping_nat"]:
        mapping = "natural"
    elif param_mapping.get() == lang["mapping_gray"]:
        mapping = "gray"
    elif param_mapping.get() == lang["mapping_non-binary"]:
        mapping = "non-binary"
        return_format = "char"

    try:
        if debug is True:
            print("Démodulation MFSK en cours...")
        if param_method.get() == "main":
            symbols, clock = dm.wpcr(freq_diff, frame_rate, clock, tau, precision, debug)
        elif param_method.get() == "alt":
        # EXPERIMENTAL
            tone_freqs, t, tone_idx, tone_freq, tone_powers, clock = dm.detect_and_track_mfsk_auto(iq_wave, frame_rate, clock, num_tones=order, peak_thresh_db=8, switch_penalty=0.05)       
            tone_freq /= np.max(np.abs(tone_freq))
            symbols, _ = dm.wpcr(tone_freq, frame_rate, target_rate=None, tau=tau, precision=precision, debug=debug)
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
        elif return_format == "char":
            charset = (string.digits + string.ascii_uppercase + string.ascii_lowercase + string.punctuation) # charset récupéré de la fonction demod.slice_mfsk
            mapping = {ch: i for i, ch in enumerate(charset)}
            bits_plot = np.array([mapping[ch] for ch in bits if ch in mapping])
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
        text_output = f"{lang['estim_bits']} ({len(bits)}). {lang['clock_frequency']} {clock} Hz : \n"
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
    global iq_wave, frame_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Demande la taille du filtre. 4 options (Léger pour 3, Modéré pour 5, Agressif pour 9, sinon personnalisé) avec liste dans la fenêtre
    popup = tk.Toplevel()
    popup.title(lang["median_filter"])
    popup.geometry("300x150")
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
            kernel_size = int(frame_rate / (symbol_rate * 2)) | 1
        else:
            # Si l'utilisateur a choisi une option non reconnue, on affiche un message d'erreur
            tk.messagebox.showerror(lang["error"], lang["kernel_invalid"])
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
        iq_wave = sm.median_filter(iq_wave, kernel_size)
        if debug is True:
            print(f"Filtre médian de taille {kernel_size} appliqué")
    except Exception as e:
        if debug:
            print(f"Erreur lors de l'application du filtre médian: {e}")
    plot_initial_graphs()

def apply_moving_average():
    global iq_wave, frame_rate
    if not filepath:
        print(lang["no_file"])
        return
    
    # Demande la taille du filtre. 4 options (Léger pour 3, Modéré pour 5, Agressif pour 9, sinon personnalisé) avec liste dans la fenêtre
    popup = tk.Toplevel()
    popup.title(lang["moving_average"])
    popup.geometry("300x150")
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
            window_size = int(frame_rate / (symbol_rate)) | 1
        else:
            # Si l'utilisateur a choisi une option non reconnue, on affiche un message d'erreur
            tk.messagebox.showerror(lang["error"], lang["window_invalid"])
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
        iq_wave = sm.moving_average(iq_wave, window_size)
        if debug is True:
            print(f"Moyenne mobile de taille {window_size} appliquée")
    except Exception as e:
        if debug:
            print(f"Erreur lors de l'application de la moyenne mobile: {e}")
    plot_initial_graphs()

def apply_fir_filter():
    global iq_wave, frame_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Demande la fréquence de coupure et le nombre de taps
    cutoff_freq = tk.simpledialog.askfloat(lang["fir_filter"], lang["freq_pass"], minvalue=1, maxvalue=frame_rate/2, parent=root)
    num_taps = tk.simpledialog.askinteger(lang["fir_filter"], lang["fir_taps"], minvalue=1, parent=root)
    if cutoff_freq is None or num_taps is None:
        return
    try:
        iq_wave = sm.fir_filter(iq_wave, frame_rate, cutoff_freq, 'lowpass', num_taps)
        if debug is True:
            print(f"Filtre FIR avec fréquence de coupure {cutoff_freq} Hz et {num_taps} taps appliqué")
    except Exception as e:
        print(f"Erreur lors de l'application du filtre FIR: {e}")
    plot_initial_graphs()

def apply_wiener_filter():
    global iq_wave, frame_rate
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
        iq_wave = sm.wiener_filter(iq_wave, size, noise)
        if debug is True:
            print(f"Filtre Wiener de taille {size} appliqué avec variance de bruit {noise}")
    except Exception as e:
        if debug:
            print(f"Erreur lors de l'application du filtre Wiener: {e}")
    plot_initial_graphs()

# Fonction pour appliquer un filtre adapté
def apply_matched_filter():
    global iq_wave, frame_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Popup pour sélectionner le filtre
    popup = tk.Toplevel()
    popup.title(lang["matched_filter"])
    popup.geometry("300x150")
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
        iq_wave = sm.matched_filter(iq_wave, frame_rate, symbol_rate, factor=factor, pulse_shape=pulse_shape)
        if debug:
            print(f"Filtre adapté appliqué: {pulse_shape}, facteur={factor}, rapidité={symbol_rate}")
    except Exception as e:
        tk.messagebox.showerror(lang["error"], f"{lang["error_matched_filter"]}: {e}")
        if debug:
            print(f"Erreur lors de l'application du filtre adapté : {e}")
    
    plot_initial_graphs()


def shift_frequency():
    global iq_wave, frame_rate
    # décale la fréquence centrale de la moitié de fréquence d'échantillonnage : prend en compte les fichiers qui ne sont pas IQ mais réels
    if not filepath:
        return
    iq_wave = iq_wave * ((-1) ** np.arange(len(iq_wave)))
    if debug is True:
        print("Décalage de fréquence Fe/2 appliqué")
    plot_initial_graphs()

def prepare_audio(sig):
    if sig.ndim > 1:
        sig = sig[:, 0]
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
        prepared_audio = prepare_audio(iq_wave) # suppose demodulation deja faite
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

        audio_stream = sd.OutputStream(callback=audio_callback, samplerate=frame_rate, channels=1, dtype='float32')
        audio_stream.start()

    except:
        return

def update_progress_bar():
    global stream_position, progress
    if is_playing and not is_paused and iq_wave is not None:
        progress_value = (stream_position / len(iq_wave)) * 100
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
    info_menu.add_cascade(label=lang["params"], menu=param_submenu)
    # Changer langue. Label "English" si langue = fr, "Français" si langue = en
    lang_switch = "Switch language: English" if lang['lang'] == "Français" else "Changer langue: Français"
    info_menu.add_command(label=lang_switch, command=change_lang)
    # Modifications du signal
    # Submenu FC
    move_submenu = tk.Menu(mod_menu,tearoff=0)
    move_submenu.add_command(label=lang["move_frq"], command=move_frequency)
    move_submenu.add_command(label=lang["move_freq_cursors"], command=move_frequency_cursors)
    move_submenu.add_command(label=lang["auto_center"],command=center_signal)
    mod_menu.add_cascade(label=lang["center_frq"],menu=move_submenu)
    # Echantillonnage
    sample_submenu = tk.Menu(mod_menu,tearoff=0)
    sample_submenu.add_command(label=lang["downsample"], command=downsample_signal)
    sample_submenu.add_command(label=lang["upsample"], command=upsample_signal)
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
    power_menu.add_command(label=lang["psf"], command=psf)
    power_menu.add_command(label=lang["mts"], command=mts)
    power_menu.add_command(label=lang["pseries"], command=pseries)
    power_menu.add_command(label=lang["cyclospectrum"], command=cyclospectrum)
    power_menu.add_command(label=lang["dsp"], command=dsp)
    power_menu.add_command(label=lang["dsp_max"], command=dsp_max)
    power_menu.add_command(label=lang["time_amp"], command=time_amplitude)
    # Phase
    diff_menu.add_command(label=lang["constellation"], command=constellation)
    diff_menu.add_command(label=lang["phase_spectrum"], command=phase_spectrum)
    diff_menu.add_command(label=lang["distrib_phase"], command=phase_cumulative)
    diff_menu.add_command(label=lang["diff_phase"], command=phase_difference)
    # Frequence
    freq_menu.add_command(label=lang["diff_freq"], command=freq_difference)
    freq_menu.add_command(label=lang["distrib_freq"], command=frequency_cumulative)
    freq_menu.add_command(label=lang["persist_spectrum"], command=spectre_persistance)
    # ACF
    acf_menu.add_command(label=lang["autocorr"], command=autocorr)
    acf_menu.add_command(label=lang["autocorr_full"], command=autocorr_full)
    # OFDM
    ofdm_menu.add_command(label=lang["ofdm_symbol"], command=alpha_from_symbol)
    ofdm_menu.add_command(label=lang["ofdm_results"], command=ofdm_results)
    # Demod
    demod_menu.add_command(label=lang["demod_fsk"], command=demod_fsk)
    demod_menu.add_command(label=lang["demod_fm"], command=demod_fm)
    demod_menu.add_command(label=lang["demod_am"], command=demod_am)
    demod_menu.add_command(label=lang["demod_mfsk"], command=demod_mfsk)
    # demod_menu.add_command(label=lang["demod_psk"], command=demod_psk)
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
