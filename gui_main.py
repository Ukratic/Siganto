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

# Initialisation : chargement des librairies et de la langue dans une fonction pour pouvoir l'intégrer dans un thread
# Ce thread permet de charger les librairies en arrière-plan avec une fenêtre de chargement, en attendant que tout soit prêt
loading_end = Event()
def loading_libs():
    # Librairies
    print("Chargement des dépendances...")
    global struct, gc, FigureCanvasTkAgg, NavigationToolbar2Tk, plt, cm, np, wav, ll, em, lang, mg, sm, dm, scrolledtext
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
    import demod as dm
    # Langue, noms des fonctions, etc.
    lang = ll.get_fra_lib()
    # loading_end.wait() à la fin du script
    loading_end.set()
    
# Fenêtre de chargement
t = Thread(target=loading_libs, daemon=True)
t.start()
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
# root = tk.Tk()
root.title(lang["siganto"])
root.geometry("1024x768")
menu_bar = tk.Menu(root)
print("Chargement des fonctions...")
# Init des variables
filepath = None
frame_rate = None
iq_wave = None
bw = None
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
# messages de debug
debug = True

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
        sample_width = struct.unpack('<H', header[34:36])[0]
    return sample_width

def load_wav():
    global filepath, frame_rate, iq_wave, N, overlap
    if filepath is None:
        filepath = tk.filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not filepath:
        return
    
    encodage = find_sample_width(filepath)
    if encodage == 32:
        n_bits = np.int32
    elif encodage == 16:
        n_bits = np.int16
    elif encodage == 8:
        n_bits = np.int8
    elif encodage == 64:
        n_bits = np.int64
    else:   
        n_bits = np.int32
        if debug is True:
            print("Encodage inconnu. Utilisation de l'encodage 32 bits par défaut")

    frame_rate, s_wave = wav.read(filepath)
    iq_wave = np.frombuffer(s_wave, dtype=n_bits)  
    left, right = iq_wave[0::2], iq_wave[1::2]

    if len(iq_wave) > 1e6: # si plus d'un million d'échantillons
        N = 4096
    elif len(iq_wave) < frame_rate: # si moins d'une seconde
        N = (frame_rate//50)*(len(iq_wave)/frame_rate) # résolution de 50 Hz par défaut, plus proportionnellement à la durée si inférieur à 1 seconde
        N = (int(N/2))*2 # N pair de préférence
        if N < 4: # taille minimum de 4 échantillons
            N = 4
    else:
        N = 512 # taille de fenêtre FFT par défaut
    overlap = N//overlap_value

    try:
        iq_wave = left + 1j * right  # signal IQ complexe
    except:
        # retire quelques échantillons pour éviter les erreurs si nécessaire
        iq_wave = left[:min(len(left), len(right))] + 1j * right[:min(len(left), len(right))]
        if debug is True:
            print("Erreur de conversion IQ. Retrait de quelques échantillons pour faire correspondre I et Q")
    display_file_info()
    # Plot graphes initiaux après chargement du fichier
    plot_initial_graphs()

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
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_wave, frame_rate, N)
    line, = ax[1].plot(f, Pxx)
    # Suppression du segment le plus long qui affiche une ligne inutile
    # Cherche le plus long segment
    distances = np.sqrt(np.diff(f)**2 + np.diff(Pxx)**2)
    longest_segment_index = np.argmax(distances)
    # Coordonnées du segment le plus long
    x1, y1 = f[longest_segment_index], Pxx[longest_segment_index]
    x2, y2 = f[longest_segment_index + 1], Pxx[longest_segment_index + 1]
    # Retire le segment le plus long du graphe en le remplaçant par une ligne blanche
    ax[1].plot([x1, x2], [y1, y2], color="white", linewidth=3) # width = 3 pour être sûr de bien couvrir le segment même si la ligne est inclinée
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
    del stft_matrix, canvas, spec, a0, a1, freqs, times, fmin, fmax, f, Pxx, line, distances, longest_segment_index, x1, y1, x2, y2

# Fonc pour changer la taille de fenêtre FFT
def define_N():
    global N, overlap_value, overlap
    N = int(tk.simpledialog.askstring("N", lang["define_n"]))
    if N is None:
        if debug is True:
            print("Taille de fenêtre FFT non définie")
        return
    N = (int(N/2))*2 # N doit être pair
    overlap = N//overlap_value
    print(lang["fft_window"], N)
    stft_solo()
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
    popup.geometry("300x200")
    window_list = tk.StringVar()
    window_list.set(window_choice)
    tk.Radiobutton(popup, text="Hann", variable=window_list, value="hann").pack()
    tk.Radiobutton(popup, text="Kaiser", variable=window_list, value="kaiser").pack()
    tk.Radiobutton(popup, text="Hamming", variable=window_list, value="hamming").pack()
    tk.Radiobutton(popup, text="Blackman", variable=window_list, value="blackman").pack()
    tk.Radiobutton(popup, text="Bartlett", variable=window_list, value="bartlett").pack()
    tk.Radiobutton(popup, text="Flat Top", variable=window_list, value="flattop").pack()
    tk.Radiobutton(popup, text="Rectangular", variable=window_list, value="rect").pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.grab_set()
    popup.wait_window()
    window_choice = window_list.get()

    if window_choice is None or window_choice == "":
        window_choice = "hann"
        if debug is True:
            print("Pas de fenêtre définie pour la STFT")
        return
    print("Fenêtre définie pour la STFT: ", window_choice)
    stft_solo()

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
    persistance_bins = tk.simpledialog.askstring(lang["params"], lang["pers_bins"])
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
    diff_window = tk.simpledialog.askstring(lang["params"], lang["smoothing_val"])
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
    hist_bins = tk.simpledialog.askstring(lang["params"], lang["smoothing_val"])
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
    enter_overlap = tk.simpledialog.askstring(lang["params"], lang["overlap_val"])
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
    stft_solo()

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
                    {lang['estim_speed_3']} {estim_speed_3[0]*2} / {estim_speed_3[1]*2} Bds").pack()
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
    info_label.config(text=f"{filepath}. {lang['encoding']} {find_sample_width(filepath)} bits. \
                      \n{lang['samples']} {len(iq_wave)}. {lang['sampling_frq']} {frame_rate} Hz. {lang['duree']}: {len(iq_wave)/frame_rate:.2f} sec.\
                      \n {lang['fft_window']} {N}. Overlap : {overlap}. {lang['f_resol']} {frame_rate/N:.2f} Hz.")
    if debug is True:
        print("Affichage des informations du fichier")
        print("Chargé: ", filepath)
        print("Encodage: ", find_sample_width(filepath), " bits")
        print("Echantillons: ", len(iq_wave))
        print("Fréquence d'échantillonnage: ", frame_rate, " Hz")
        print("Durée: ", len(iq_wave)/frame_rate, " secondes")
        print("Taille fenêtre FFT: ", N)
        print("Recouvrement: ", overlap)

# Fonctions de traitement du signal (filtres, déplacement de fréquence, sous-échantillonnage, sur-échantillonnage, coupure)
def move_frequency():
    # déplacement de la fréquence centrale (valeur entrée par l'utilisateur)
    global iq_wave, frame_rate
    fcenter = float(tk.simpledialog.askstring(lang["fc"], lang["move_txt"]))
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
    popup.grab_set()
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
    popup.grab_set()
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
    popup.grab_set()
    popup.wait_window()
    if mean_filter.get() == lang["def_level"]:
        iq_floor = float(tk.simpledialog.askstring(lang["level"], lang["enter_level"]))
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
    rate = tk.simpledialog.askstring(lang["downsample"], lang["down_value"])
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
    rate = tk.simpledialog.askstring(lang["upsample"], lang["up_value"])
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
    popup.title(lang["cut"])
    popup.geometry("300x200")
    start = tk.StringVar()
    end = tk.StringVar()
    tk.Label(popup, text=lang["start_cut"]).pack()
    tk.Entry(popup, textvariable=start).pack()
    tk.Label(popup, text=lang["end_cut"]).pack()
    tk.Entry(popup, textvariable=end).pack()
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.grab_set()
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

def dsp():
    # affichage de la densité spectrale de puissance et de la bande passante estimée
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text, bw
    print(lang["dsp"])
    print(lang["bandwidth"])
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["dsp"])
    ax = fig.add_subplot()
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_wave, frame_rate, N)
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
    diff_window = tk.simpledialog.askstring(lang["params"], lang["smoothing_val"])
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
    Tu = tk.simpledialog.askstring(lang["alpha"], lang["estim_tu"])
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
    alpha_0 = tk.simpledialog.askstring(lang["alpha"], lang["alpha0"])
    if alpha_0 is None:
        if debug is True:
            print("Fréquence alpha0 non définie")
        return
    alpha_0 = float(alpha_0)
    dsp()
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_wave, frame_rate, N)
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
            param_mapping.set(lang["mapping_nat"])
        else:
            mapping_nat.config(state=tk.DISABLED)
            mapping_gray.config(state=tk.DISABLED)
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
    tk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.grab_set()
    popup.wait_window()
    if target_rate.get() == "":
        if debug is True:
            print("Rapidité de démodulation non définie.")
        return
    elif target_rate == 0:
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
        mapping = "grey"

    # fonction de démod et slice bits en fonction de l'ordre
    try:
        symbols, clock = dm.wpcr(freq_diff, frame_rate, target_rate, tau, precision, debug)
        if len(symbols) > 2 and order == 2:
            bits=dm.slice_binary(symbols)
        elif len(symbols) > 2 and order == 4:
            bits = dm.slice_4fsk(symbols,mapping)
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
        if len(bits) > 5000: # graphe allégé si signal long
            bits_plot = bits[:5000]
            fig.suptitle(f"{lang['estim_bits']} {lang["short_bits"]}")
        else:
            bits_plot = bits
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
    text_box = scrolledtext.ScrolledText(plot_frame,height=6, wrap=tk.NONE)
    text_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    text_box.config(state=tk.NORMAL)  # Temporarily enable text box
    try:
        text_output= f"{lang['estim_bits']} ({len(bits)}). {lang['clock_frequency']} {clock} Hz : \n"
        # lignes de bits
        bits_per_line = 64  # formattage à ajuster
        bit_lines = ["".join(map(str, bits[i:i+bits_per_line])) for i in range(0, len(bits), bits_per_line)]
        formatted_bits = "\n".join(bit_lines)
        text_output += formatted_bits
    except:
        bits,bits_plot,formatted_bits = "","",""
        text_output = lang["bits_fail"]
    text_box.insert(tk.END, text_output)
    text_box.config(state=tk.DISABLED)
    del canvas, time, freq_diff, bits, bits_plot, formatted_bits, text_output, text_box


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
    power_menu = tk.Menu(menu_bar, tearoff=0)
    diff_menu = tk.Menu(menu_bar, tearoff=0)
    acf_menu = tk.Menu(menu_bar, tearoff=0)
    ofdm_menu = tk.Menu(menu_bar, tearoff=0)
    freq_menu = tk.Menu(menu_bar, tearoff=0)
    demod_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label=lang["display"], menu=info_menu)
    menu_bar.add_cascade(label=lang["modify"], menu=mod_menu)
    menu_bar.add_cascade(label=lang["main_viz"], menu=graphs_menu)
    menu_bar.add_cascade(label=lang["power_estimate"], menu=power_menu)
    menu_bar.add_cascade(label=lang["freq_estimate"], menu=freq_menu)
    menu_bar.add_cascade(label=lang["phase_estimate"], menu=diff_menu)
    menu_bar.add_cascade(label=lang["cyclo_estimate"], menu=acf_menu)
    menu_bar.add_cascade(label=lang["ofdm"], menu=ofdm_menu)
    menu_bar.add_cascade(label=lang["demod"], menu=demod_menu)
    # Redéfinit les labels des boutons
    load_button.config(text=lang["load"])
    close_button.config(text=lang["close"])
    mode_button.config(text=lang["cursors_off"])
    clear_button.config(text=lang["clear_cursors"])
    info_label.config(text=lang["load_msg"])
    peak_button.config(text=lang["peak_find"])
    # Options des menus cascade
    # Graphes : spectre, STFT, 3D, taille FFT
    graphs_menu.add_command(label=lang["group_spec"], command=plot_initial_graphs)
    graphs_menu.add_command(label=lang["group_const"], command=plot_other_graphs)
    graphs_menu.add_command(label=lang["spec_3d"], command=plot_3d_spectrogram)
    graphs_menu.add_command(label=lang["spectrogram"], command=stft_solo)
    # Ajouter des infos sur le signal
    info_menu.add_command(label=lang["frq_info"], command=display_frq_info)
    # Changer langue. Label "English" si langue = fr, "Français" si langue = en
    lang_switch = "Switch language: English" if lang['lang'] == "Français" else "Changer langue: Français"
    info_menu.add_command(label=lang_switch, command=change_lang)
    info_menu.add_command(label=lang["param_phase_freq"], command=set_diff_params)
    info_menu.add_command(label=lang["param_spectre_persistance"], command=param_spectre_persistance)
    info_menu.add_command(label=lang["param_hist_bins"],command=set_hist_bins)
    info_menu.add_command(label=lang["fft_size"], command=define_N)
    info_menu.add_command(label=lang["set_window"], command=set_window)
    info_menu.add_command(label=lang["set_overlap"], command=set_overlap)
    # Modifications du signal
    mod_menu.add_command(label=lang["filter_high_low"], command=apply_filter_high_low)
    mod_menu.add_command(label=lang["filter_band"], command=apply_filter_band)
    # Filtrage curseurs à ajouter
    mod_menu.add_command(label=lang["move_frq"], command=move_frequency)
    mod_menu.add_command(label=lang["move_freq_cursors"], command=move_frequency_cursors)
    mod_menu.add_command(label=lang["mean"], command=mean_filter)
    mod_menu.add_command(label=lang["downsample"], command=downsample_signal)
    mod_menu.add_command(label=lang["upsample"], command=upsample_signal)
    mod_menu.add_command(label=lang["cut"], command=cut_signal)
    mod_menu.add_command(label=lang["cut_cursors"], command=cut_signal_cursors)
    mod_menu.add_command(label=lang["save_wav"], command=save_as_wav)
    # Analyse du signal
    # Puissance
    power_menu.add_command(label=lang["psf"], command=psf)
    power_menu.add_command(label=lang["mts"], command=mts)
    power_menu.add_command(label=lang["pseries"], command=pseries)
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
    # demod_menu.add_command(label=lang["demod_psk"], command=demod_psk)
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
