"""
SigAnTo - Signal Analysis Toolbox - GUI
Author: Arnaud Barraquand (@Ukratic)
Date: 2024-11-21 -> 2026-10-04

A GUI for signal analysis using Python. 
4 main libraries used : tkinter, matplotlib, numpy & scipy.
Everything else is either native python, a custom module, or optional.
The GUI allows the user to load a WAV file, display its spectrogram, DSP... and many other graphs.
Some signal modification functions are available : apply filters, cut the signal, move the center frequency...
And some estimations, mostly through graphs : Modulation, symbol rate, bandwidth, OFDM parameters.

Many thanks to Dr Marc Lichtman, author of PySDR.
His guide was a great help & starter for the conception of this project.
"""
import sys
import os
import tkinter as tk
from tkinter import ttk
from threading import Thread, Event

# Activation option audio
with_sound = True
# Messages de debug
debug = True
# Drag & drop
drag_drop = True
if drag_drop is True:
    from tkinterdnd2 import TkinterDnD, DND_FILES

loading_end = Event()
def loading_libs():
    """Fonction de chargement des librairies et de la langue, 
    dans un thread pour afficher une fenêtre de chargement.
    En attendant que tout soit prêt, la fenêtre de chargement reste affichée."""
    # Librairies
    print("Chargement des dépendances...")
    global struct, gc, FigureCanvasTkAgg, NavigationToolbar2Tk, plt, cm, np, wav, ll, em, lang, mg, sm, df, dm, scrolledtext, ttk, sd, hf, string
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
    import helper_funcs as hf
    import string

    if with_sound:
        import sounddevice as sd
    # Langue, noms des fonctions, etc.
    lang = ll.get_fra_lib()
    # loading_end.wait() à la fin du script
    loading_end.set()

# fonction pour trouver le chemin relatif des ressources même après compilation avec PyInstaller
# utilisé uniquement pour l'icône actuellement, mais peut être réutilisé si besoin
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

icon_path = resource_path("waves.ico")

# Fenêtre de chargement
t = Thread(target=loading_libs, daemon=True)
t.start()
if drag_drop is True:
    root = TkinterDnD.Tk()
else:
    root = tk.Tk()

root.withdraw()
loading_screen = tk.Toplevel(root)
loading_screen.title("SigAnTo v1.3")
loading_screen.geometry("250x100")
loading_screen.iconbitmap(icon_path)
loading_label = ttk.Label(loading_screen, text="Loading...")
loading_label.pack()

while t.is_alive():
    root.update()

# Init fenêtre principale
print("Chargement de l'application...")
# Fenêtre principale
root.title(lang["siganto"])
root.geometry("1024x768")
root.iconbitmap(icon_path)
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
diff_window = 5 # fenêtre de lissage des transitions
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
filter_kind = 'Butterworth' # type de filtre par défaut (Butterworth)
peak_prominence = 6 # proéminence des pics pour le centrage du signal, en dB
morlet_fc = 6.0  # param de fréquence centrale par défaut pour ondelette de Morlet
morlet_nfreq = 96  # nombre de fréquences par défaut pour CWT Morlet
acf_min_distance = 25  # distance min par défaut pour ACF
scf_alpha_step = 10  # pas alpha par défaut pour SCF
costas_damping = 0.707  # facteur d'amortissement par défaut pour Costas
costas_bw_factor = 0.0314  # facteur de boucle bande passante par défaut pour Costas
eye_num_traces = 500  # nombre de traces par défaut pour diagramme de l'oeil
eye_channel = 'I'  # canal par défaut pour diagramme de l'oeil
eye_symbols = 2 # nombre de symboles par défaut pour diagramme de l'oeil
mfsk_tresh_db = 8  # seuil en dB par défaut pour MFSK
mfsk_peak_prom_db = None  # proéminence des pics en dB par défaut pour MFSK
mfsk_win_factor = 1.0  # facteur de taille de fenêtre par défaut pour MFSK
mfsk_hop_factor = 0.25  # facteur de saut par défaut pour MFSK
mfsk_bin_width_cluster_factor = 1.2 # facteur de largeur de bin par défaut pour MFSK
mfsk_viterbi_penalty = 0.05  # pénalité Viterbi par défaut pour MFSK
alpha_noise = 1.2 # facteur de seuil pour le débruitage spectral, à ajuster selon les besoins
constel_hexbins = True # pour forcer la constellation en scatter plot indépendamment du nombre de points
decim_spec = True # pour décimation du spectrogramme si nécessaire (affichage plus rapide)

# Frame pour les graphes
plot_frame = ttk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Fonctions de chargement de fichier WAV & conversion en IQ, avec vérification de l'encodage et des canaux
def find_sample_width(file_path):
    with open(file_path, 'rb') as f:
        # Check RIFF/WAVE
        if f.read(4) != b'RIFF':
            tk.messagebox.showerror(lang["error"], lang["invalid_wav"], parent=root)
            raise ValueError(lang["invalid_wav"])
        f.read(4)  # skip taille du fichier
        if f.read(4) != b'WAVE':
            tk.messagebox.showerror(lang["error"], lang["invalid_wav"], parent=root)
            raise ValueError(lang["invalid_wav"])
        # recherche du chunk "fmt" pour trouver le format d'encodage & bits par échantillon
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                tk.messagebox.showerror(lang["error"], lang["invalid_wav"], parent=root)
                raise ValueError(lang["invalid_wav"])
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                if not bits_per_sample % 2 == 0 or bits_per_sample > 64 or bits_per_sample < 8:
                    tk.messagebox.showerror(lang["error"], lang["unsupported_bits"].format(bits_per_sample), parent=root)
                    raise ValueError(lang["unsupported_bits"].format(bits_per_sample))
                # if audio_format not in [1, 3]:
                #     tk.messagebox.showerror(lang["error"], lang["unsupported_format"], parent=root)
                #     raise ValueError(lang["unsupported_format"])
                audio_format = "PCM" if audio_format == 1 else "IEEE Float" if audio_format == 3 else "Unknown"
                return bits_per_sample, audio_format
            else:
                # skip, on avance dans le fichier pour trouver le chunk "fmt"
                f.seek(chunk_size, 1)

def load_wav():
    """Charge un fichier WAV, convertit en I/Q,
    et affiche les graphes initiaux."""
    global filepath, s_rate, iq_sig, N, overlap, corr, convert_button
    if filepath is None:
        filepath = tk.filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not filepath:
        return
    try:
        s_rate, s_wave = wav.read(filepath)
    except:
        if debug is True:
            print("Erreur de lecture du fichier WAV")
        tk.messagebox.showerror(lang["error"], lang["invalid_wav"], parent=root)
        wav_metadata()
        close_wav()

    if len(s_wave) > 5e8: # si plus de 500 millions d'échantillons
        if debug is True:
            print("Fichier WAV trop volumineux. Clipping...")
        tk.messagebox.showwarning(lang["error"], lang["file_too_large"].format(int(5e8/1e6)), parent=root)
        s_wave = s_wave[:int(5e8)] # on ne garde que 500 millions d'échantillons pour éviter les erreurs de mémoire

    try:
        if s_wave.ndim == 2 and s_wave.shape[1] == 2:
            # Fichier stéréo : on suppose I/Q sur 2 canaux
            left = s_wave[:, 0].astype(np.float32)
            right = s_wave[:, 1].astype(np.float32)
            # on assure le même nb d'échantillons pour éviter les erreurs
            min_len = min(len(left), len(right))
            left = left[:min_len]
            right = right[:min_len]
            # Vérification de corrélation entre les canaux gauche et droit
            corr = np.corrcoef(left, right)[0,1]
            if abs(corr) > 0.99:
                if debug is True:
                    print("Attention: canaux droite & gauche identiques. On force en mono.")
                mono_signal = left
                analytic_signal = sm.hilbert(mono_signal)
                iq_sig = analytic_signal
            else: # si les canaux sont différents, on suppose qu'ils contiennent bien I et Q
                iq_sig = left + 1j*right
        elif s_wave.ndim == 1:
            # Mono : 2 hypothèses
            if not convert_button:
                load_mono_button.pack(side=tk.LEFT, padx=5)
                convert_button = True
            if mono_real is True: # signal réel à convertir en IQ via transformée de Hilbert
                iq_sig = sm.hilbert(s_wave)
                iq_sig = iq_sig * np.exp(-1j*2*np.pi*(s_rate//4)*np.arange(len(iq_sig))/s_rate)
                iq_sig, s_rate = sm.downsample(iq_sig, s_rate, 2)
            else: # signal déjà échantillonné en I/Q sur un seul canal (alternance I/Q)
                left = s_wave[0::2].astype(np.float32)
                right = s_wave[1::2].astype(np.float32)
                min_len = min(len(left), len(right))
                iq_sig = left[:min_len] + 1j * right[:min_len]

        else:
            tk.messagebox.showerror(lang["error"], lang["wav_conversion"], parent=root)
            if debug is True:
                print("Format WAV inattendu")
            wav_metadata()
            close_wav()
            return

    except:
        if debug is True:
            print("Erreur de conversion IQ")
        tk.messagebox.showerror(lang["error"], lang["wav_conversion"], parent=root)
        wav_metadata()
        close_wav()
    if len(iq_sig) > 5e6: # si plus de 5 millions d'échantillons
        N = 4096
        # pas de recouvrement pour les signaux très longs, pour accélérer le calcul
        overlap = 0
    elif len(iq_sig) > 1e6: # si plus d'un million d'échantillons
        N = 4096
        overlap = N//(overlap_value*4)
    elif 1e5 < len(iq_sig) < 1e6 : # si entre 100k et 1 million d'échantillons
        N = 1024
        overlap = N//overlap_value
    elif len(iq_sig) < s_rate: # si moins d'une seconde
        # base de résolution = 25 Hz par défaut, dégradation proportionnelle à la durée si < 1 seconde
        N = (s_rate//25)*(len(iq_sig)/s_rate) 
        N = (int(N/2))*2 # N pair de préférence
        if N < 4: # taille minimum de 4 échantillons
            N = 4
        if overlap_value > 2:
            overlap = N//overlap_value*2 # recouvrement plus important pour les signaux très courts
        else:
            overlap = N//overlap_value
    elif len(iq_sig) < 5e3:
        N = 128 # taille de fenêtre FFT pour les signaux > 1 sec mais < 5k échantillons
        if overlap_value > 2:
            overlap = N//overlap_value*2 # recouvrement plus important pour les signaux très courts
        else:
            overlap = N//overlap_value
    else:
        N = 512 # taille de fenêtre FFT par défaut
        overlap = N//overlap_value

    display_file_info()
    # Plot graphes initiaux après chargement du fichier
    plot_initial_graphs()

    if len(iq_sig) > 5e7: # gros fichier (clippé ou non)
            tk.messagebox.showwarning("Info",lang["large_file"], parent=root)

def split_file():
    global filepath
    # Fonction pour splitter le fichier WAV en plusieurs fichiers plus petits
    if filepath is None:
        filepath = tk.filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not filepath:
        return
    try:
        s_rate, s_wave = wav.read(filepath)
    except:
        if debug is True:
            print("Erreur de lecture du fichier WAV")
        tk.messagebox.showerror(lang["error"], lang["invalid_wav"], parent=root)
        wav_metadata()
        close_wav()
        return

    # Demande à l'utilisateur la durée maximale de chaque fichier en secondes
    max_duration = tk.simpledialog.askinteger(lang["split_file"], lang["max_duration"], parent=root)
    if not max_duration:
        return
    # Calculer le nombre d'échantillons pour la durée maximale
    max_samples = int(max_duration * s_rate)
    # Calculer le nombre de fichiers à créer
    num_files = int(np.ceil(len(s_wave) / max_samples))
    if num_files == 1:
        if debug is True:
            print("Durée supérieure au fichier, aucun split réalisé")
        return

    # Charge le fichier et le split en plusieurs fichiers wav, conservant le même format d'encodage et le même nombre de canaux
    for i in range(num_files):
        start_idx = i * max_samples
        end_idx = min((i + 1) * max_samples, len(s_wave))
        split_wave = s_wave[start_idx:end_idx]
        split_filename = f"{os.path.splitext(filepath)[0]}_part{i+1}.wav"
        wav.write(split_filename, s_rate, split_wave)
        if debug is True:
            print(f"Fichier {split_filename} créé avec succès.")
    
    tk.messagebox.showinfo(lang["split_file"], lang["files_created"].format(num_files), parent=root)

def on_file_drop(event):
    global filepath
    files = root.tk.splitlist(event.data)  # gère les chemins avec espaces
    if files:
        filepath = files[0]  # prends le premier fichier si plusieurs sont déposés
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
        raise Exception("Erreur de nettoyage du graphe")

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

def update_on_zoom(event_ax):
    # Callback pour mettre à jour les données du graphe décimé, lors d'un zoom/déplacement
    xmin, xmax = event_ax.get_xlim()
    global iq_sig, s_rate
    
    # Déterminer si cas particulier (utilisé notamment pour la fréquence instantanée)
    if hasattr(event_ax, "_custom_time") and hasattr(event_ax, "_custom_data"):
        full_time = event_ax._custom_time
        full_data = event_ax._custom_data
    else:
        # Sinon défaut (amplitude du signal par ex)
        full_time = np.arange(len(iq_sig)) / s_rate
        full_data = iq_sig / np.max(np.abs(iq_sig))
    if debug is True and ((xmax - xmin) < len(full_time)/s_rate): # si zoom < la durée totale
        print("Zoom sur l'axe X : ", xmin, " à ", xmax)

    # Sélectionner uniquement les données visibles dans la fenêtre zoomée
    visible_mask = (full_time >= xmin) & (full_time <= xmax)
    slice_time = full_time[visible_mask]
    slice_data = full_data[visible_mask]
    if len(slice_time) < 2:
        return
    
    # Décimation sur les données visibles pour l'affichage
    t_ds, data_ds = hf.downsample_for_plotting(slice_time, slice_data, target_points=4000)
    # Mise à jour des données du graphe avec les données décimées
    line = event_ax.get_lines()[0]
    line.set_data(t_ds, data_ds)
    event_ax.figure.canvas.draw_idle()

def wav_metadata():
    global filepath
    if filepath is None:
        filepath = tk.filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not filepath:
        return
    metadata = hf.read_wav_metadata(filepath)
    if not metadata:
        tk.messagebox.showinfo(lang["error"], lang["no_metadata"], parent=root)
        return
    # Affiche les métadonnées dans une fenêtre scrolledtext
    metadata_window = tk.Toplevel(root)
    metadata_window.title(lang["metadata"])
    metadata_window.geometry("600x400")
    text_area = scrolledtext.ScrolledText(metadata_window, wrap=tk.WORD)
    text_area.pack(fill=tk.BOTH, expand=True)
    for key, value in metadata.items():
        text_area.insert(tk.END, f"{key}: {value}\n")
    text_area.config(state=tk.DISABLED)

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
    freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice, nfft=N*4)
    if debug is True:
        print("STFT calculée: ", stft_matrix.shape[0], " colonnes, ", stft_matrix.shape[1], " lignes")
    if decim_spec and stft_matrix.shape[0] > 1024 and stft_matrix.shape[1] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=1024)  # Décimation pour performance
    elif decim_spec and stft_matrix.shape[1] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=stft_matrix.shape[0])
    elif decim_spec and stft_matrix.shape[0] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=stft_matrix.shape[1], target_rows=1024)
    else:
        pass # pas de décimation si déjà < 1024x1024 ou si décimation désactivée
    if debug is True:
        print("Spectrogramme final: ", stft_matrix.shape[0], " colonnes, ", stft_matrix.shape[1], " lignes")
    if freqs is None:
        print(lang["error_stft"])
        # message d'erreur si la STFT n'a pas pu être calculée
        tk.messagebox.showerror(lang["error"], lang["error_stft"], parent=root)
        return
    df, dt = freqs[1]-freqs[0], times[1]-times[0]
    extent = [freqs[0]-df/2, freqs[-1]+df/2, times[-1]+dt/2, times[0]-dt/2]
    ax[0].imshow(stft_matrix, aspect='auto', extent=extent, cmap='jet')
    ax[0].set_ylabel(f"{lang['time_xy']} [s]")
    ax[0].set_title(f"{lang['window']} {window_choice}")

    # DSP
    bw, fmin, fmax, f, Pxx = mg.estimate_bandwidth(iq_sig, s_rate, N, overlap, window_choice)
    ax[1].plot(f, Pxx)
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

## Autre groupe de graphes de base et flèches pour ajuster la fréquence centrale
def plot_other_graphs():
    global toolbar, ax, fig, canvas, iq_sig, original_iq_sig, fcenter, freq_label
    # Figure avec 3 sous-graphes. 
    # Le premier est sur deux lignes, les deux autres se partagent la 3eme ligne
    original_iq_sig = iq_sig.copy() # copie du signal original pour les modifications
    fcenter = 0  # Init de l'offset FC
    clear_plot()
    fig = plt.figure()
    spec = fig.add_gridspec(2, 2)
    fig.suptitle(lang["spec_const"])
    fig.tight_layout()
    a0 = fig.add_subplot(spec[0:1, :])
    a1 = fig.add_subplot(spec[1, 0])
    a2 = fig.add_subplot(spec[1, 1])
    ax = (a0, a1, a2)
    if not filepath:
        print(lang["no_file"])
        return
    # STFT
    freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice, nfft=N*4)
    if debug is True:
        print("STFT calculée: ", stft_matrix.shape[0], " colonnes, ", stft_matrix.shape[1], " lignes")
    if decim_spec and stft_matrix.shape[0] > 1024 and stft_matrix.shape[1] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=1024)  # Décimation pour performance
    elif decim_spec and stft_matrix.shape[1] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=stft_matrix.shape[0])
    elif decim_spec and stft_matrix.shape[0] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=stft_matrix.shape[1], target_rows=1024)
    else:
        pass # pas de décimation si déjà < 1024x1024 ou si décimation désactivée
    if debug is True:
        print("Spectrogramme final: ", stft_matrix.shape[0], " colonnes, ", stft_matrix.shape[1], " lignes")
    if freqs is None:
        print(lang["error_stft"])
        # message d'erreur si la STFT n'a pas pu être calculée
        tk.messagebox.showerror(lang["error"], lang["error_stft"], parent=root)
        return
    df, dt = freqs[1]-freqs[0], times[1]-times[0]
    extent = [freqs[0]-df/2, freqs[-1]+df/2, times[-1]+dt/2, times[0]-dt/2]
    stft = ax[0].imshow(stft_matrix, aspect='auto', extent=extent, cmap=cm.jet)
    ax[0].set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax[0].set_ylabel(f"{lang['time_xy']} [s]")
    ax[0].set_title(f"{lang['window']} {window_choice}")

    # Constellation
    iq_constel = iq_sig/np.max(np.abs(iq_sig))
    if len(iq_constel) > 1000:
        grid_size = int(np.sqrt(len(iq_constel) // 2))  # Ajuster la taille de la grille en fonction du nombre de points
    else:
        grid_size = 200
    # si il n'y a pas assez de points, on affiche un scatter plot classique
    if constel_hexbins is True:
        line_constellation = ax[1].hexbin(np.real(iq_constel), np.imag(iq_constel), gridsize=grid_size, cmap='jet', bins='log', mincnt=1)
    else:
        line_constellation = ax[1].scatter(np.real(iq_constel), np.imag(iq_constel), s=5)
    ax[1].set_xlabel("In-Phase")
    ax[1].set_ylabel("Quadrature")

    # Transitions de phase
    time, phase_diff = em.phase_time_angle(iq_sig, s_rate, diff_window, window_choice)
    ax[2]._custom_time = time
    ax[2]._custom_data = phase_diff
    time, phase_diff = hf.downsample_for_plotting(time, phase_diff, target_points=4000)
    ax[2].plot(time, phase_diff)
    ax[2].grid()
    ax[2].set_xlabel(f"{lang['time_xy']} [s]")
    ax[2].set_ylabel(f"{lang['diff_phase']} [rad]")

    # Label pour afficher l'offset FC
    freq_label = ttk.Label(plot_frame, text=f"{lang['offset_freq']}: {fcenter} Hz")
    freq_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_graph():
        global iq_sig, fcenter
        # Recompute iq_sig avec l'offset de fréquence à partir de l'iq_sig original
        iq_sig = original_iq_sig * np.exp(-1j * 2 * np.pi * fcenter * np.arange(len(original_iq_sig)) / s_rate)
        # Màj label
        freq_label.config(text=f"{lang['offset_freq']}: {fcenter} Hz")
        # Màj STFT
        freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice, nfft=N*4)

        if decim_spec and stft_matrix.shape[0] > 1024 and stft_matrix.shape[1] > 1024:
            stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=1024)  # Décimation pour performance
        elif decim_spec and stft_matrix.shape[1] > 1024:
            stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=stft_matrix.shape[0])
        elif decim_spec and stft_matrix.shape[0] > 1024:
            stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=stft_matrix.shape[1], target_rows=1024)
        else:
            pass # pas de décimation si déjà < 1024x1024 ou si décimation désactivée
        stft.set_data(stft_matrix)
        # Màj constellation
        iq_constel = iq_sig/np.max(np.abs(iq_sig))
        line_constellation.set_offsets(np.c_[np.real(iq_constel), np.imag(iq_constel)])
        # Màj transitions de phase
        time, phase_diff = em.phase_time_angle(iq_sig, s_rate, diff_window, window_choice)
        ax[2]._custom_time = time
        ax[2]._custom_data = phase_diff
        time, phase_diff = hf.downsample_for_plotting(time, phase_diff, target_points=4000)
        ax[2].lines[0].set_data(time, phase_diff)
        if debug is True:
            print("Mise à jour du graphe avec décalage: ", fcenter, " Hz")

        canvas.draw()

    def change_freq(step, event=None):
        global fcenter
        fcenter += step
        update_graph()

    # Flèches du clavier pour déplacer la fréquence centrale grossièrement
    root.bind("<Left>", lambda e: change_freq(-1, e))
    root.bind("<Right>", lambda e: change_freq(1, e))
    # Sinon boutons sur l'interface pour déplacer finement
    button_frame = ttk.Frame(plot_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)
    left_button = ttk.Button(button_frame, text="←", command=lambda: change_freq(-0.01))
    left_button.pack(side=tk.LEFT, padx=10)
    right_button = ttk.Button(button_frame, text="→", command=lambda: change_freq(0.01))
    right_button.pack(side=tk.RIGHT, padx=10)

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del spec, a0, a1, a2, stft_matrix, freqs, times, iq_constel, time, phase_diff

# Fonc de changement de fenêtre FFT pour STFT
def set_window():
    global window_choice
    #dropdown pour choisir la fenêtre
    popup = tk.Toplevel()
    place_relative(popup, root, 300, 300)
    popup.title(lang["window_choice"])
    window_list = tk.StringVar()
    window_list.set(window_choice)
    ttk.Radiobutton(popup, text="Hann", variable=window_list, value="hann").pack()
    ttk.Radiobutton(popup, text="Kaiser", variable=window_list, value="kaiser").pack()
    ttk.Radiobutton(popup, text="Hamming", variable=window_list, value="hamming").pack()
    ttk.Radiobutton(popup, text="Blackman", variable=window_list, value="blackman").pack()
    ttk.Radiobutton(popup, text="Bartlett", variable=window_list, value="bartlett").pack()
    ttk.Radiobutton(popup, text="Flat Top", variable=window_list, value="flattop").pack()
    ttk.Radiobutton(popup, text="Blackman-Harris 7-term", variable=window_list, value="blackmanharris7term").pack()
    ttk.Radiobutton(popup, text="Rectangular", variable=window_list, value="rectangular").pack()
    ttk.Radiobutton(popup, text="Gaussian", variable=window_list, value="gaussian").pack()
    ttk.Radiobutton(popup, text="Lanczos", variable=window_list, value="lanczos").pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
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
    time, iq_norm = hf.downsample_for_plotting(time, iq_norm, target_points=4000)
    ax.plot(time, iq_norm)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel(lang["norm_amplitude"])
    ax.grid(True)
    ax.callbacks.connect('xlim_changed', update_on_zoom)
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
    freqs, times, stft_matrix = mg.compute_stft(iq_sig, s_rate, window_size=N, overlap=overlap, window_func=window_choice, nfft=N*4)
    if debug is True:
        print("STFT calculée: ", stft_matrix.shape[0], " colonnes, ", stft_matrix.shape[1], " lignes")
    if decim_spec and stft_matrix.shape[0] > 1024 and stft_matrix.shape[1] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=1024)  # Décimation pour performance
    elif decim_spec and stft_matrix.shape[1] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=1024, target_rows=stft_matrix.shape[0])
    elif decim_spec and stft_matrix.shape[0] > 1024:
        stft_matrix = hf.downsample_spectrogram(stft_matrix, target_cols=stft_matrix.shape[1], target_rows=1024)
    else:
        pass # pas de décimation si déjà < 1024x1024 ou si décimation désactivée
    if debug is True:
        print("Spectrogramme final: ", stft_matrix.shape[0], " colonnes, ", stft_matrix.shape[1], " lignes")
    if freqs is None:
        print(lang["error_stft"])
        # message d'erreur si la STFT n'a pas pu être calculée
        tk.messagebox.showerror(lang["error"], lang["error_stft"], parent=root)
        return
    if debug is True:
        print("Affichage STFT")
    df, dt = freqs[1]-freqs[0], times[1]-times[0]
    extent = [freqs[0]-df/2, freqs[-1]+df/2, times[-1]+dt/2, times[0]-dt/2]
    ax.imshow(stft_matrix, aspect='auto', extent = extent,cmap=cm.jet)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(f"{lang['time_xy']} [s]")
    ax.set_title(f"{lang['window']} {window_choice}")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del stft_matrix, canvas, freqs, times

# Affichage infos supplémentaires sur le signal : Mesures de puissance.
# Estimations de largeur de bande, rapidité de modulation & ACF.
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
    freqs, psd = mg.compute_dsp(iq_sig, s_rate, N, overlap, window_choice)
    max_lvl = 10*np.log10(np.max(psd))
    low_lvl = 10*np.log10(np.min(psd))
    mean_lvl = 10*np.log10(np.mean(psd))
    bw = round(mg.estimate_bandwidth(psd=psd, freqs=freqs)[0],2)
    estim_speed_2 = round(np.abs(em.mean_threshold_spectrum(iq_sig, s_rate)[2]),2)
    estim_speed = round(np.abs(em.power_spectrum_envelope(iq_sig, s_rate)[2]),2)
    _,_,_, peak_squared_freq, peak_quartic_freq = em.power_series(iq_sig, s_rate)
    estim_speed_3 = [round(abs(peak_squared_freq),2),round(abs(peak_quartic_freq),2)]
    _, freq_diff = em.frequency_transitions(iq_sig, s_rate, diff_window, window_choice)
    freq_diff /= np.max(np.abs(freq_diff))
    clock,_,_ = dm.estimate_baud_rate(freq_diff, s_rate)
    estim_speed_4 = round(float(clock),2)
    if estim_speed_4 == 0 or np.isnan(estim_speed_4) or np.isinf(estim_speed_4):
        estim_speed_4 = 0.01 # pour éviter les erreurs de division par zéro ou d'affichage
    estim_speed_5 = round(np.abs(em.envelope_spectrum(iq_sig, s_rate)[2]),2)
    acf_peak = round(np.abs(em.autocorrelation_peak(iq_sig, s_rate, min_distance=acf_min_distance)[1]),2)

    # estimation de rapidité de modulation via différentes méthodes et indicateur de confiance
    # résultats exploitables sur canal unique dans le wav ou après filtrage
    confidence = 1
    if abs(estim_speed - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    if abs(estim_speed_2 - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    if abs(estim_speed_3[0]*2 - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    if (abs(estim_speed_3[1]*4 - estim_speed_4)/estim_speed_4 < 0.1) or (abs(estim_speed_3[1]*2 - estim_speed_4)/estim_speed_4 < 0.1) :
        confidence +=1
    if abs(estim_speed_5 - estim_speed_4)/estim_speed_4 < 0.1:
        confidence +=1
    estim_speed_6 = round(np.mean([estim_speed,estim_speed_2,estim_speed_5]),2)
    other_conf_a = abs(estim_speed_2 - estim_speed)/estim_speed_2 + abs(estim_speed_3[1]*2 - estim_speed_5)/(estim_speed_3[1]*2)
    other_conf_b = abs(estim_speed_2 - estim_speed)/estim_speed_2 + abs(estim_speed_3[0]*2 - estim_speed_5)/(estim_speed_3[0]*2)

    if confidence <= 1 and (estim_speed_6-estim_speed_3[0]*2)/estim_speed_6 < 0.1 and estim_speed_6 > bw/10 and estim_speed_6 < bw:
        confidence_level = lang['low_confidence']
        estim_speed_ag = f"{estim_speed_5} Bds ({confidence_level})"
    elif confidence <= 1 and (other_conf_a <= 0.1 or other_conf_b <= 0.1) and estim_speed_5 > bw/10 and estim_speed_5 < bw:
        confidence_level = lang['low_confidence']
        estim_speed_ag = f"{estim_speed_5} Bds ({confidence_level})"
    elif confidence >= 2 and confidence < 3 and estim_speed_4 > bw/10 and estim_speed_4 < bw:
        confidence_level = lang['medium_confidence']
        estim_speed_ag = f"{estim_speed_4} Bds ({confidence_level})"
    elif confidence >= 3 and confidence <= 4 and estim_speed_4 > bw/10 and estim_speed_4 < bw:
        confidence_level = lang['high_confidence']
        estim_speed_ag = f"{estim_speed_4} Bds ({confidence_level})"
    elif confidence >= 5 and estim_speed_6 > bw/10 and estim_speed_6 < bw:
        confidence_level = lang['v_high_confidence']
        estim_speed_ag = f"{round(np.mean([estim_speed_6,estim_speed_4]),2)} Bds ({confidence_level})"
    else: # confidence < 2 sans consensus ou estimations hors bande passante
        confidence_level = lang['low_confidence']
        estim_speed_ag = f"{estim_speed_4} Bds ({confidence_level})"

    popup = tk.Toplevel()
    popup.title(lang["frq_info"])
    place_relative(popup, root, 600, 200)
    ttk.Label(popup, text=f"{lang['high_freq']}: {f_pmax:.2f} Hz. {lang['low_freq']}: {f_pmin:.2f} Hz\n \
                    {lang['high_level']} {max_lvl:.2f}. {lang['low_level']} {low_lvl:.2f}.\n \
                    {lang['mean_level']} {mean_lvl:.2f} dB ").pack()
    ttk.Label(popup, text=f"{lang['estim_bw']} {round(bw,2)} Hz").pack()
    ttk.Label(popup, text=f"{lang['estim_speed']} {estim_speed_ag}").pack()
    ttk.Label(popup, text=f"{lang['acf_peak_txt']} {acf_peak} ms" ).pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()

    if debug is True:
        print("Fréquence à la puissance maximale: ", f_pmax)
        print("Niveau le plus haut: ", max_lvl)
        print("Fréquence à la puissance minimale: ", f_pmin)
        print("Niveau le plus bas: ", low_lvl)
        print("Niveau moyen en dB: ", mean_lvl)
        print("Largeur de bande estimée: ", bw, " Hz")
        print("Rapidité de modulation estimée avec le spectre des transitions: ", estim_speed_2, " Bauds")
        print("Rapidité de modulation estimée par spectre de puissance: ", estim_speed, " Bauds")
        print("Rapidité de modulation estimée par puissances d'ordre: ", estim_speed_3[0]*2, "/", estim_speed_3[1]*2, "Bauds")
        print("Rapidité de modulation estimée par transitions de fréquence: ", estim_speed_4, " Bauds")
        print("Rapidité de modulation estimée par spectre de l'enveloppe: ", estim_speed_5, " Bauds")
        print("Confiance dans l'estimation de rapidité de modulation: ", confidence, "/5")
        print("Autocorrélation estimée: ", acf_peak, " ms")
    del _,estim_speed,estim_speed_2, wav_mag, f, f_pmax, f_pmin, max_lvl, low_lvl, mean_lvl, freq_diff, acf_peak

# Affichage des informations du fichier : nom, encodage, durée, fréquence d'échantillonnage,
# taille de fenêtre FFT, recouvrement, résolution en fréquence
def display_file_info():
    if filepath is None:
        info_label.config(text=lang["no_file"])
        return
    # si l'extension n'est pas .wav
    if not filepath.lower().endswith(".wav"):
        info_label.config(text=f"{filepath}. {lang['file_type']} Custom. \
                          \n{lang['sampling_frq']} {s_rate} Hz. {lang['samples']} {len(iq_sig)}. {lang['duree']}: {len(iq_sig)/s_rate:.2f} sec.\
                          \n{lang['fft_window']} {N}. Overlap : {overlap}. {lang['f_resol']} {s_rate/N:.2f} Hz. {lang['time_interval']} {(N - overlap)/s_rate*1000:.2f} ms.")
        if debug:
            print("Type de fichier non supporté pour l'affichage des informations. Seuls les fichiers wav sont supportés.")
        return
    encodage, formatting = find_sample_width(filepath)
    info_label.config(text=f"{filepath}. {lang['encoding']} {encodage} bits. Format {formatting}. \
                      \n{lang['samples']} {len(iq_sig)}. {lang['sampling_frq']} {s_rate} Hz. {lang['duree']}: {len(iq_sig)/s_rate:.2f} sec.\
                      \n{lang['fft_window']} {N}. Overlap : {overlap}. {lang['f_resol']} {s_rate/N:.2f} Hz. {lang['time_interval']} {(N - overlap)/s_rate*1000:.2f} ms.")
    if debug is True:
        print("Affichage des informations du fichier")
        print("Chargé: ", filepath)
        print("Encodage: ", encodage, " bits")
        print("Format: ", formatting)
        print("Echantillons: ", len(iq_sig))
        print("Fréquence d'échantillonnage: ", s_rate, " Hz")
        print("Durée: ", len(iq_sig)/s_rate, " secondes")
        print("Taille fenêtre FFT: ", N)
        print("Recouvrement: ", overlap)
        print("Résolution en fréquence: ", s_rate/N, " Hz")
        print("Résolution en temps: ", (N - overlap)/s_rate*1000, " ms")

## Fonctions de traitement du signal (filtres, déplacement de fréquence,
# sous-échantillonnage, sur-échantillonnage, coupure)
def move_frequency():
    # déplacement de la fréquence centrale (valeur entrée par l'utilisateur)
    global iq_sig, s_rate
    fcenter = tk.simpledialog.askstring(lang["fc"], lang["move_txt"], parent=root)
    if fcenter is None:
        if debug is True:
            print("Modification de fréquence centrale annulée, valeur non définie")
        return
    try:
        fcenter = float(fcenter)
    except ValueError:
        tk.messagebox.showinfo(lang["error"], lang["invalid_value"], parent=root)
        return
    iq_sig = iq_sig * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_sig))/s_rate)
    if debug is True:
        print("Fréquence centrale déplacée de ", fcenter, " Hz")
    plot_initial_graphs()

def move_frequency_cursors():
    # déplacement de la fréquence centrale sur le curseur (inactif si 2 curseurs)
    global cursor_points, iq_sig, s_rate
    try:
        if len(cursor_points) != 1 and cursor_points[0][0] != cursor_points[1][0]:
            tk.messagebox.showinfo(lang["error"], lang["1pt_cursors"], parent=root)
            return
        fcenter = cursor_points[0][0]
        iq_sig = iq_sig * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_sig))/s_rate)
        if debug is True:
            print("Fréquence centrale déplacée de ", fcenter, " Hz")
        plot_initial_graphs()
    except IndexError:
        tk.messagebox.showinfo(lang["error"], lang["1pt_cursors"], parent=root)
        return

def read_other_filetype():
    # lecture d'un fichier autre que wav (raw, bin, sans extension et/ou en-tête...)
    global iq_sig, s_rate, filepath, overlap, N
    filepath = tk.filedialog.askopenfilename(filetypes=[(lang["all_files"], "*.*")])
    if not filepath:
        if debug is True:
            print("Aucun fichier sélectionné")
        return
    # liste des types de données supportés
    data_types = ["int8", "int16", "int32", "float32", "float64","complex64", "complex128",
                  "int10", "int12", "int14", "int18", "int20", "int22", "int24", "int26",
                  "int28", "int30"]
    popup = tk.Toplevel()
    popup.title(lang["file_type"])
    place_relative(popup, root, 300, 400)
    data_type_var = tk.StringVar()
    ttk.Label(popup, text=lang["data_type_txt"]).pack()
    data_type_menu = ttk.OptionMenu(popup, data_type_var, data_types[1], *data_types)
    data_type_menu.pack()
    # coche pour endianness
    endianness_var = tk.StringVar()
    endianness_var.set("little")  # par défaut
    ttk.Label(popup, text=lang["endianness_txt"]).pack()
    ttk.Radiobutton(popup, text=lang["little_endian"], variable=endianness_var, value="little").pack()
    ttk.Radiobutton(popup, text=lang["big_endian"], variable=endianness_var, value="big").pack()
    signed_var = tk.BooleanVar()
    signed_var.set(True)  # par défaut
    ttk.Checkbutton(popup, text=lang["signed"], variable=signed_var).pack()
    # demander à l'utilisateur de choisir réel ou IQ
    layout_var = tk.StringVar()
    layout_var.set("IQ")  # par défaut
    ttk.Label(popup, text=lang["layout_txt"]).pack()
    ttk.Radiobutton(popup, text=lang["iq_layout"], variable=layout_var, value="IQ").pack()
    ttk.Radiobutton(popup, text=lang["real_layout"], variable=layout_var, value="real").pack()
    # demander si inversion I/Q (par défaut non)
    invert_iq_var = tk.BooleanVar()
    invert_iq_var.set(False)
    ttk.Checkbutton(popup, text=lang["invert_iq"], variable=invert_iq_var).pack()
    # définir offset de bits; par ex pour les fichiers avec en-tête
    ttk.Label(popup, text=lang["offset_txt"]).pack()
    offset_var = tk.StringVar()
    offset_var.set("0")  # valeur par défaut
    ttk.Entry(popup, textvariable=offset_var).pack()
    # retirer DC (par défaut non)
    dc_removal_var = tk.BooleanVar()
    dc_removal_var.set(False)
    ttk.Checkbutton(popup, text=lang["remove_dc"], variable=dc_removal_var).pack()
    # demander freq d'échantillonnage
    ttk.Label(popup, text=lang["s_rate_txt"]).pack()
    s_rate_var = tk.StringVar()
    s_rate_var.set("0")  # valeur par défaut
    ttk.Entry(popup, textvariable=s_rate_var).pack()

    ttk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    data_type = data_type_var.get()
    endianness = endianness_var.get()
    layout = layout_var.get()
    if offset_var.get() == "":
        offset_var.set("0")
    offset = int(offset_var.get())
    dc_removal = dc_removal_var.get()
    invert_iq = invert_iq_var.get()
    s_rate = int(s_rate_var.get())
    sign = signed_var.get()
    if s_rate <= 0:
        if debug is True:
            print("Fréquence d'échantillonnage non définie ou invalide. Lecture du fichier annulée")
        return
    if sign is False and "int" in data_type:
        data_type = data_type.replace("int", "uint")
    numpy_dtypes = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64","complex64", "complex128"]
    # lecture du fichier avec numpy et le dtype défini quand possible
    try:
        if data_type not in numpy_dtypes:
            if endianness == "little":
                little = True
            else:
                little = False
            bit_type = int(data_type.replace("uint", "").replace("int", ""))
            s_wave = hf.read_integer_bits(filepath, bits=bit_type, signed=sign, little_endian=little, offset=offset)
            # fonction de conversion pour les types non standard, en théorie applicale à tous les types de bits,
            # mais peu testée, donc on reste sur la lecture numpy quand possible
        else:
            s_wave = np.fromfile(filepath, dtype=data_type, offset=offset)
        if debug is True:
            print("Fichier chargé: ", filepath)
            print("Type de données: ", data_type)
            print("Endianness: ", endianness)
            print("Layout: ", layout)
            print("Offset: ", offset, " octets")
            print("DC removal: ", dc_removal)
            print("Invert I/Q: ", invert_iq)
            print("Fréquence d'échantillonnage: ", s_rate, " Hz")
        if len(s_wave) == 0:
            tk.messagebox.showerror(lang["error"], lang["file_empty"], parent=root)
            if debug is True:
                print("Fichier vide ou lecture impossible")
            return
        if debug is True:
            print("Nombre d'échantillons lus: ", len(s_wave))
        if endianness == "big" and data_type in numpy_dtypes:
            s_wave = s_wave.byteswap()
        if dc_removal:
            s_wave = s_wave - np.mean(s_wave)
        if layout == "IQ" :
            # on suppose que le fichier contient I/Q sur un seul canal avec alternance I/Q
            # sauf si déjà complexe, auquel cas on ne fait rien
            if data_type.startswith("complex"):
                iq_sig = s_wave.astype(np.complex64)
            else:
                left = s_wave[0::2].astype(np.float32)
                right = s_wave[1::2].astype(np.float32)
                min_len = min(len(left), len(right))
                iq_sig = left[:min_len] + 1j * right[:min_len]
            if invert_iq:
                iq_sig = np.conj(iq_sig)
        elif layout == "real":
            # on suppose que le fichier contient un signal réel à convertir en IQ via transformée de Hilbert
            iq_sig = sm.hilbert(s_wave)
            iq_sig = iq_sig * np.exp(-1j*2*np.pi*(s_rate//4)*np.arange(len(iq_sig))/s_rate)
            iq_sig, s_rate = sm.downsample(iq_sig, s_rate, 2)

        if len(iq_sig) > 5e6: # si plus de 5 millions d'échantillons
            N = 4096
            # pas de recouvrement pour les signaux très longs, pour accélérer le calcul
            overlap = 0
        elif len(iq_sig) > 1e6: # si plus d'un million d'échantillons
            N = 4096
            overlap = N//(overlap_value*4)
        elif 1e5 < len(iq_sig) < 1e6 : # si entre 100k et 1 million d'échantillons
            N = 1024
            overlap = N//overlap_value
        elif len(iq_sig) < s_rate: # si moins d'une seconde
            # base de résolution = 25 Hz par défaut, dégradation proportionnelle à la durée si < 1 seconde
            N = (s_rate//25)*(len(iq_sig)/s_rate) 
            N = (int(N/2))*2 # N pair de préférence
            if N < 4: # taille minimum de 4 échantillons
                N = 4
            if overlap_value > 2:
                overlap = N//overlap_value*2 # recouvrement plus important pour les signaux très courts
            else:
                overlap = N//overlap_value
        elif len(iq_sig) < 5e3:
            N = 128 # taille de fenêtre FFT pour les signaux > 1 sec mais < 5k échantillons
            if overlap_value > 2:
                overlap = N//overlap_value*2 # recouvrement plus important pour les signaux très courts
            else:
                overlap = N//overlap_value
        else:
            N = 512 # taille de fenêtre FFT par défaut
            overlap = N//overlap_value
    
        display_file_info()
        # Plot graphes initiaux après chargement du fichier
        plot_initial_graphs()
    except Exception as e:
        tk.messagebox.showerror(lang["error"], f"{lang['file_read_error']} {e}", parent=root)
        if debug is True:
            print("Erreur lors de la lecture du fichier: ", e)
        close_wav()
        return


def apply_filter_high_low():
    # passage d'un filtre passe-haut ou passe-bas
    global iq_sig, s_rate
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    popup.title(lang["high_low"])
    place_relative(popup, root, 300, 200)
    filter_type = tk.StringVar()
    filter_type.set(lang["low_val"])
    ttk.Radiobutton(popup, text=lang["low_val"], variable=filter_type, value=lang["low_val"]).pack()
    ttk.Radiobutton(popup, text=lang["high_val"], variable=filter_type, value=lang["high_val"]).pack()
    cutoff = tk.StringVar()
    ttk.Label(popup, text=lang["freq_pass"]).pack()
    ttk.Entry(popup, textvariable=cutoff).pack()
    # si un curseur est actif, pré-remplit la fréquence de coupure avec la valeur du curseur
    if len(cursor_points) == 1 or (len(cursor_points) == 2 and cursor_points[0][0] == cursor_points[1][0]):
        cutoff.set(str(abs(round(cursor_points[0][0],3))))
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if cutoff.get() == "" or float(cutoff.get()) <= 0 or float(cutoff.get()) >= s_rate/2:
        if debug is True:
            print("Filtre passe-", filter_type.get(), " non appliqué. Fréquence de coupure non définie")
        return
    if filter_type.get() == lang["low_val"]:
        low_high = 'low'
    elif filter_type.get() == lang["high_val"]:
        low_high = 'high'
    iq_sig = sm.low_high_pass_filter(iq_sig, float(cutoff.get()), s_rate, filter_order,filter_kind,low_high)
    if debug is True:
        print("Filtre passe-", filter_type.get(), " appliqué. Fréquence de coupure: ", cutoff.get(), "Hz")
    plot_initial_graphs()

def apply_filter_band():
    # passage d'un filtre passe-bande / coupe-bande
    global iq_sig, s_rate
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    popup.title(lang["bandpass"])
    place_relative(popup, root, 300, 200)
    filter_type = tk.StringVar()
    filter_type.set(lang["bandpass"])
    ttk.Radiobutton(popup, text=lang["bandpass"], variable=filter_type, value=lang["bandpass"]).pack()
    ttk.Radiobutton(popup, text=lang["bandstop"], variable=filter_type, value=lang["bandstop"]).pack()
    lowcut = tk.StringVar()
    highcut = tk.StringVar()
    ttk.Label(popup, text=lang["freq_low"]).pack()
    ttk.Entry(popup, textvariable=lowcut).pack()
    ttk.Label(popup, text=lang["freq_high"]).pack()
    ttk.Entry(popup, textvariable=highcut).pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if (lowcut.get() == "" and highcut.get() != "") or (lowcut.get() != "" and highcut.get() == ""):
        tk.messagebox.showinfo(lang["error"], lang["freq_valid"], parent=root)
        return
    elif lowcut.get() == "" and highcut.get() == "":
        return
    else:
        if filter_type.get() == lang["bandpass"]:
            bandtype = 'band'
        elif filter_type.get() == lang["bandstop"]:
            bandtype = 'stop'
        iq_sig = sm.band_filter(iq_sig, float(lowcut.get()), float(highcut.get()), s_rate, filter_order, filter_kind, bandtype)
    if debug is True:
        print("Filtre passe-bande / coupe-bande appliqué. Fréquence de coupure basse: ", lowcut.get(), "Hz")
        print("Fréquence de coupure haute: ", highcut.get(), "Hz")
    plot_initial_graphs()

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
    if len(cursor_points) == 1 or (len(cursor_points) == 2 and cursor_points[0][0] == cursor_points[1][0]):
        init_value_poly = int(abs(cursor_points[0][0])*2) # pré-remplit la fréquence de rééchantillonnage avec la valeur du curseur
    else:        
        init_value_poly = ""
    new_fs = tk.simpledialog.askstring(lang["resample_poly"], lang["resample_value"], parent=root, initialvalue=init_value_poly)
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
    ttk.Label(popup, text=lang["start_cut"]).pack()
    ttk.Entry(popup, textvariable=start).pack()
    ttk.Label(popup, text=lang["end_cut"]).pack()
    ttk.Entry(popup, textvariable=end).pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    if start.get() =="" and end.get() =="":
        # tk.messagebox.showinfo(lang["error"], lang["valid_cut"], parent=root)
        return
    if start.get() =="":
        start = 0
    else:
        start = int(float(start.get())*s_rate)
    if end.get() =="":
        end = len(iq_sig)
    else:
        end = int(float(end.get())*s_rate)
    if start < 0 or end < 0:
        tk.messagebox.showinfo(lang["error"], lang["valid_cut"], parent=root)
    if start > end:
        tk.messagebox.showinfo(lang["error"], lang["valid_cut"], parent=root)
    if debug is True:
        print("Valeurs de découpe définies ; début :", start, "fin :",end)
    iq_sig = iq_sig[start:end]
    if debug is True:
        print("Signal coupé de ", start, "échantillons à ", end, "échantillons")
        print("Soit ", end-start, "échantillons restants")
        print("Nouvelle durée du signal : ", len(iq_sig)/s_rate, " secondes")
    plot_initial_graphs()
    display_file_info()

def cut_signal_cursors():
    # coupure du signal entre les 2 curseurs (ne prend en compte que la durée, pas l'écart en fréquence)
    global iq_sig, s_rate, cursor_points
    if len(cursor_points) < 2:
        tk.messagebox.showinfo(lang["error"], lang["2pt_cursors"], parent=root)
        return
    # signal coupé entre les 2 points. On ne sait pas quel point est le début et lequel est la fin, 
    # donc on prend les valeurs y les plus petites et les plus grandes
    start = int(cursor_points[0][1]*s_rate)
    end = int(cursor_points[1][1]*s_rate)
    if debug is True:
        print(cursor_points[1][1],cursor_points[0][1])
    if cursor_points[1][1] < cursor_points[0][1]:
        iq_sig = iq_sig[end:start]
        if debug is True:
            print("Signal coupé de ", end, "échantillons à ", start, "échantillons")
            print("Soit ", start-end, "échantillons restants")
            print("Nouvelle durée du signal : ", len(iq_sig)/s_rate, " secondes")
    else:
        iq_sig = iq_sig[start:end]
        if debug is True:
            print("Signal coupé de ", start, "échantillons à ", end, "échantillons")
            print("Soit ", end-start, "échantillons restants")
            print("Nouvelle durée du signal : ", len(iq_sig)/s_rate, " secondes")
    plot_initial_graphs()
    display_file_info()

def center_signal_coarse():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # param de proéminence à ajuster
    try:
        iq_sig, center = df.center_signal(iq_sig, s_rate, prominence=peak_prominence)
    except:
        tk.messagebox.showinfo(lang["error"], lang["centering_fail"], parent=root)
        return
    if debug is True:
        print(f"Signal centré grossièrement par déplacement de {round(center, 2)} Hz.")

    plot_initial_graphs()

def center_signal_fine():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # Définition de l'ordre de la méthode de centrage (M) par l'utilisateur, ou None pour estimation automatique
    # popup avec menu déroulant pour choisir l'ordre de centrage (M) ou None pour estimation automatique
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    popup.title(lang["center_fine_kind"])
    # sous-titre d'explication
    ttk.Label(popup, text=lang["center_fine_details"]).pack()
    place_relative(popup, root, 400, 100)
    order_var = tk.StringVar()
    order_var.set("FSK/CPM")
    order_menu = tk.OptionMenu(popup, order_var, "FSK/CPM", "Porteuse", "PSK2", "PSK4/QAM", "PSK8/Qpi4")
    order_menu.pack(pady=10)
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    order = order_var.get()
    try:
        if order == "FSK/CPM":
            M = None
        elif order == "Porteuse":
            M = 1
        # si l'ordre est défini, il est utilisé pour adapter la méthode de centrage
        elif order == "PSK2":
            M = 2
        elif order == "PSK4/QAM":
            M = 4
        elif order == "PSK8/Qpi4":
            M = 8
        iq_sig, center = df.estimate_carrier_weighted(iq_sig, s_rate, M)
    except ValueError:
        if debug is True:
            print("Centrage fin annulé, ordre ou méthode non valide")
        return

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
    # Variation de la fonction précédente, plus efficace sur certains signaux.
    # En général performant sur les signaux de modulation de phase
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
    # ligne en rouge du pic de puissance estimé sur le graphe à -peak_freq & peak_freq, sauf si < 25 Hz
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
    # mesure de la rapidité de modulation par les ordres de puissance,
    # efficace sur les signaux de modulation d'amplitude et de fréquence
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
    f, cyclic_corr_avg, peak_freq = em.cyclic_spectrum_sliding_fft(iq_sig, s_rate, window=window_choice, frame_len=N*2, step=overlap*2)
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

def cepstrum():
    # Cepstre
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["cepstrum"])
    print(lang["cepstrum"])
    if not filepath:
        print(lang["no_file"])
        return
    peak_freq, q, cepstrum = em.cepstral_estimator(iq_sig, s_rate, window=window_choice)
    cepstrum = cepstrum / np.max(np.abs(cepstrum))
    cep_filtered = np.copy(cepstrum)
    cep_filtered[:5] = 0

    mask = q > 0  # avoid division by zero
    freq_axis = 1 / q[mask]
    cep_valid = cep_filtered[mask]
    
    ax = plt.subplot()
    ax.plot(freq_axis,cep_valid)
    if abs(peak_freq) > 25:
        ax.axvline(x=-peak_freq, color='r', linestyle='--')
        ax.axvline(x=peak_freq, color='r', linestyle='--')
        ax.set_title(f"{lang['estim_peak']} {round(peak_freq,2)} Hz")
    else:
        ax.set_title(lang['estim_failed'])
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["cepstrum"])
    ax.set_xlim(0, s_rate / 2)

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del cepstrum, q, peak_freq, canvas

def clock_graph():
    # Affichage de la fréquence d'horloge
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["clock_graph"])
    print(lang["clock_graph"])
    if not filepath:
        print(lang["no_file"])
        return
    _, freq_diff = em.frequency_transitions(iq_sig, s_rate, diff_window, window_choice)
    clock, freqs, mag = dm.estimate_baud_rate(freq_diff, s_rate,debug=debug)
    if clock == 0 :
        tk.messagebox.showinfo(lang["error"], lang["estim_failed"], parent=root)
        return
    if debug is True:
        print("Fréquence de l'horloge estimée: ", clock, " Hz")
    mag = mag / np.max(mag)
    ax = plt.subplot()
    ax.plot(freqs[:len(freqs)//2], mag[:len(mag)//2])
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])
    if abs(clock) > 25:
        ax.axvline(x=clock, color='r', linestyle='--')
        ax.set_title(f"{lang['estim_peak']} {round(clock,2)} Hz")
    else:
        ax.set_title(lang['estim_failed'])
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del clock, freqs, mag, freq_diff, canvas

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
    ax.plot(f, Pxx)
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
    if debug is True:
        print(f"Points avant décimation du graphe Pmax: {len(f)}")
    peak_idx = np.argmax(wav_mag)
    peak_freq = f[peak_idx]
    peak_val = wav_mag[peak_idx]
    f, wav_mag = hf.downsample_for_plotting(f, wav_mag, target_points=4000)
    if debug is True:
        print(f"Points après décimation du graphe Pmax: {len(f)}")
    ax.plot(f, wav_mag)
    ax.plot(peak_freq, peak_val, 'rx') # max
    ax.grid()
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"])

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas, wav_mag, f

def correct_constellation_rotation():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text, iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # demander si QAM ou PSK et l'ordre
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 400, 150)
    popup.title(lang["rotation_correction"])
    modulation_type = tk.StringVar()
    modulation_type.set("PSK")
    ttk.Radiobutton(popup, text="PSK2", variable=modulation_type, value="PSK2").pack()
    ttk.Radiobutton(popup, text="QAM", variable=modulation_type, value="QAM").pack()
    ttk.Radiobutton(popup, text="PSK4", variable=modulation_type, value="PSK4").pack()
    ttk.Radiobutton(popup, text="PSK8/Qpi4", variable=modulation_type, value="PSK8/Qpi4").pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    mod_type = modulation_type.get()
    if mod_type == "PSK2":
        order = 2 ; mod_type = "PSK"
    elif mod_type == "PSK4":
        order = 4 ; mod_type = "PSK"
    elif mod_type == "PSK8/Qpi4":
        order = 8 ; mod_type = "PSK"
    elif mod_type == "QAM":
        order = 16 ; mod_type = "QAM"
    elif mod_type not in ["PSK2", "PSK4", "PSK8/Qpi4", "QAM"]:
        if debug is True:
            print("Correction de rotation de constellation annulée, type de modulation non valide")
        return
    clear_plot()
    iq_sig, theta = df.correct_constellation_rotation(iq_sig, int(order), mod_type)
    if debug is True:
        print(f"Correction de rotation de constellation appliquée avec un ordre de {order}")
        print(f"et une rotation de {theta} rad : ({np.degrees(theta)}°).")
    constellation()
    ax.set_title(f"{lang['rotation_corrected']} : {np.round(np.degrees(theta),2):.2f}°")

def constellation():
    # Affichage de la constellation du signal.
    # Fortement dépendant d'un bon calibrage de la fréquence centrale
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
    if len(iq_constel) > 1000:
        grid_size = int(np.sqrt(len(iq_constel) // 2))  # Ajuster la taille de la grille en fonction du nombre de points
    else:
        grid_size = 200
    # selon paramétrage : affichage de la constellation en hexbin (par défaut) ou en scatter
    if constel_hexbins is True:
        ax.hexbin(np.real(iq_constel), np.imag(iq_constel), gridsize=grid_size, cmap='jet', bins='log', mincnt=1)
    else:
        ax.scatter(np.real(iq_constel), np.imag(iq_constel), s=5)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas

def constellation_synchronized():
    # Affichage de la constellation du signal synchronisée sur le débit symbole
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["constellation_sync"])
    print(lang["constellation_sync"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    iq_constel = iq_sig/np.max(np.abs(iq_sig))
    symbol_rate = tk.simpledialog.askstring(lang["constellation_sync"], lang["symbol_rate"], parent=root)
    if symbol_rate is None or symbol_rate == "" or float(symbol_rate) <= 0:
        if debug is True:
            print("Constellation synchronisée non affichée, valeur non définie")
        return
    symbol_rate = float(symbol_rate)
    pts = dm.constel_points(iq_constel, s_rate, symbol_rate)
    lim = np.percentile(np.abs(pts), 99.9) * 1.1
    if constel_hexbins:
        g = int(np.clip(np.sqrt(len(pts)) / 4, 30, 100))
        hb = ax.hexbin(pts.real, pts.imag, gridsize=g, extent=(-lim, lim, -lim, lim),
                    cmap='jet', bins='log', mincnt=1)
        fig.colorbar(hb, ax=ax, label="log10(N)")
    else:
        if len(pts) > 200000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 200000, replace=False)]
        ax.scatter(pts.real, pts.imag, s=3, alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{symbol_rate} Bd")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas

def signal_power():
    # Affichage du signal à puissance N
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["signal_power"])
    print(lang["signal_power"])
    if not filepath:
        print(lang["no_file"])
        return
    ax = plt.subplot()
    # demander à l'utilisateur l'ordre de la puissance
    req_order = tk.simpledialog.askstring(lang["signal_power"], lang["power_order"], parent=root)
    if req_order is None:
        return
    f, sig_pow, c_residue = em.power_signal(iq_sig,s_rate,int(req_order))
    ax.plot(f, sig_pow)
    ax.set_xlabel(f"{lang['freq_xy']} [Hz]")
    ax.set_ylabel(lang["norm_power"] + f" ^{req_order}")
    if c_residue is not None:
        ax.axvline(x=c_residue*int(req_order), color='r', linestyle='--')
        ax.set_title(f"{lang['residue_estimate']}")
        # ajouter une information : le pic de résidu n'est valable que s'il est central.
        ax.text(c_residue*int(req_order), ax.get_ylim()[1]*0.9, f"{lang['residue_peak']} {round(c_residue,3)} Hz", color='r', ha='center')
        if debug is True:
            print("Résidu de porteuse trouvé à ", round(c_residue,3), " Hz")

    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del f, sig_pow, c_residue, canvas

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
    scf_alpha_step = s_rate/2000
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
    ax._custom_time = time
    ax._custom_data = phase_diff
    time, phase_diff = hf.downsample_for_plotting(time, phase_diff, target_points=4000)
    ax.plot(time, phase_diff)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel(f"{lang['diff_phase']} [rad]")
    ax.set_title(f"{lang['smoothing']} {diff_window}")
    ax.grid(True)
    ax.callbacks.connect('xlim_changed', update_on_zoom)
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
    ax._custom_time = time
    ax._custom_data = freq_diff
    time, freq_diff = hf.downsample_for_plotting(time, freq_diff, target_points=4000)
    ax.plot(time, freq_diff)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel(f"{lang['freq_xy']} [Hz]")
    ax.set_title(f"{lang['smoothing']} {diff_window}")
    ax.grid(True)
    ax.callbacks.connect('xlim_changed', update_on_zoom)
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
    hist, bins = em.phase_cumulative_distribution(iq_sig, s_rate, num_bins=hist_bins, window_size=diff_window, window_type=window_choice)
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
    # fonc de distribution de la fréquence instantanée. A évaluer/améliorer
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
    if debug is True:
        print("Génération de la transformée en ondelettes de Morlet")
    ax = plt.subplot()
    power, fc, t, coi = df.morlet_scalogram(iq_sig, s_rate, nfreq=morlet_nfreq, w=morlet_fc,density=True)
    pdb = 10 * np.log10(np.maximum(power, 1e-30))
    im = ax.pcolormesh(t, fc, pdb, shading='nearest', cmap='inferno',
                   vmin=pdb.max() - 60, vmax=pdb.max())
    ax.set_yscale('symlog', linthresh=np.abs(fc).min()) # log de chaque côté de 0
    T = t[-1] + t[0]
    ax.fill_betweenx(fc, 0, coi, color='w', alpha=0.5) # cône d'influence
    ax.fill_betweenx(fc, T - coi, T, color='w', alpha=0.5)
    ax.set_xlim(0, T)
    ax.set_xlabel(f"{lang['time_xy']} [s]")
    ax.set_ylabel(f"{lang['freq_xy']} [Hz]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(lang["power_scale"])
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del canvas

# OFDM
def alpha_from_symbol():
    global Tu, toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    # calcul de alpha frq à partir de la durée symbole OFDM (Tu),
    # estimée par l'utilisateur avec la fonction d'autocorrélation.
    # Demander à l'utilisateur de rentrer la durée estimée de Tu
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
    ttk.Label(popup, text=f"{lang['estim_bw']} :{round(bw,2)} Hz").pack()
    new_bw = tk.StringVar()
    ttk.Label(popup, text=f"{lang['unreliable_sscarrier']}\n{lang['redef_bw'] }").pack()
    ttk.Entry(popup, textvariable=new_bw).pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
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
    ttk.Label(popup, text=f"{lang['tu']} = {Tu:.6f} ms").pack()
    ttk.Label(popup, text=f"{lang['tg']} = {Tg:.6f} ms").pack()
    ttk.Label(popup, text=f"{lang['ts']} = {Ts:.6f} ms").pack()
    ttk.Label(popup, text=f"{lang['df']} = {Df:.6f} Hz").pack()
    ttk.Label(popup, text=f"{lang['num_ssp']} = {num}").pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()

    clear_plot()
    # afficher les sous-porteuses OFDM sur la DSP du signal avec la bande passante estimée,
    # qui sert de fenêtre dans laquelle les sous-porteuses sont affichées
    fig = plt.figure()
    fig.suptitle(lang["dsp"])
    ax = fig.add_subplot()
    ax.plot(f, Pxx)
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

# Experimental : bispectre
def bispectrum_plot():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    clear_plot()
    fig = plt.figure()
    fig.suptitle(lang["bispectrum_desc"])
    if not filepath:
        print(lang["no_file"])
        return
    if not tk.messagebox.askokcancel(lang["bispectrum_desc"], lang["confirm_wait"], parent=root):
        return
    ax = plt.subplot()
    # Génération du bispectre (NFFT=X crée une image de taille X*X)
    f, B_mag = em.compute_bispectrum(iq_sig, s_rate, nfft=N//2, overlap=N//4, window_type=window_choice)
    if B_mag is None:
        if debug is True:
            print("Echec de la génération du bispectre")
        return
    # Normalisation du bispectre pour l'affichage
    B_mag_norm = B_mag / np.max(B_mag)
    # Le plan de symétrie devrait montrer une ligne diagonale à 45 degrés de couplage de phase.
    im = ax.imshow(B_mag_norm, aspect='auto', extent=[s_rate/-2, s_rate/2, s_rate/-2, s_rate/2], cmap=cm.jet, origin='lower')
    ax.set_xlabel(lang["freq_xy"] + " $f_1$ [Hz]")
    ax.set_ylabel(lang["freq_xy"] + " $f_2$ [Hz]")
    fig.colorbar(im, ax=ax, label=lang["norm_power"])
    ax.grid(True, alpha=0.3)
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    del f, B_mag, B_mag_norm, im, canvas

## Fonctions de gestion des curseurs
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
    peaks = hf.find_peaks(y_data)
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
        dist, dx, dy = hf.calculate_distance(cursor_points[0], cursor_points[1])
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
            dist, dx, dy = hf.calculate_distance(cursor_points[0], cursor_points[1])
            # Màj ou crée texte distances
            if distance_text:
                distance_text.remove()
            if type(ax) is not tuple:
                # ignore "dist" : distance en diagonale pas pertinente ici
                distance_text = ax.text(0.05, 0.95, f"ΔX: {dx:.6f}, ΔY: {dy:.6f}", 
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

 # Fonction de sauvegarde des coordonnées des points cliqués dans une fenêtre dédiée
def open_coordinates_window():
    global cursor_points, cursor_lines, distance_text
    # sans effet si pas de fichier chargé
    if not filepath:
        print(lang["no_file"])
        return
    coord_window = tk.Toplevel()
    place_relative(coord_window, root, 300, 200)
    coord_window.title(lang["coord_window"])
    text_area = tk.Text(coord_window, wrap=tk.NONE)
    text_area.pack(fill=tk.BOTH, expand=True)
    scrollbar_y = tk.Scrollbar(coord_window, orient=tk.VERTICAL, command=text_area.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    text_area.config(yscrollcommand=scrollbar_y.set)
    # Rappel pour enregistrer les coordonnées des clics dans la zone de texte
    def on_click_save_coords(event):
        if event.inaxes:
            coord_text = f"({event.xdata:.6f}, {event.ydata:.6f})\n"
            text_area.insert(tk.END, coord_text)
    # Connecte l'événement de clic à la fonction d'enregistrement des coordonnées
    click_event_id = fig.canvas.mpl_connect('button_press_event', on_click_save_coords)
    # Ferme la fenêtre de coordonnées et désactive le mode curseur
    def close_window():
        fig.canvas.mpl_disconnect(click_event_id)
        toggle_cursor_mode()
        coord_window.destroy()
    coord_window.protocol("WM_DELETE_WINDOW", close_window)
    # Si le mode curseur désactivé, l'active pour permettre de cliquer sur le graphe
    if not cursor_mode:
        toggle_cursor_mode()
    # Empêche de désactiver le mode curseur tant que la fenêtre de coordonnées est ouverte
    def disable_cursor_toggle():
        mode_button.config(state=tk.DISABLED)
    def enable_cursor_toggle():
        mode_button.config(state=tk.NORMAL)
    mode_button.config(state=tk.DISABLED)
    coord_window.bind("<Destroy>", lambda event: enable_cursor_toggle())
    # Nettoie pour éviter les conflits avec les autres fonctions de curseurs
    clear_cursors()

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

# fonc de modif du signal après sélection d'une zone par curseurs.
def modify_signal_in_zone():
    global iq_sig, s_rate, cursor_points, cursor_lines, distance_text, N, overlap
    if len(cursor_points) == 2 :
        # déplace la fréquence centrale du signal à 0 Hz en fonction de la position du curseur (moyenne des 2 points)
        fcenter = cursor_points[0][0]/2 + cursor_points[1][0]/2
        iq_sig = iq_sig * np.exp(-1j*2*np.pi*fcenter*np.arange(len(iq_sig))/s_rate)
        new_fs = int(abs(cursor_points[0][0] - cursor_points[1][0])*2) # nouvelle fréquence d'échantillonnage = 2x la distance en x entre les 2 points
        iq_sig = sm.low_high_pass_filter(iq_sig, max(abs(cursor_points[0][0]), abs(cursor_points[1][0]))-abs(fcenter), s_rate, filter_order, filter_kind,'low')
        # le filtrage coupe les composantes au-delà de la moitié de la nouvelle fréquence d'échantillonnage, centrée sur 0 Hz
    else: # si pas 2 points, affiche message d'erreur et ne modifie rien
        tk.messagebox.showinfo(lang["error"], lang["2pt_cursors"], parent=root)
        if debug is True:
            print("Deux points de curseurs doivent être définis pour modifier le signal")
        return
    new_fs = int(new_fs)
    if s_rate > new_fs: # modifie la fréquence d'échantillonnage uniquement si inférieure à l'actuelle
        iq_sig, s_rate = sm.resample_polyphase(iq_sig, s_rate, new_fs)
    else:
        pass
    # ajuste les paramètres de la FFT en fonction de la nouvelle fréquence d'échantillonnage et durée du signal (idem load classique)
    if len(iq_sig) > 1e6: # si plus d'un million d'échantillons
        N = 4096
    elif 1e5 < len(iq_sig) < 1e6 : # si entre 100k et 1 million d'échantillons
        N = 1024
    elif len(iq_sig) < s_rate: # si moins d'une seconde
        # base de résolution = 25 Hz par défaut, dégradation proportionnelle à la durée si < 1 seconde
        N = (s_rate//25)*(len(iq_sig)/s_rate) 
        N = (int(N/2))*2 # N pair de préférence
        if N < 4: # taille minimum de 4 échantillons
            N = 4
    elif len(iq_sig) < 5e3:
        N = 128 # taille de fenêtre FFT pour les signaux > 1 sec mais < 5k échantillons
    else:
        N = 512 # taille de fenêtre FFT par défaut
    overlap = N//overlap_value
    # coupe le signal (en durée) entre les 2 points sélectionnés
    cut_signal_cursors()
    # la fonction de découpe envoie aussi les graphes de base à jour, donc pas besoin de les refaire ici
    if debug is True:
        print("Signal modifié entre les points de curseurs, nouvelle fréquence d'échantillonnage : ", s_rate)
        print("N ajusté à ", N, " overlap à ", overlap)
        print("Nombre d'échantillons du signal modifié : ", len(iq_sig))
        print("Durée du signal modifié : ", len(iq_sig)/s_rate, " secondes")

# sauvegarde le signal (modifié ou non) en nouveau fichier wav
def save_as_wav():
    global iq_sig, s_rate
    filename = tk.filedialog.asksaveasfilename(title=lang["save_wav"],
        defaultextension=".wav", filetypes=[("Waveform Audio File", "*.wav"), ("All Files", "*.*")])
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
        print("Ecriture du nouveau wav stereo : ", filename)

def save_as_mono_wav():
    global iq_sig, s_rate
    filename = tk.filedialog.asksaveasfilename(title=lang["save_wav"],
        defaultextension=".wav",filetypes=[("Waveform Audio File", "*.wav"), ("All Files", "*.*")])
    # Conversion en mono réel
    iq_sig, s_rate = sm.upsample(iq_sig, s_rate, 2)
    t = np.arange(len(iq_sig)) / s_rate
    reconstructed = iq_sig * np.exp(1j * 2*np.pi*(s_rate//4)*t)
    mono = np.real(reconstructed)
    # Normalise
    max_amplitude = np.max(np.abs(mono))
    if max_amplitude > 0:
        mono = mono / max_amplitude
    # Formate en 16 bits
    mono_int16 = (mono * 32767).astype(np.int16)
    # Enregistre en mono
    wav.write(filename, s_rate, mono_int16)
    if debug:
        print(f"Ecriture du nouveau wav mono : {filename}")

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
            if modulation_type.get() == lang["demod_psk"]:
                param_pi4.config(state=tk.NORMAL)
        else:
            mapping_nat.config(state=tk.DISABLED)
            mapping_gray.config(state=tk.DISABLED)
            # mapping_custom.config(state=tk.DISABLED)
            param_pi4.config(state=tk.DISABLED)
    def toggle_diff_offset():
        # seulement si PSK
        if modulation_type.get() == lang["demod_psk"]:
            param_diff.config(state=tk.NORMAL)
            param_offset.config(state=tk.NORMAL)
            if param_order.get() == lang["param_order4"]:
                param_pi4.config(state=tk.NORMAL)
        else:
            param_diff.config(state=tk.DISABLED)
            param_offset.config(state=tk.DISABLED)
            param_pi4.config(state=tk.DISABLED)
            if param_order.get() != lang["param_order4"]:
                param_pi4.config(state=tk.DISABLED)
    def toggle_pi4():
        if use_pi4.get():
            param_diff.config(state=tk.DISABLED)
            param_offset.config(state=tk.DISABLED)
            param_mapping.set(lang["mapping_gray"])
        else:
            param_diff.config(state=tk.NORMAL)
            param_offset.config(state=tk.NORMAL)

    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 350, 350)
    popup.title(lang["demod_param"])
    modulation_type = tk.StringVar()
    ttk.Label(popup, text=lang["demod_type"]).pack()
    ttk.Radiobutton(popup, text=lang["demod_fsk"], variable=modulation_type, value=lang["demod_fsk"], command=toggle_diff_offset).pack()
    ttk.Radiobutton(popup, text=lang["demod_psk"], variable=modulation_type, value=lang["demod_psk"], command=toggle_diff_offset).pack()
    modulation_type.set(lang["demod_fsk"])

    param_order = tk.StringVar()
    ttk.Label(popup, text=lang["param_order"]).pack()
    ttk.Radiobutton(popup, text=lang["param_order2"], variable=param_order, value=lang["param_order2"], command=toggle_mapping).pack()
    ttk.Radiobutton(popup, text=lang["param_order4"], variable=param_order, value=lang["param_order4"], command=toggle_mapping).pack()
    param_order.set(lang["param_order2"])
    target_rate = tk.StringVar()
    ttk.Label(popup, text=lang["demod_speed"]).pack()
    ttk.Entry(popup, textvariable=target_rate).pack()
    param_mapping = tk.StringVar()
    param_mapping.set(lang["mapping"])
    mapping_nat = ttk.Radiobutton(popup, text=lang["mapping_nat"], variable=param_mapping, value=lang["mapping_nat"], state=tk.DISABLED)
    mapping_nat.pack()
    mapping_gray = ttk.Radiobutton(popup, text=lang["mapping_gray"], variable=param_mapping, value=lang["mapping_gray"], state=tk.DISABLED)
    mapping_gray.pack()
    # mapping_custom = ttk.Radiobutton(popup, text=lang["mapping_custom"], variable=param_mapping, value=lang["mapping_custom"], state=tk.DISABLED)
    # mapping_custom.pack()
    use_diff = tk.BooleanVar(value=False)
    use_offset = tk.BooleanVar(value=False)
    use_pi4 = tk.BooleanVar(value=False) 
    param_diff = ttk.Checkbutton(popup, text=lang["param_diff"], variable=use_diff, state=tk.DISABLED)
    param_diff.pack()
    param_offset = ttk.Checkbutton(popup, text=lang["param_offset"], variable=use_offset, state=tk.DISABLED)
    param_offset.pack()
    param_pi4 = ttk.Checkbutton(popup, text="π/4-DQPSK", variable=use_pi4, state=tk.DISABLED, command=toggle_pi4)
    param_pi4.pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
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
        pi4 = use_pi4.get()
        try:
            if target_rate is not None:
                clock = target_rate
            else:
                clock,_,_ = dm.estimate_baud_rate(diff, s_rate)
            bits, lock_score = dm.psk_demodulate(iq_sig, s_rate, clock, order, gray=gray, 
                                     differential=differential, offset=offset, pi4=pi4, 
                                     costas_damping=costas_damping, costas_bw_factor=costas_bw_factor)
            if pi4:
                alt_psk = "π/4-D"
            elif differential and not offset:
                alt_psk = "D"
            elif offset and not differential:
                alt_psk = "O"
            elif differential and offset:
                alt_psk = "DO"
            else:
                alt_psk = ""
            if debug is True :
                print(f"Démodulation {alt_psk}{mod_type}{order} réalisée avec mapping {mapping}, rapidité {clock} bauds, bits: {len(bits)}")
                print(f"Score de verrouillage : {round(lock_score*100,2)}%") 
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
                    print(f"Démodulation {mod_type} {order} réalisée avec mapping {mapping},")
                    print(f"rapidité {clock} bauds, bits: {len(bits)}") 
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
            text_output= f"{alt_psk}{mod_type}{order} {clock} bauds, {mapping} mapping. {lang['estim_bits']} : {len(bits)}. {lang['lock_score']}: {round(lock_score*100,2)}% \n"
        elif mod_type == "CPM/FSK" and order == 2:
            text_output= f"{mod_type}{order}. {lang['clock_frequency']} {clock} Hz. {lang['estim_bits']} : {len(bits)}. \n"
        elif mod_type == "PSK" and order == 2:
            text_output= f"{alt_psk}{mod_type}{order} {clock} bauds. {lang['estim_bits']} : {len(bits)}. {lang['lock_score']}: {round(lock_score*100,2)}% \n"
        # lignes de bits
        max_display = 50000  # nombre max de bits à afficher
        display_bits = bits[:max_display]
        formatted_bits = "".join(map(str, display_bits))
        if len(bits) >= max_display:
            formatted_bits += f"\n... ({len(bits) - max_display} {lang['more_bits']})"
            file = open(f'demod_results_{mod_type}_{order}_{clock}.txt','a+')
            output_bits = "".join(map(str, bits))
            file.write(output_bits)
            file.close()
            tk.messagebox.showinfo("Trop grand nombre de bits",f"{len(bits)} enregistrés dans un fichier 'demod_results_{mod_type}_{order}_{clock}.txt'", parent=root)
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

def demod_ssb():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    # demande choix LSB/USB
    ssb_choice = tk.StringVar()
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 400, 100)
    popup.title(lang["ssb_choice"])
    ttk.Radiobutton(popup, text="LSB", variable=ssb_choice, value="LSB").pack()
    ttk.Radiobutton(popup, text="USB", variable=ssb_choice, value="USB").pack()
    ssb_choice.set("None")
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
    popup.wait_window()
    # demod ssb
    try:
        iq_sig = dm.ssb_to_iq(iq_sig, s_rate, ssb_choice.get())
        if debug is True:
            print(f"Démodulation {ssb_choice.get()} réalisée")
    except:
        if debug is True:
            print(f"Echec de démodulation SSB : {ssb_choice.get()}")
        return
    plot_other_graphs()

def demod_mfsk():
    global toolbar, ax, fig, cursor_points, cursor_lines, distance_text
    time, freq_diff = em.frequency_transitions(iq_sig, s_rate, diff_window, window_choice)
    freq_diff /= np.max(np.abs(freq_diff))
    # vars pour fonctions de démod
    target_rate = None
    mapping = None
    tone_freqs = None
    spacing = None # espacement entre les symboles. Pour l'instant pas utilisé
    # on demande rapidité, ordre, espacement et mapping
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 500, 275)
    popup.title(lang["demod_param"])
    param_method = tk.StringVar()
    ttk.Radiobutton(popup, text=lang["mfsk_discrete_diff"], variable=param_method, value="nrz").pack()
    ttk.Radiobutton(popup, text=lang["mfsk_tone_detection"], variable=param_method, value="alt").pack()
    param_method.set("alt")
    param_order = tk.StringVar()
    ttk.Label(popup, text=lang["param_order"]).pack()
    ttk.Entry(popup, textvariable=param_order).pack()
    target_rate = tk.StringVar()
    ttk.Label(popup, text=lang["demod_speed"]).pack()
    ttk.Entry(popup, textvariable=target_rate).pack()
    param_mapping = tk.StringVar()
    param_mapping.set(lang["mapping"])
    mapping_nat = ttk.Radiobutton(popup, text=lang["mapping_nat"], variable=param_mapping, value=lang["mapping_nat"])
    mapping_nat.pack()
    mapping_gray = ttk.Radiobutton(popup, text=lang["mapping_gray"], variable=param_mapping, value=lang["mapping_gray"])
    mapping_gray.pack()
    mapping_custom = ttk.Radiobutton(popup, text=lang["mapping_non-binary"], variable=param_mapping, value=lang["mapping_non-binary"])
    mapping_custom.pack()
    param_mapping.set(lang["mapping_nat"])
    freq_tones = tk.StringVar()
    options_tones = [lang["auto_detect"], lang["manual_select"]]
    options_tones_menu = ttk.OptionMenu(popup, freq_tones, *options_tones)
    freq_tones.set(lang["auto_detect"])
    options_tones_menu.pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()

    def toggle_tone_selection():
        # tone detection seulement si méthode alternative sélectionnée
        if param_method.get() == "alt":
            options_tones_menu.config(state=tk.NORMAL)
        else:            
            options_tones_menu.config(state=tk.DISABLED)
    param_method.trace_add("write", lambda *args: toggle_tone_selection())

    popup.wait_window()
    if target_rate.get() == "":
        if debug is True:
            print("Rapidité de démodulation non définie.")
        return
    elif float(target_rate.get()) < 1:
        print("Pas de rapidité de démodulation définie. Essai aveugle")
        target_rate = None
        clock,_,_ = dm.estimate_baud_rate(freq_diff, s_rate, target_rate, precision, debug)
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
        if param_method.get() == "nrz":
            symbols, clock = dm.wpcr(freq_diff, s_rate, clock, tau, precision, debug)
            if len(symbols) > 2:
                bits = dm.slice_mfsk(symbols, int(param_order.get()), mapping, return_format)
                if debug is True:
                    print(f"Démodulation MFSK réalisée avec mapping {mapping}, rapidité {clock} bauds, bits: {len(bits)}")
            else:
                bits = 0
        elif param_method.get() == "alt":
            # si auto_detect, on laisse la fonction trouver les fréquences de tons automatiquement. 
            # Sinon, saisie utilisateur et passe à la fonction de détection de tons pour
            # se concentrer sur ces fréquences spécifiques.
            if freq_tones.get() == lang["manual_select"]:
                # on demande à l'utilisateur de saisir les fréquences de tons séparées par des virgules
                popup_manual_freqs = tk.Toplevel(root)
                popup_manual_freqs.bind("<Return>", lambda event: popup_manual_freqs.destroy())
                place_relative(popup_manual_freqs, root, 500, 100)
                popup_manual_freqs.title(lang["manual_select"])
                ttk.Label(popup_manual_freqs, text=lang["enter_freq_tones"]).pack()
                manual_freqs_var = tk.StringVar()
                ttk.Entry(popup_manual_freqs, textvariable=manual_freqs_var, width=75).pack()
                ttk.Button(popup_manual_freqs, text="OK", command=popup_manual_freqs.destroy).pack()
                popup_manual_freqs.wait_window()
                manual_freqs = manual_freqs_var.get()
                try:
                    tone_freqs_manual = np.array([float(f.strip()) for f in manual_freqs.split(",")])
                except:
                    if debug is True:
                        print("Fréquences de tons manuelles invalides. Revenir à la détection automatique.")
                    tone_freqs_manual = None
            else:                
                tone_freqs_manual = None
            # Détection de tons par clustering pour MFSK, 
            # plus robuste que la méthode discrète sur le signal différentiel
            tone_freqs, t, tone_idx, tone_freq, tone_powers, clock = dm.detect_and_track_mfsk_auto(
                iq_sig, s_rate, clock, order, mfsk_tresh_db, mfsk_peak_prom_db, mfsk_win_factor, 
                mfsk_hop_factor, mfsk_bin_width_cluster_factor, mfsk_viterbi_penalty, tone_freqs_manual)     
            if debug is True:
                print(f"Fréquences de tons MFSK détectées : {tone_freqs}")
                if len(tone_freqs) < order:
                    print(f"Attention : seulement {len(tone_freqs)} tons détectés sur {order} attendus."
                    f" Espacement des tons probablement trop fin pour la résolution actuelle.")
            frames_per_symbol = (1.0 / clock) / np.mean(np.diff(t))
            symbols, phase0 = dm.majority_vote_resample(tone_idx, frames_per_symbol, len(tone_freqs))
            if len(symbols) > 2:
                bits = dm.indices_to_output(symbols, order, mapping, return_format)
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
            # charset récupéré de la fonction demod.slice_mfsk
            charset = string.digits + string.ascii_uppercase + string.ascii_lowercase + string.punctuation 
            charmap = {ch: i for i, ch in enumerate(charset)}
            bits_plot = np.array([charmap[ch] for ch in bits if ch in charmap])
            ax.set_yticks(range(len(charset)))
            ax.set_yticklabels(list(charset))
        else:
            bits_plot = bits
            ax.set_yticks(range(order))
            ax.set_yticklabels(range(order))

        if len(bits) > 5000:
            display_bits = bits_plot[:5000]
            fig.suptitle(f"{lang['estim_bits']} {lang['short_bits']}")
        else:
            display_bits = bits_plot
        ax.plot(display_bits, "o-")
        ax.set_xlabel("Bits")
        ax.set_ylabel(lang["bits_value"])
    except Exception as e:
        clear_plot()
        fig = plt.figure()
        fig.suptitle(lang["bits_fail"])
        ax = plt.subplot()
        ax.plot(0)
        if debug is True:
            print(f"Echec de démodulation MFSK : {e}")
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
        if param_method.get() == "nrz":
            text_output = f"MFSK{order}. {lang['clock_frequency']} {clock} Hz, {mapping} mapping. {lang['estim_bits']} : {len(bits)}. \n"
        elif param_method.get() == "alt":
            tone_freqs_display = [f"{f:.2f}Hz" for f in tone_freqs]
            text_output= f"MFSK{order} {clock} bauds, {mapping} mapping. {lang['estim_bits']} : {len(bits)}.  \
                \n{lang['mfsk_tones_detected']} {', '.join(map(str, tone_freqs_display))}.\n"
        # lignes de bits
        max_display = 50000  # nombre max de bits à afficher
        display_bits = bits[:max_display]
        if return_format == "int": # si format int, on sépare par une virgule chaque symbole
            formatted_bits = ",".join(map(str, display_bits))
        else:
            formatted_bits = "".join(map(str, display_bits))
        if len(bits) >= max_display:
            formatted_bits += f"\n... ({len(bits) - max_display} {lang['more_bits']})"
            file = open(f'demod_results_MFSK_{order}_{clock}.txt','a+')
            if return_format == "int":
                output_bits = ",".join(map(str, bits))
            else:
                output_bits = "".join(map(str, bits))
            file.write(output_bits)
            file.close()
            tk.messagebox.showinfo("Trop grand nombre de bits",f"{len(bits)} enregistrés dans un fichier 'demod_results_MFSK_{order}_{clock}.txt'", parent=root)
        text_output += formatted_bits
    except:
        bits, bits_plot, formatted_bits, display_bits = "", "", "", ""
        if tone_freqs is not None:
            text_output = f"{lang['bits_fail']}. {lang['mfsk_tones_detected']} {', '.join(map(str, tone_freqs))}."
        else:
            text_output = lang["bits_fail"]
    text_box.insert(tk.END, text_output)
    text_box.config(state=tk.DISABLED)
    del canvas, time, freq_diff, bits, bits_plot, formatted_bits, display_bits, text_output, text_box

# Filtres supplémentaires
def apply_median_filter():
    global iq_sig, s_rate
    if not filepath:
        print(lang["no_file"])
        return
    # Demande la taille du filtre. 4 options listées :
    # Léger pour 3, Modéré pour 5, Agressif pour 9, sinon personnalisé
    popup = tk.Toplevel()
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 300, 150)
    popup.title(lang["median_filter"])
    ttk.Label(popup, text=lang["kernel_select"]).pack()
    kernel_size = tk.StringVar()
    options = [lang["light_filter"], lang["medium_filter"], lang["aggressive_filter"], lang["dynamic_filter"], "Custom"]
    kernel_size.set("Custom")  # Valeur par défaut
    kernel_size_menu = ttk.OptionMenu(popup, kernel_size, *options)
    kernel_size_menu.pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
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

    # Demande la taille du filtre. 4 options listées :
    # Léger pour 3, Modéré pour 5, Agressif pour 9, sinon personnalisé
    popup = tk.Toplevel()
    popup.title(lang["moving_average"])
    popup.bind("<Return>", lambda event: popup.destroy())
    place_relative(popup, root, 300, 150)
    ttk.Label(popup, text=lang["window_size_select"]).pack()
    window_size = tk.StringVar()
    options = [lang["light_filter"], lang["medium_filter"], lang["aggressive_filter"], lang["dynamic_filter"], "Custom"]
    window_size.set("Custom")  # Valeur par défaut
    window_size_menu = ttk.OptionMenu(popup, window_size, *options)
    window_size_menu.pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
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
    win_type = tk.simpledialog.askstring(lang["fir_filter"], lang["window_choice"], initialvalue="hamming", parent=root)
    if cutoff_freq is None or num_taps is None:
        return
    try:
        iq_sig = sm.fir_filter(iq_sig, s_rate, cutoff_freq, 'lowpass', num_taps, window=win_type)
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

def apply_spectral_denoise():
    global iq_sig
    if not filepath:
        print(lang["no_file"])
        return
    try:
        iq_sig = sm.spectral_soft_denoise(iq_sig, alpha_noise)
        if debug is True:
            print(f"Débruitage spectral appliqué avec un seuil de {alpha_noise} fois le niveau de bruit estimé")
    except Exception as e:
        if debug:
            print(f"Erreur lors de l'application du débruitage spectral: {e}")
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
    ttk.Label(popup, text=lang["pulse_shape"]).pack()
    pulse_shape_var = tk.StringVar()
    options = ['rectangular', 'gaussian', 'raised_cosine', 'root_raised_cosine', 'sinc', 'rsinc']
    pulse_shape_var.set('root_raised_cosine')  # default
    pulse_shape_menu = ttk.OptionMenu(popup, pulse_shape_var, *options)
    pulse_shape_menu.pack()
    ttk.Button(popup, text="OK", command=popup.destroy).pack()
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
            symbol_rate,_,_ = dm.estimate_baud_rate(freq_diff, s_rate, target_rate=baud_rate, precision=precision, debug=debug)
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
    mod_order = tk.simpledialog.askinteger(lang["eye_diagram"], lang["eye_order"], minvalue=2, parent=root)
    if mod_order is None or mod_order < 2:
        if debug is True:
            print("Ordre de modulation non défini ou invalide.")
        return
    try:
        time, traces, lock = dm.eye_diagram(iq_sig, s_rate, symbol_rate, mod_order, eye_channel, eye_num_traces, eye_symbols, costas_bw_factor)
        if debug is True:
            print("Diagramme de l'oeil calculé. Score de verrouillage :",round(lock*100,2),"%")
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
    except Exception as e:
        clear_plot()
        time, traces, lock = None, None, None
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
    del canvas, traces, time

def shift_frequency():
    global iq_sig, s_rate
    # décale la fréquence centrale de la moitié de fréquence d'échantillonnage :
    # prend en compte les fichiers encodés "à l'envers"
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

    play_button = ttk.Button(plot_frame, text="Play", command=toggle_playback)
    play_button.pack(side=tk.LEFT, padx=5, pady=5)

    stop_button = ttk.Button(plot_frame, text="Stop", command=stop_playback)
    stop_button.pack(side=tk.LEFT, padx=5, pady=5)

    progress = ttk.Progressbar(plot_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress.pack(side=tk.LEFT, padx=5, pady=5)
    update_progress_bar()

# expérimental : lora score
def lora_score():
    global iq_sig, s_rate
    detection = ""
    estim_bw,_,_,_,_ = mg.estimate_bandwidth(iq_sig, s_rate, N, overlap, window_choice)
    for bw in [125e3/16, 125e3/12, 125e3/8, 125e3/6, 125e3/4, 
               125e3/3, 125e3/2, 125e3, 250e3, 500e3,
               52e6/256, 52e6/128, 52e6/64, 52e6/32]:
        if bw > s_rate:
            continue
        elif bw < estim_bw/3 or bw > estim_bw*3:
            continue
        for sf in range(5,13):
            score=round(df.lora_score(iq_sig, s_rate, bw, sf)/10,2)
            if debug is True and score > 10:
                print(lang["lora_detect"].format(bw, sf, score))
            if score > 20:
                detection += lang["lora_detect"].format(bw, sf, score)
                detection += "\n"
    if detection != "":
        print(detection)
        popup = tk.Toplevel()
        popup.title("LoRa Score")
        place_relative(popup, root, 600, 200)
        ttk.Label(popup, text=detection).pack()
        ttk.Button(popup, text="OK", command=popup.destroy).pack()
    else:
        if debug is True:
            print(lang["lora_detect_no_signal"])
        popup = tk.Toplevel()
        popup.title("LoRa Score")
        place_relative(popup, root, 600, 200)
        ttk.Label(popup, text=lang["lora_detect_no_signal"]).pack()
        ttk.Button(popup, text="OK", command=popup.destroy).pack()

def advanced_settings():
    # Distance min pour fc et nfreq pour CWT Morlet,
    # (db_thresh,win_factor,hop_factor,peak_prominence,cluster_bin_width_factor,viteri_penalty) pour détection de tons MFSK,
    # eye diagram (num_traces, channel, symbols_per_trace), costas loop (zeta damping_factor,loop_bw_factor),
    global filter_order, peak_prominence, acf_min_distance, tau_modifier, precision, morlet_fc, morlet_nfreq
    global scf_alpha_step, costas_damping, costas_bw_factor, eye_channel, eye_num_traces, eye_symbols
    global mfsk_tresh_db, mfsk_peak_prom_db, mfsk_win_factor, mfsk_viterbi_penalty, mfsk_bin_width_cluster_factor, mfsk_hop_factor
    global alpha_noise, filter_kind, constel_hexbins, decim_spec
    # Popup pour les paramètres avancés
    popup = tk.Toplevel()
    popup.title(lang["advanced_settings"])
    place_relative(popup, root, 600, 900)
    # Type de filtre Butterworth, Chebyshev I, Chebyshev II : menu déroulant
    ttk.Label(popup, text=lang["filter_kind"]).pack()
    filter_kind_var = tk.StringVar(value=filter_kind)
    filter_options = ["Butterworth", "Chebyshev I", "Chebyshev II"]
    filter_kind_menu = ttk.OptionMenu(popup, filter_kind_var, *filter_options)
    filter_kind_menu.pack()
    ttk.Label(popup, text=lang["filter_order"]).pack()
    filter_order_var = tk.IntVar(value=filter_order) # valeur initiale
    ttk.Entry(popup, textvariable=filter_order_var).pack() # ordre du filtre Butterworth
    ttk.Label(popup, text=lang["alpha_noise"]).pack()
    alpha_noise_var = tk.DoubleVar(value=alpha_noise)
    ttk.Entry(popup, textvariable=alpha_noise_var).pack() # alpha pour débruitage spectral
    ttk.Label(popup, text=lang["peak_prominence"]).pack()
    peak_prominence_var = tk.DoubleVar(value=peak_prominence)
    ttk.Entry(popup, textvariable=peak_prominence_var).pack() # proéminence des pics pour centrage
    ttk.Label(popup, text=lang["acf_min_distance"]).pack()
    acf_min_distance_var = tk.IntVar(value=acf_min_distance)
    ttk.Entry(popup, textvariable=acf_min_distance_var).pack() # distance min pour considérer pic ACF
    ttk.Label(popup, text=lang["tau_modifier"]).pack()
    tau_modifier_var = tk.DoubleVar(value=tau_modifier)
    ttk.Entry(popup, textvariable=tau_modifier_var).pack() # modificateur de tau pour wpcr
    ttk.Label(popup, text=lang["precision"]).pack()
    precision_var = tk.DoubleVar(value=precision)
    ttk.Entry(popup, textvariable=precision_var).pack() # précision pour recherche de rapidité
    ttk.Label(popup,text=lang["morlet_params"]).pack()
    morlet_fc_var = tk.DoubleVar(value=morlet_fc)
    morlet_nfreq_var = tk.IntVar(value=morlet_nfreq)
    ttk.Entry(popup, textvariable=morlet_fc_var).pack() # param fréquence centrale CWT Morlet
    ttk.Entry(popup, textvariable=morlet_nfreq_var).pack() # nombre de fréquences CWT Morlet
    ttk.Label(popup, text=lang["scf_alpha_step"]).pack()
    scf_alpha_step_var = tk.IntVar(value=scf_alpha_step)
    ttk.Entry(popup, textvariable=scf_alpha_step_var).pack() # pas alpha pour SCF
    ttk.Label(popup, text=lang["costas_damping"]).pack()
    costas_damping_var = tk.DoubleVar(value=costas_damping)
    ttk.Entry(popup, textvariable=costas_damping_var).pack() # facteur d'amortissement pour Costas
    ttk.Label(popup, text=lang["costas_bw_factor"]).pack()
    costas_bw_factor_var = tk.DoubleVar(value=costas_bw_factor)
    ttk.Entry(popup, textvariable=costas_bw_factor_var).pack() # facteur de boucle bande passante pour Costas
    ttk.Label(popup, text=lang["eye_diagram_params"]).pack()
    eye_channel_var = tk.StringVar(value=eye_channel)
    eye_num_traces_var = tk.IntVar(value=eye_num_traces)
    eye_symbols_var = tk.IntVar(value=eye_symbols)
    ttk.Entry(popup, textvariable=eye_channel_var).pack() # canal pour diagramme de l'oeil
    ttk.Entry(popup, textvariable=eye_num_traces_var).pack() # nombre de traces pour diagramme de l'oeil
    ttk.Entry(popup, textvariable=eye_symbols_var).pack() # nombre de symboles par trace pour diagramme de l'oeil
    ttk.Label(popup, text=lang["mfsk_tone_detection_params"]).pack()
    ttk.Label(popup, text=lang["mfsk_tresh_db"]).pack()
    mfsk_tresh_db_var = tk.DoubleVar(value=mfsk_tresh_db)
    ttk.Entry(popup, textvariable=mfsk_tresh_db_var).pack() # seuil en dB pour démodulation MFSK
    ttk.Label(popup, text=lang["mfsk_peak_prom_db"]).pack()
    mfsk_peak_prom_db_var = tk.DoubleVar(value=mfsk_peak_prom_db)
    ttk.Entry(popup, textvariable=mfsk_peak_prom_db_var).pack() # proéminence des pics pour démodulation MFSK
    ttk.Label(popup, text=lang["mfsk_win_factor"]).pack()
    mfsk_win_factor_var = tk.DoubleVar(value=mfsk_win_factor)
    ttk.Entry(popup, textvariable=mfsk_win_factor_var).pack() # facteur de taille de fenêtre pour démodulation MFSK
    ttk.Label(popup, text=lang["mfsk_hop_factor"]).pack()
    mfsk_hop_factor_var = tk.DoubleVar(value=mfsk_hop_factor)
    ttk.Entry(popup, textvariable=mfsk_hop_factor_var).pack() # facteur de saut pour démodulation MFSK
    ttk.Label(popup, text=lang["mfsk_cluster_bin_width_factor"]).pack()
    mfsk_cluster_bin_width_factor_var = tk.DoubleVar(value=mfsk_bin_width_cluster_factor)
    ttk.Entry(popup, textvariable=mfsk_cluster_bin_width_factor_var).pack() # facteur de largeur de bin pour clustering dans démodulation MFSK
    ttk.Label(popup, text=lang["mfsk_viterbi_penalty"]).pack()
    mfsk_viterbi_penalty_var = tk.DoubleVar(value=mfsk_viterbi_penalty)
    ttk.Entry(popup, textvariable=mfsk_viterbi_penalty_var).pack() # pénalité Viterbi pour démodulation MFSK
    constel_hexbins_var = tk.BooleanVar(value=constel_hexbins) # pour l'affichage en hexbins de la constellation
    ttk.Checkbutton(popup, text=lang["constel_hexbins"], variable=constel_hexbins_var).pack()
    decim_spec_var = tk.BooleanVar(value=decim_spec) # pour activer ou non la décimation du spectrogramme
    ttk.Checkbutton(popup, text=lang["decim_spec"], variable=decim_spec_var).pack()

    def save_settings():
        nonlocal filter_order_var, peak_prominence_var, acf_min_distance_var, tau_modifier_var, precision_var
        nonlocal morlet_fc_var, morlet_nfreq_var, scf_alpha_step_var, costas_damping_var, costas_bw_factor_var
        nonlocal eye_channel_var, eye_num_traces_var, eye_symbols_var
        nonlocal mfsk_tresh_db_var, mfsk_peak_prom_db_var, mfsk_win_factor_var, mfsk_viterbi_penalty_var
        nonlocal mfsk_cluster_bin_width_factor_var, mfsk_hop_factor_var
        nonlocal alpha_noise_var, filter_kind_var, constel_hexbins_var, decim_spec_var
        global filter_order, peak_prominence, acf_min_distance, tau_modifier, precision, morlet_fc, morlet_nfreq
        global scf_alpha_step, costas_damping, costas_bw_factor, eye_channel, eye_num_traces, eye_symbols
        global mfsk_tresh_db, mfsk_peak_prom_db, mfsk_win_factor, mfsk_viterbi_penalty, mfsk_bin_width_cluster_factor, mfsk_hop_factor
        global alpha_noise, filter_kind, constel_hexbins, decim_spec
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
        alpha_noise = alpha_noise_var.get()
        filter_kind = filter_kind_var.get()
        constel_hexbins = constel_hexbins_var.get()
        decim_spec = decim_spec_var.get()
        if debug is True:
            print(f"Paramètres avancés mis à jour : "
                  f"filter_order={filter_order}, peak_prominence={peak_prominence}, acf_min_distance={acf_min_distance}, "
                  f"tau_modifier={tau_modifier}, precision={precision}, morlet_fc={morlet_fc}, morlet_nfreq={morlet_nfreq}, "
                  f"scf_alpha_step={scf_alpha_step}, costas_damping={costas_damping}, costas_bw_factor={costas_bw_factor}, "
                  f"eye_channel={eye_channel}, eye_num_traces={eye_num_traces}, eye_symbols={eye_symbols}, "
                  f"mfsk_tresh_db={mfsk_tresh_db}, mfsk_peak_prom_db={mfsk_peak_prom_db}, mfsk_win_factor={mfsk_win_factor}, "
                  f"mfsk_hop_factor={mfsk_hop_factor}, mfsk_bin_width_cluster_factor={mfsk_bin_width_cluster_factor}, "
                  f"mfsk_viterbi_penalty={mfsk_viterbi_penalty}, alpha_noise={alpha_noise}, filter_kind={filter_kind}, "
                  f"constel_hexbins={constel_hexbins}, decim_spec={decim_spec}")
        popup.destroy()
    ttk.Button(popup, text="OK", command=save_settings).pack()

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
    experimental_menu = tk.Menu(menu_bar, tearoff=0)
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
    menu_bar.add_cascade(label=lang["experimental"], menu=experimental_menu)
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
    param_submenu.add_command(label=lang["read_file"], command=read_other_filetype)
    param_submenu.add_command(label=lang["metadata"], command=wav_metadata)
    info_menu.add_cascade(label=lang["params"], menu=param_submenu)
    info_menu.add_command(label=lang["coord_window"], command=open_coordinates_window)
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
    cut_submenu.add_command(label=lang["modify_in_zone"], command=modify_signal_in_zone)
    mod_menu.add_cascade(label=lang["cut_signal"],menu=cut_submenu)
    # Submenu fenêtrage
    window_submenu = tk.Menu(mod_menu,tearoff=0)
    window_submenu.add_command(label=lang["fft_size"], command=define_N)
    window_submenu.add_command(label=lang["set_window"], command=set_window)
    window_submenu.add_command(label=lang["set_overlap"], command=set_overlap)
    mod_menu.add_cascade(label=lang["window_options"], menu=window_submenu)
    # autres modifs en vrac
    adds_submenu = tk.Menu(mod_menu,tearoff=0)
    # Lissage
    adds_submenu.add_command(label=lang["param_phase_freq"], command=set_diff_params)
    adds_submenu.add_command(label=lang["rotation_correction"], command=correct_constellation_rotation)
    mod_menu.add_cascade(label=lang["adds_options"], menu=adds_submenu)
    # Enregistrer nouveau wav
    save_submenu = tk.Menu(mod_menu, tearoff=0)
    save_submenu.add_command(label="Stereo", command=save_as_wav)
    save_submenu.add_command(label="Mono", command=save_as_mono_wav)
    save_submenu.add_command(label=lang["split_file"], command=split_file)
    mod_menu.add_cascade(label=lang["save_wav"], menu=save_submenu)
    # Menu Filtre
    filter_menu.add_command(label=lang["filter_high_low"], command=apply_filter_high_low)
    filter_menu.add_command(label=lang["filter_band"], command=apply_filter_band)
    filter_menu.add_command(label=lang["spectral_denoise"], command=apply_spectral_denoise)
    filter_menu.add_command(label=lang["wiener_filter"], command=apply_wiener_filter)
    filter_menu.add_command(label=lang["fir_filter"], command=apply_fir_filter)
    filter_menu.add_command(label=lang["matched_filter"], command=apply_matched_filter)
    filt_submenu = tk.Menu(filter_menu, tearoff=0)
    filt_submenu.add_command(label=lang["median_filter"], command=apply_median_filter)
    filt_submenu.add_command(label=lang["moving_average"], command=apply_moving_average)
    filter_menu.add_cascade(label=lang["filter_legacy"], menu=filt_submenu)
    # Analyse du signal
    # Puissance
    power_menu.add_command(label=lang["dsp"], command=dsp)
    power_menu.add_command(label=lang["dsp_max"], command=dsp_max)
    power_menu.add_command(label=lang["time_amp"], command=time_amplitude)
    power_menu.add_command(label=lang["signal_power"], command=signal_power)
    # Phase
    diff_menu.add_command(label=lang["constellation"], command=constellation)
    diff_menu.add_command(label=lang["distrib_phase"], command=phase_cumulative)
    diff_menu.add_command(label=lang["diff_phase"], command=phase_difference)
    diff_menu.add_command(label=lang["eye_diagram"], command=eye_diagram)
    diff_menu.add_command(label=lang["constellation_sync"], command=constellation_synchronized)
    # Frequence
    freq_menu.add_command(label=lang["diff_freq"], command=freq_difference)
    freq_menu.add_command(label=lang["distrib_freq"], command=frequency_cumulative)
    freq_menu.add_command(label=lang["persist_spectrum"], command=spectre_persistance)
    # Rapidité de modulation
    speed_menu.add_command(label=lang["psf"], command=psf)
    speed_menu.add_command(label=lang["mts"], command=mts)
    speed_menu.add_command(label=lang["pseries"], command=pseries)
    speed_menu.add_command(label=lang["cyclospectrum"], command=cyclospectrum)
    speed_menu.add_command(label=lang["clock_graph"], command=clock_graph)
    # ACF
    acf_menu.add_command(label=lang["autocorr"], command=autocorr)
    acf_menu.add_command(label=lang["scf"], command=scf)
    # OFDM
    ofdm_menu.add_command(label=lang["ofdm_symbol"], command=alpha_from_symbol)
    ofdm_menu.add_command(label=lang["ofdm_results"], command=ofdm_results)
    # Demod
    demod_menu.add_command(label=lang["demod_cpm_psk"], command=demod_cpm_psk)
    demod_menu.add_command(label=lang["demod_fm"], command=demod_fm)
    demod_menu.add_command(label=lang["demod_am"], command=demod_am)
    demod_menu.add_command(label=lang["demod_ssb"], command=demod_ssb)
    demod_menu.add_command(label=lang["demod_mfsk"], command=demod_mfsk)
    # Audio
    if with_sound:
        audio_menu.add_command(label=lang["audio"], command=audio_output)
    # Experimental
    experimental_menu.add_command(label=lang["scalogram"], command=morlet_wavelet)
    experimental_menu.add_command(label=lang["bispectrum"], command=bispectrum_plot)
    experimental_menu.add_command(label=lang["cepstrum"], command=cepstrum)
    experimental_menu.add_command(label=lang["lora_score"], command=lora_score)

# Label d'infos au départ
info_label = ttk.Label(root, text=lang["load_msg"])
info_label.pack()

# Button frame
button_frame = ttk.Frame(root)
button_frame.pack(fill=tk.X, padx=5, pady=5)

# Charge fichier
load_button = ttk.Button(button_frame, text=lang["load"], command=load_wav)
load_button.pack(side=tk.LEFT)
close_button = ttk.Button(button_frame, text=lang["close"], command=close_wav)
close_button.pack(side=tk.LEFT, padx=5)
shift_button = ttk.Button(button_frame, text=lang["shift_frq"], command=shift_frequency)
shift_button.pack(side=tk.LEFT, padx=5)
load_mono_button = ttk.Button(button_frame, text=lang["mono_real"], command=load_real)
# Active/désactive curseurs
mode_button = ttk.Button(button_frame, text=lang["cursors_off"], command=toggle_cursor_mode)
mode_button.pack(side=tk.RIGHT)
clear_button = ttk.Button(button_frame, text=lang["clear_cursors"], command=clear_cursors)
clear_button.pack(side=tk.RIGHT, padx=5)
peak_button = ttk.Button(button_frame, text=lang["peak_find"], command=move_cursor_to_nearest_peak)
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
