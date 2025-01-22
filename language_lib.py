# Lib de langue pour l'interface graphique de SigAnTo

def get_fra_lib():
    # load FR
    fra_lib = {
        "siganto": "SigAnTo : Outil d'analyse de signaux",
        "lang": "Français",
        "load": "Charger un fichier WAV / Rétablir le fichier WAV",
        "load_msg": "Charger un fichier WAV pour commencer l'analyse",
        "close": "Fermer le fichier WAV",
        "invalid_wav": "Fichier WAV invalide",
        "no_file": "Aucun fichier chargé",
        "encoding": "Encodage: ",
        "samples": "Echantillons: ",
        "sampling_frq": "Fréquence d'échantillonnage: ",
        "error": "Erreur",
        "apply": "Appliquer",
        "fin": "Fin",
        "duree": "Durée",
        "oui": "Oui",
        "non": "Non",
        "params": "Paramètres",
        "main_viz": "Visualisation Principale",
        "group_spec": "Groupe: Spectrogramme et DSP",
        "spec_dsp": "Spectrogramme et DSP",
        "group_const": "Groupe: Spectrogramme, Constellation, DSP max",
        "spec_const": "Spectrogramme, Constellation, DSP max",
        "spec_3d": "Spectrogramme 3D",
        "display": "Affichage",
        "frq_info": "Informations de fréquence/puissance",
        "cursors_on": "Curseurs: ON",
        "cursors_off": "Curseurs: OFF",
        "clear_cursors": "Retirer curseurs",
        "modify": "Modifier Signal",
        "move_frq": "Déplacer la fréquence centrale",
        "high_low": "Filtre passe-bas ou passe-haut",
        "low_val": "Bas",
        "high_val": "Haut",
        "freq_pass": "Fréquence de coupure (Hz) :",
        "filter_high_low": "Appliquer un filtre passe-haut/bas",
        "bandpass": "Filtre passe-bande",
        "filter_band": "Appliquer un filtre passe-bande",
        "freq_low": "Fréquence de coupure basse en Hz:",
        "freq_high": "Fréquence de coupure haute en Hz:",
        "freq_valid": "Entrez des fréquences de coupure valides",
        "mean": "Moyenner le signal",
        "mean_level": "Niveau moyen en dB: ",
        "level": "Niveau de moyennage: ",
        "apply_mean": "Appliquer le moyennage par défaut",
        "def_level": "Définir le seuil",
        "enter_level": "Entrer le niveau en dB :", 
        "downsample": "Sous-échantillonner",
        "down_value": "Entrez le taux de sous-échantillonnage (entier) : ",
        "upsample": "Sur-échantillonner",
        "up_value": "Entrez le taux de sur-échantillonnage (entier) : ",
        "cut": "Couper la durée du signal (valeurs)",
        "start_cut": "Entrez le début de la coupure en secondes: ",
        "end_cut": "Entrez la fin de la coupure en secondes: ",
        "valid_cut" : "Entrez des valeurs valides pour la coupure",
        "cut_cursors": "Couper la durée du signal (curseurs)",
        "2pt_cursors": "Sélectionnez 2 points pour couper le signal",
        "move_freq_cursors": "Déplacer la fréquence centrale (curseur)",
        "1pt_cursors": "Sélectionnez 1 point pour sélectionner la fréquence centrale",
        "power_estimate": "Mesures Puissance",
        "psf": "FFT de puissance",
        "mts": "FFT de puissance (seuil moyen)",
        "mag": "Puissance",
        "estim_peak": "Pic de puissance estimé à :",
        "pseries": "Signal puissance",
        "transitions_estimate": "Mesures Phase",
        "diff_phase": "Transitions de phase",
        "diff_freq": "Transitions de fréquence",
        "param_phase_freq": "Paramètres de transitions de phase et fréquence",
        "distrib_phase": "Distribution de phase cumulée",
        "density": "Densité",
        "insta_phase": "Phase",
        "cyclo_estimate": "Mesures Cyclostationnarité",
        "autocorr": "Autocorrélation rapide",
        "autocorr_full": "Autocorrélation complète",
        "acf_peak": "Recherche automatique d'autocorrélation",
        "acf_peak_txt": "Pic d'autocorrélation détecté à",
        "min_acf_dist": "Distance minimale pour rechercher un pic d'autocorrélation :",
        "ofdm": "Mesures OFDM",
        "fft_size": "Changer la taille de la fenêtre FFT",
        "fft_window": "Fenêtre FFT :",
        "freq_xy": "Fréquence",
        "time_xy": "Temps",
        "fc": "Fréquence Centrale",
        "move_txt": "Entrer la correction de FC en Hz (en négatif pour décaler vers la droite) :",
        "confirm_wait" : "Cela peut prendre du temps. Continuer?",
        "smoothing_val": "Entrez une valeur de lissage des transitions (désactivé par défaut) :",
        "alpha": "Recherche de l'alpha optimal",
        "estim_tu": "Entrer la valeur estimée de la durée symbole OFDM (par la fonction d'autocorrélation) :",
        "alpha0": "Entrer le décalage alpha0 en Hz:",
        "ofdm_results": "Caractérisation OFDM",
        "tu": "Temps utile OFDM",
        "tg": "Temps garde OFDM",
        "ts": "Temps utile + garde OFDM",
        "df": "Delta F",
        "alpha_peak": "Pic alpha détecté à :",
        "clock_recovery": "Synchronisation d'horloge de phase",
        "experimental": "EXPERIMENTAL",
        "high_level": "Niveau le plus haut en dB :",
        "low_level": "Niveau le plus bas en dB :",
        "high_freq": "Fréquence à la puissance max",
        "low_freq": "Fréquence à la puissance min",
        "estim_speed": "Estimation de rapidité de modulation par FFT de puissance - seuil moyen :",
        "estim_speed_2" : "Estimation secondaire de rapidité de modulation :",
        "phase_spectrum": "Spectre de phase",
        "bandwidth": "Bande passante",
        "dsp": "Densité Spectrale de Puissance",
        "constellation": "Constellation",
        "define_n": "Entrer la nouvelle valeur de fenêtre FFT :",
        "estim_failed": "Echec de l'estimation de rapidité de modulation. Tentez une autre méthode.",
        "time_estimate": "Mesures Temporelles",
        "time_amp": "Temps/Amplitude",
        "persist_spectrum": "Spectre de persistance",
        "spectrogram": "Spectrogramme",
        "smoothing": "Lissage",
        "num_ssp": "Nombre de sous-porteuses",
        "estim_bw": "Estimation de la bande passante :",
        "redef_bw": "Pour redéfinir la bande passante, entrez une nouvelle valeur. Sinon, appuyez sur 'OK'.",
        "unreliable_sscarrier": "L'estimation de position des sous-porteuses est dépendant du calage en fréquence et de la bande passante.",
        "pers_bins": "Nombre de bins pour le spectre de persistance (par défaut 50) :",
        "param_spectre_persistance": "Paramètres du spectre de persistance",
        "save_wav" : "Sauvegarder modifications (nouveau .wav)",
        "estim_peaks" : "Ecart (abs) entre deux pics de puissance :",
        "peak_find" : "Pic proche curseur",
        "window" : "Fenêtre :",
        "f_resol" : "Résolution en fréquence :",
        "set_window" : "Définir la fonction de fenêtrage",
        "window_choice" : "Choix de la fonction de fenêtrage :",
        "overlap_val" : "Valeur de recouvrement (taille de fenêtre//{nouveau facteur}) :",
        "set_overlap" : "Définir la valeur de recouvrement",
        "dsp_max" : "DSP max",
        "error_stft" : "Erreur dans le calcul de la STFT. Essayez une autre taille de fenêtre, fonction et/ou recouvrement.",
        "freq_adjust" : "Ajustement de fréquence",
        "overlap_valid" : "Entrez une valeur de recouvrement valide",
        "offset_freq" : "Décalage de fréquence",
        "not_apply" : "Pas de moyennage",
        "wav_load" : "Chargement du wav",
    
    }
    return fra_lib

def get_eng_lib():
    # load EN
    eng_lib = {
        "siganto": "SigAnTo : Signal Analysis Tool",
        "lang": "English",
        "load": "Load a WAV file / Restore WAV file",
        "load_msg": "Load a WAV file to start the analysis",
        "close": "Close the WAV file",
        "invalid_wav": "Invalid WAV file",
        "no_file": "No file loaded",
        "encoding": "Encoding: ",
        "samples": "Samples: ",
        "sampling_frq": "Sampling frequency: ",
        "error": "Error",
        "apply": "Apply",
        "debut": "Start",
        "fin": "End",
        "duree": "Duration",
        "oui": "Yes",
        "non": "No",
        "params": "Parameters",
        "main_viz": "Main Graphs",
        "group_spec": "Group: Spectrogram and PSD",
        "spec_dsp": "Spectrogram and PSD",
        "group_const": "Group: Spectrogram, Constellation, PSD max",
        "spec_const": "Spectrogram, Constellation, PSD max",
        "spec_3d": "3D Spectrogram",
        "display": "Display",
        "frq_info": "Frequency/Power Information",
        "cursors_on": "Cursors: ON",
        "cursors_off": "Cursors: OFF",
        "clear_cursors": "Remove cursors",
        "modify": "Modify Signal",
        "move_frq": "Move center frequency",
        "high_low": "High/Low pass filter",
        "low_val": "Low",
        "high_val": "High",
        "freq_pass": "Cut-off frequency (Hz) :",
        "filter_high_low": "Apply high/low pass filter",
        "bandpass": "Band pass filter",
        "filter_band": "Apply band pass filter",
        "freq_low": "Low cut-off frequency in Hz:",
        "freq_high": "High cut-off frequency in Hz:",
        "freq_valid": "Enter valid cut-off frequencies",
        "mean": "Average signal",
        "mean_level": "Mean level in dB: ",
        "level": "Averaging level: ",
        "apply_mean": "Apply default mean",
        "def_level": "Set threshold",
        "enter_level": "Enter level in dB :",
        "downsample": "Downsample",
        "down_value": "Enter the downsampling ratio (int): ",
        "upsample": "Upsample",
        "up_value": "Enter the upsampling ratio (int): ",
        "cut": "Cut length of signal (values)",
        "start_cut": "Enter the start of the cut in seconds: ",
        "end_cut": "Enter the end of the cut in seconds: ",
        "valid_cut" : "Enter valid values for the cut",
        "cut_cursors": "Cut length of signal (cursors)",
        "2pt_cursors": "Select 2 points to cut the signal",
        "1pt_cursors": "Select 1 point to set the center frequency",
        "move_freq_cursors": "Move center frequency (cursor)",
        "power_estimate": "Power Metrics",
        "psf": "Power Spectrum FFT",
        "mts": "Power Spectrum FFT (mean threshold)",
        "mag": "Magnitude",
        "estim_peak": "Estimated power peak at :",
        "pseries": "Power Series FFT",
        "transitions_estimate": "Phase Metrics",
        "diff_phase": "Phase Transitions",
        "diff_freq": "Frequency Transitions",
        "param_phase_freq": "Phase and Frequency Transitions Parameters",
        "distrib_phase": "Cumulative Phase Distribution",
        "density": "Density",
        "insta_phase": "Phase",
        "cyclo_estimate": "Cyclostationarity Metrics",
        "autocorr": "Fast Autocorrelation",
        "autocorr_full": "Full Autocorrelation",
        "acf_peak": "Autocorrelation Peak Automatic Search",
        "acf_peak_txt": "Autocorrelation peak detected at",
        "min_acf_dist": "Minimum distance to find autocorrelation peaks :",
        "ofdm": "OFDM Metrics",
        "fft_size": "Change FFT window size",
        "fft_window": "FFT Window :",
        "freq_xy": "Frequency",
        "time_xy": "Time",
        "fc": "Center Frequency",
        "move_txt": "Enter the CF correction in Hz (negative to shift right) :",
        "confirm_wait" : "This may take some time. Continue?",
        "smoothing_val": "Enter a smoothing value for transitions (disabled by default) :",
        "alpha": "Optimal alpha search",
        "estim_tu": "Enter the estimated OFDM symbol duration value (from the autocorrelation function) :",
        "alpha0": "Enter the alpha0 offset in Hz :",
        "ofdm_results": "OFDM Parameters",
        "tu": "OFDM Symbol Duration",
        "tg": "OFDM Guard Duration",
        "ts": "OFDM Symbol + Guard Duration",
        "df": "Delta F",
        "alpha_peak": "Alpha peak detected at :",
        "experimental": "EXPERIMENTAL",
        "high_level": "Highest level in dB :",
        "low_level": "Lowest level in dB :",
        "high_freq": "Frequency at max power",
        "low_freq": "Frequency at min power",
        "estim_speed": "Symbol rate estimation with Power FFT - mean threshold :",
        "estim_speed_2" : "Secondary symbol rate estimation : ",
        "phase_spectrum": "Phase Spectrum",
        "bandwidth": "Bandwidth",
        "dsp": "Power Spectral Density",
        "constellation": "Constellation",
        "define_n": "Enter the new FFT window size :",
        "estim_failed": "Symbol rate estimation failed. Try another method.",
        "time_estimate": "Time Metrics",
        "time_amp": "Time/Amplitude",
        "persist_spectrum": "Persistence Spectrum",
        "spectrogram": "Spectrogram",
        "smoothing": "Smoothing",
        "num_ssp": "Number of subcarriers",
        "estim_bw": "Bandwidth estimation :",
        "redef_bw": "To redefine the bandwidth, enter a new value. Otherwise, press 'OK'.",
        "unreliable_sscarrier": "Subcarriers position estimation is dependent on frequency offset and bandwidth.",
        "pers_bins": "Number of bins for the persistence spectrum (default 50) :",
        "param_spectre_persistance": "Persistence Spectrum Parameters",
        "save_wav" :"Save signal modifications (new .wav)",
        "estim_peaks" : "Gap (abs) between 2 power peaks :",
        "peak_find" : "Peak near cursor",
        "window" : "Window :",
        "f_resol" : "Frequency resolution :",
        "set_window" : "Set window function",
        "window_choice" : "Window function choice :",
        "overlap_val" : "Overlap value (window size//{new factor}) :",
        "set_overlap" : "Set overlap value",
        "dsp_max" : "Max PSD",
        "error_stft" : "Error in STFT computation. Try another window size, function and/or overlap.",
        "freq_adjust" : "Frequency fine-tuning",
        "overlap_valid" : "Enter a valid overlap value",
        "offset_freq" : "Frequency offset",
        "not_apply" : "No averaging",
        "wav_load" : "Wav loading",

    }
    return eng_lib