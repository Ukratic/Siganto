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
