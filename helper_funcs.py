"""Fonctions annexes de l'application"""
import numpy as np
import struct
import os
from fractions import Fraction

def downsample_for_plotting(x_array, y_array, target_points=4000):
    """Décimation 1D pour l'affichage, en conservant les pics et transitoires."""
    n_samples = len(y_array)
    if n_samples <= target_points:
        return x_array, y_array

    # Calculer la taille du bin pour atteindre le nombre de points cible
    bin_size = int(np.ceil(n_samples / (target_points // 2)))
    # Répartition des données en bins, avec padding si nécessaire
    pad_size = (bin_size - (n_samples % bin_size)) % bin_size
    if pad_size > 0:
        y_padded = np.pad(y_array, (0, pad_size), mode='edge')
        x_padded = np.pad(x_array, (0, pad_size), mode='edge')
    else:
        y_padded = y_array
        x_padded = x_array

    # Reshape pour créer des bins et trouver les min/max dans chaque bin
    reshaped_data = y_padded.reshape(-1, bin_size)
    reshaped_x = x_padded.reshape(-1, bin_size)
    # Extraction des indices des min et max pour chaque bin
    min_idx = np.argmin(reshaped_data, axis=1)
    max_idx = np.argmax(reshaped_data, axis=1)
    # Générer les indices globaux pour les min et max
    row_offsets = np.arange(reshaped_data.shape[0]) * bin_size
    flat_min_idx = min_idx + row_offsets
    flat_max_idx = max_idx + row_offsets

    # Créer un tableau d'indices intercalés pour conserver l'ordre chronologique
    interleaved_idx = np.empty(flat_min_idx.size * 2, dtype=int)
    # Déterminer si le min ou le max vient en premier pour chaque bin
    min_first = min_idx < max_idx
    interleaved_idx[0::2] = np.where(min_first, flat_min_idx, flat_max_idx)
    interleaved_idx[1::2] = np.where(min_first, flat_max_idx, flat_min_idx)

    # Slice les tableaux d'origine avec les indices intercalés pour obtenir les données décimées
    return x_padded[interleaved_idx], y_padded[interleaved_idx]

def downsample_spectrogram(matrix, target_cols=1024, target_rows=1024):
    """Décimation 2D pour l'affichage, en conservant les pics et transitoires."""
    rows, cols = matrix.shape
    # 1. Décimation des colonnes (Axe temporel ou fréquentiel selon l'orientation)
    if cols > target_cols:
        col_bin = int(np.ceil(cols / target_cols))
        pad_cols = (col_bin - (cols % col_bin)) % col_bin
        if pad_cols > 0:
            matrix = np.pad(matrix, ((0, 0), (0, pad_cols)), mode='edge')
        # Reshape et prendre le max le long de l'axe des bins
        rows, cols = matrix.shape
        matrix = matrix.reshape(rows, -1, col_bin).max(axis=2)
        rows, cols = matrix.shape

    # 2. Décimation des lignes
    if rows > target_rows:
        row_bin = int(np.ceil(rows / target_rows))
        pad_rows = (row_bin - (rows % row_bin)) % row_bin
        if pad_rows > 0:
            matrix = np.pad(matrix, ((0, pad_rows), (0, 0)), mode='edge')
        # Reshape comme pour les colonnes
        matrix = matrix.reshape(-1, row_bin, cols).max(axis=1)

    return matrix

def calculate_distance(p1, p2):
    """Calcule la distance entre deux points p1 et p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx**2 + dy**2), dx, dy

def find_peaks(data):
    """Renvoie les indices des pics locaux dans un tableau 1D."""
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i - 1] < data[i] > data[i + 1]: # condition
            peaks.append(i)
    return peaks

def rational_resample_ratio(samp_rate, fs_new, max_den=2000):
    frac = Fraction(int(fs_new), int(samp_rate)).limit_denominator(max_den)
    return frac.numerator, frac.denominator # up, down

def read_wav_metadata(filename):
    """Lit les métadonnées d'un fichier WAV sans charger les échantillons.
    Renvoie un dictionnaire contenant tous les chunks RIFF reconnus."""
    metadata = {"file_size": os.path.getsize(filename),"chunks": []}

    with open(filename, "rb") as f:
        # RIFF header
        riff = f.read(4)
        if riff not in (b"RIFF", b"RF64"):
            raise ValueError("Pas un fichier WAV valide (RIFF ou RF64 attendu)")

        metadata["riff_type"] = riff.decode()
        metadata["riff_size"] = struct.unpack("<I", f.read(4))[0]

        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Pas un fichier WAV valide")

        # Boucle sur les chunks jusqu'à la fin du fichier
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break

            chunk_id, chunk_size = struct.unpack("<4sI", hdr)
            chunk_id = chunk_id.decode(errors="replace")
            chunk_start = f.tell()

            metadata["chunks"].append({"id": chunk_id,"offset": chunk_start,"size": chunk_size})

            if chunk_id == "fmt ":
                data = f.read(chunk_size)

                (audio_format, channels, sample_rate, byte_rate,
                block_align,bits_per_sample) = struct.unpack("<HHIIHH", data[:16])

                metadata.update({"audio_format": audio_format, "channels": channels,
                    "sample_rate (Hz)": sample_rate, "byte_rate": byte_rate,
                    "block_align": block_align, "bits_per_sample": bits_per_sample})

                if chunk_size > 16:
                    metadata["fmt_extra"] = data[16:]

            elif chunk_id == "fact":
                data = f.read(chunk_size)
                if chunk_size >= 4:
                    metadata["fact_samples"] = struct.unpack("<I", data[:4])[0]

            elif chunk_id == "data":
                metadata["data_offset"] = chunk_start
                metadata["data_size"] = chunk_size
                # Skip la lecture des échantillons, on ne veut que les métadonnées
                f.seek(chunk_size, 1)
            elif chunk_id == "LIST":
                metadata["LIST"] = f.read(chunk_size)
            elif chunk_id == "bext":
                metadata["bext"] = f.read(chunk_size)
            elif chunk_id == "iXML":
                metadata["iXML"] = f.read(chunk_size)
            elif chunk_id == "cue ":
                metadata["cue"] = f.read(chunk_size)
            elif chunk_id == "smpl":
                metadata["smpl"] = f.read(chunk_size)
            else:
                # Chunk inconnu, ignoré
                f.seek(chunk_size, 1)

            # Les chunks sont alignés sur des "mots" de 2 octets, 
            # si la taille du chunk est impaire, on saute un octet
            if chunk_size & 1:
                f.seek(1, 1)

    # Post-traitement des métadonnées pour obtenir des informations dérivées
    if "fmt_extra" in metadata and len(metadata["fmt_extra"]) >= 2:
        metadata["fmt_extra_size"] = struct.unpack("<H", metadata["fmt_extra"][:2])[0]
    if "data_size" in metadata and "block_align" in metadata:
        metadata["num_frames (samples)"] = metadata["data_size"] // metadata["block_align"]
    if "num_frames (samples)" in metadata and "sample_rate (Hz)" in metadata:
        metadata["duration (s)"] = metadata["num_frames (samples)"] / metadata["sample_rate (Hz)"]

    if "audio_format" in metadata:
        if metadata["audio_format"] == 1:
            metadata["audio_format_desc"] = "PCM"
        elif metadata["audio_format"] == 2:
            metadata["audio_format_desc"] = "ADPCM"
        elif metadata["audio_format"] == 3:
            metadata["audio_format_desc"] = "IEEE float"
        elif metadata["audio_format"] == 4 or metadata["audio_format"] == 5:
            metadata["audio_format_desc"] = "Compressed"
        elif metadata["audio_format"] == 6:
            metadata["audio_format_desc"] = "8-bit ITU-T G.711 A-law"
        elif metadata["audio_format"] == 7:
            metadata["audio_format_desc"] = "8-bit ITU-T G.711 µ-law"
        elif metadata["audio_format"] == 65534:
            metadata["audio_format_desc"] = "Extensible"
        else:
            metadata["audio_format_desc"] = "Unknown"

    if "channels" in metadata:
        if metadata["channels"] == 1:
            metadata["channel_desc"] = "Mono"
        elif metadata["channels"] == 2:
            metadata["channel_desc"] = "Stereo"
        else:
            metadata["channel_desc"] = f"{metadata['channels']}-channel"

    # Bext = Broadcast Extension Chunk
    if "bext" in metadata:
        bext_data = metadata["bext"]
        if len(bext_data) >= 602:
            metadata["bext_description"] = bext_data[0:256].decode(errors="replace").rstrip('\x00')
            metadata["bext_originator"] = bext_data[256:288].decode(errors="replace").rstrip('\x00')
            metadata["bext_originator_reference"] = bext_data[288:320].decode(errors="replace").rstrip('\x00')
            metadata["bext_origination_date"] = bext_data[320:328].decode(errors="replace").rstrip('\x00')
            metadata["bext_origination_time"] = bext_data[328:336].decode(errors="replace").rstrip('\x00')
            metadata["bext_time_reference_low"] = struct.unpack("<I", bext_data[336:340])[0]
            metadata["bext_time_reference_high"] = struct.unpack("<I", bext_data[340:344])[0]
            metadata["bext_version"] = struct.unpack("<H", bext_data[344:346])[0]
            metadata["bext_umid"] = bext_data[346:378]
            metadata["bext_loudness_value"] = struct.unpack("<h", bext_data[378:380])[0]
            metadata["bext_loudness_range"] = struct.unpack("<h", bext_data[380:382])[0]
            metadata["bext_max_true_peak_level"] = struct.unpack("<h", bext_data[382:384])[0]
            metadata["bext_max_momentary_loudness"] = struct.unpack("<h", bext_data[384:386])[0]
            metadata["bext_max_short_term_loudness"] = struct.unpack("<h", bext_data[386:388])[0]
            metadata["bext_reserved"] = bext_data[388:398]
            metadata["bext_coding_history"] = bext_data[398:].decode(errors="replace").rstrip('\x00')

    # iXML = Métadonnées XML pour l'audio
    if "iXML" in metadata:
        ixml_data = metadata["iXML"]
        metadata["iXML_text"] = ixml_data.decode(errors="replace").rstrip('\x00')

    # Cue = Points de repère dans le flux audio
    if "cue" in metadata:
        cue_data = metadata["cue"]
        if len(cue_data) >= 4:
            num_cue_points = struct.unpack("<I", cue_data[0:4])[0]
            metadata["num_cue_points"] = num_cue_points
            cue_points = []
            for i in range(num_cue_points):
                offset = 4 + i * 24
                if len(cue_data) >= offset + 24:
                    cue_point_id, position, data_chunk_id, chunk_start, block_start, sample_offset = struct.unpack("<IIIIII", cue_data[offset:offset + 24])
                    cue_points.append({
                        "cue_point_id": cue_point_id,"position": position,
                        "data_chunk_id": data_chunk_id.decode(errors="replace"),
                        "chunk_start": chunk_start, "block_start": block_start, "sample_offset": sample_offset})
            metadata["cue_points"] = cue_points

    # Smpl = Points de repère pour les échantillons (boucles, etc.)
    if "smpl" in metadata:
        smpl_data = metadata["smpl"]
        if len(smpl_data) >= 36:
            (manufacturer, product, sample_period, midi_unity_note, midi_pitch_fraction, smpte_format,
                smpte_offset, num_sample_loops, sampler_data) = struct.unpack("<IIIIIIIII", smpl_data[0:36])

            metadata.update({"smpl_manufacturer": manufacturer, "smpl_product": product,
                "smpl_sample_period": sample_period, "smpl_midi_unity_note": midi_unity_note,
                "smpl_midi_pitch_fraction": midi_pitch_fraction, "smpl_smpte_format": smpte_format,
                "smpl_smpte_offset": smpte_offset, "smpl_num_sample_loops": num_sample_loops,
                "smpl_sampler_data": sampler_data})

            sample_loops = []
            for i in range(num_sample_loops):
                offset = 36 + i * 24
                if len(smpl_data) >= offset + 24:
                    cue_point_id, type_, start, end, fraction, play_count = struct.unpack("<IIIIII", smpl_data[offset:offset + 24])
                    sample_loops.append({
                        "cue_point_id": cue_point_id, "type": type_,
                        "start": start, "end": end,
                        "fraction": fraction, "play_count": play_count})
            metadata["smpl_sample_loops"] = sample_loops

    return metadata

def read_integer_bits(filename, bits, signed=True, little_endian=True, offset=0):
    """Lit un binaire contenant des échantillons d'entier de taille 8 à 32 bits
    et les convertit en un tableau numpy int32."""

    bytes_per_sample = (bits + 7) // 8
    if bytes_per_sample > 4:
        raise ValueError("Taille d'échantillon non supportée (max 32 bits)")

    with open(filename, "rb") as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    if len(raw) % bytes_per_sample:
        raise ValueError("Échantillon incomplet quelque part. (Taille fichier - décalage) doit être un multiple de la taille de l'échantillon en bytes.")
    # on pourrait aussi tronquer les échantillons incomplets à la fin du fichier, mais c'est un peu hasardeux
    # raw = raw[:len(raw) - (len(raw) % bytes_per_sample)]

    raw = raw.reshape(-1, bytes_per_sample)
    x = np.zeros(len(raw), dtype=np.uint64)

    if little_endian:
        for i in range(bytes_per_sample):
            x |= raw[:, i].astype(np.uint64) << (8 * i)
    else:
        for i in range(bytes_per_sample):
            x |= raw[:, bytes_per_sample - 1 - i].astype(np.uint64) << (8 * i)

    # Retire padding et convertit en int32
    mask = (1 << bits) - 1
    x &= mask

    if signed:
        sign_bit = 1 << (bits - 1)
        x = x.astype(np.int32)
        x[x & sign_bit != 0] |= ~mask
    else:
        x = x.astype(np.int32)

    return x

def parabolic_interpol(mag, k):
    """Interpolation parabolique de pic spectral autour de k.
    Renvoie l'offset fractionnel de bin compris dans [-0.5 , 0.5]"""
    # plus idéal sur log magnitude qu'en linéaire
    if k <= 0 or k >= len(mag)-1:
        return 0.0 # si pas de bins voisins
    a = mag[k-1]
    b = mag[k]
    c = mag[k+1]
    denom = a - 2*b + c # approx de dérivée d'ordre 2
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (a - c) / denom # sommet de la parabole
    delta = np.clip(delta, -1.0, 1.0) # clip pour éviter des offsets irréalistes

    return delta