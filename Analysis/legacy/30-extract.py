#! /usr/bin/env python

import numpy as np
from scipy.fft import dct
from numpy.lib.stride_tricks import sliding_window_view

from util.loader import load_file_keys, load_handwriting, save_sequence

CONFIG = 2
FEATURES = 2

WINDOW = 192
HOP = 96

# Pipeline 0:
# SIMPLE
# Features (grouped by metric not axis)
# Temporal: mean, std, power, mcr

# Pipeline 1:
# Spectral bands: 64/32 sample DCT-II, bottom 38 bins (0-15hz), 4 equal bands with mean abs mag
# Features (grouped by metric not axis)
# Temporal: mean, std, min, max, power, mcr
# Spectral: mean, std, power (per band)

# Pipeline 2:
# Temporal like 1
# Spectral like 1 (only calibrated 6dof axes)
# Chunked into 200 sample segments

def extract_spectra(sequence: np.array) -> np.array:
    frames = sliding_window_view(sequence, window_shape=64, axis=0)[::32]
    dct_mag = dct(frames, type=2, axis=-1, norm='ortho')[..., :38] ** 2
    bands = np.stack([np.mean(b, axis=-1) for b in np.array_split(dct_mag, 4, axis=-1)])

    return bands

# Take a (t, N) series of vectors and calculate feature-wise statistical measures
def extract_features(sequence: np.array) -> np.array:
    spectrum_bands = extract_spectra(sequence)

    seq_mean = np.mean(sequence, axis=0)
    seq_std = np.std(sequence, axis=0)
    seq_pow = np.mean(sequence ** 2, axis=0)
    seq_mcr = np.count_nonzero(np.diff(np.sign(sequence - seq_mean), axis=0), axis=0) / len(sequence)

    spec_means = np.mean(spectrum_bands, axis=1).flatten()
    spec_stds = np.std(spectrum_bands, axis=1).flatten()
    spec_pows = np.mean(spectrum_bands ** 2, axis=1).flatten()

    seq_stats = [seq_mean, seq_std, seq_pow, seq_mcr]
    spec_stats = [spec_means, spec_stds, spec_pows]

    # print(np.array(seq_stats).shape, np.hstack([*seq_stats]).shape)
    if CONFIG == 0:
        return np.hstack([*seq_stats])
    else:
        return np.hstack([*seq_stats, *spec_stats])

def process_files_windowed(data: dict):
    file_count = 0
    for key in data:
        file_count += 1
        print(f"Extracting features from file {file_count:03}/{len(data)}", end='\r')

        pid, tno, desc, pss, rest = key
        
        windows = sliding_window_view(data[key], 
            window_shape=np.min([WINDOW, data[key].shape[0]]), 
            axis=0
        )[::HOP, ...]

        window_count = 0
        for window in windows:
            window_key = (pid, 100 * tno + window_count, desc, pss, rest)
            window_count += 1

            window_features = extract_features(window.transpose())

            save_sequence(
                path=f"Features/30-Extracted-A{FEATURES}-{CONFIG}W", 
                file_key=window_key, 
                sequence=window_features.reshape((-1, 1)), 
                header=["features"]
            )

    print(f"Finished processing {file_count} files.")

def process_files(data: dict):
    file_count = 0
    for key in data:
        file_count += 1
        print(f"Extracting features from file {file_count:03}/{len(data)}", end='\r')

        features = extract_features(data[key])

        save_sequence(
            path=f"Features/30-Extracted-A{FEATURES}-{CONFIG}", 
            file_key=key, 
            sequence=features.reshape((-1, 1)), 
            header=["features"]
        )

    print(f"Finished processing {file_count} files.")

if __name__ == "__main__":
    file_info, key_info = load_file_keys(f"Series/20-Augmented-{FEATURES}")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) # ,[])

    print("Processing files...")
    if CONFIG < 2:
        process_files(data)
    else:
        process_files_windowed(data)

