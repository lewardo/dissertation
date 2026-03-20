#! /usr/bin/env python

import numpy as np
from scipy.fft import dct
from numpy.lib.stride_tricks import sliding_window_view

from util.loader import load_file_keys, load_handwriting, save_sequence

CONFIG = 1
FEATURES = 2

# Pipeline 0:
# SIMPLE
# Features (grouped by metric not axis)
# Temporal: mean, std, power, mcr

# Pipeline 1:
# Spectral bands: 64/32 sample DCT-II, bottom 38 bins (0-15hz), 4 equal bands with mean abs mag
# Features (grouped by metric not axis)
# Temporal: mean, std, min, max, power, mcr
# Spectral: mean, std, power (per band)

def extract_spectra(sequence: np.array) -> np.array:
    frames = sliding_window_view(sequence, window_shape=64, axis=0)[::32]
    dct_mag = dct(frames, type=2, axis=-1, norm='ortho')[..., :38] ** 2
    bands = np.stack([np.mean(b, axis=-1) for b in np.array_split(dct_mag, 4, axis=-1)])

    return bands

# Take a (t, N) series of vectors and calculate feature-wise statistical measures
def extract_features(sequence: np.array) -> np.array:
    # sequence length
    # Each of seven axes - min, max, avg, stdev, mcr, avg power
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

    return np.hstack([
        *seq_stats, 
        *spec_stats
    ])

if __name__ == "__main__":
    file_info, key_info = load_file_keys(f"Series/20-Augmented-{FEATURES}")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) # ,[])

    print("Processing files...")

    count = 0
    for key in data:
        count += 1

        features = extract_features(data[key])
        print(f"Extracting features from file {count:03}/{len(data)}", end='\r')
        save_sequence(f"Features/30-Extracted-A{FEATURES}-{CONFIG}", key, features.reshape((-1, 1)), ["features"])

    print(f"Finished processing {count} files.")