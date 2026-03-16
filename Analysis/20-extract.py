#! /usr/bin/env python

import numpy as np
from scipy.fftpack import dct
from numpy.lib.stride_tricks import sliding_window_view

from Analysis.util.loader import load_file_keys, load_handwriting, save_sequence

def extract_spectra(sequence: np.array) -> np.array:
    frames = sliding_window_view(sequence, window_shape=64, axis=0)[::32]
    dct_mag = np.abs(dct(frames, type=2, axis=-1, norm='ortho')[..., :38])
    bands = np.stack([np.mean(b, axis=-1) for b in np.array_split(dct_mag, 4, axis=-1)])

    return bands

# Take a (t, N) series of vectors and calculate feature-wise statistical measures
def extract_features(sequence: np.array) -> np.array:
    # sequence length
    # Each of seven axes - min, max, avg, stdev, mcr, avg power
    spectrum_bands = extract_spectra(sequence)

    # Start with c. 40
    seq_len = np.log(len(sequence))
    seq_mean = np.mean(sequence, axis=0)
    seq_std =  np.std(sequence, axis=0)
    seq_min =  np.min(sequence, axis=0)
    seq_max =  np.max(sequence, axis=0)
    seq_pow = np.mean(sequence ** 2, axis=0)
    seq_mcr = np.count_nonzero(np.diff(np.sign(sequence - seq_mean), axis=0), axis=0)

    spec_means = np.mean(spectrum_bands, axis=1).flatten()
    spec_stds = np.std(spectrum_bands, axis=1).flatten()
    # spec_mins = np.min(spectrum_bands, axis=1).flatten()
    # spec_maxs = np.max(spectrum_bands, axis=1).flatten()
    spec_pows = np.mean(spectrum_bands ** 2, axis=1).flatten()
    # spec_mcr = np.count_nonzero(np.diff(np.sign(spectrum_bands - spec_mean), axis=0), axis=0)

    seq_stats = [seq_len, seq_mean, seq_std, seq_min, seq_max, seq_pow, seq_mcr]
    spec_stats = [spec_means, spec_stds, spec_pows]

    return np.hstack([*seq_stats, *spec_stats])

if __name__ == "__main__":
    file_info, key_info = load_file_keys("Series/Augmented")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) # ,[])

    print("Processing files...")

    count = 0
    for key in data:
        count += 1

        features = extract_features(data[key])

        print(f"Extracting features from file {count:03}/{len(data)}", end='\r')
        save_sequence("Features/Spectral", key, [[feature] for feature in features], ["features"])

    print(f"Finished processing {count} files.")