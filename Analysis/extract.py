#! /usr/bin/env python

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import scipy.signal as signal
import quaternion as quat

from loader import load_file_keys, load_handwriting, save_sequence
from visualise import show_sequence

# Take a (t, N) series of vectors and calculate feature-wise statistical measures
def extract_features(sequence: np.array) -> np.array:
    # sequence length
    # Each of seven axes - min, max, avg, stdev, mcr, avg power

    # Start with c. 40
    length = len(sequence)

    minimum = np.min(sequence, axis=0)
    maximum = np.max(sequence, axis=0)
    mean = np.mean(sequence, axis=0)
    deviation = np.std(sequence, axis=0)
    skew = stats.skew(sequence, axis=0)
    kurt = stats.kurtosis(sequence, axis=0)

    power = np.mean(sequence ** 2, axis=0)
    mcr = np.count_nonzero(np.diff(np.sign(sequence - mean), axis=0), axis=0)

    return np.hstack([length, minimum, maximum, mean, deviation, skew, kurt, power, mcr])

if __name__ == "__main__":
    file_info, key_info = load_file_keys("Trimmed")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) # ,[])

    print("Processing files...")

    count = 0
    for key in data:
        count += 1

        features = extract_features(data[key])

        print(f"CAlculating features for file {count:03}/{len(data)}", end='\r')
        save_sequence("Features", key, [[feature] for feature in features], ["features"])

    print(f"Finished processing {count} files.")