#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import quaternion as quat

from util.loader import load_file_keys, load_handwriting, save_sequence

CONFIG = 0

# Config 0 (see model names)
# Calibrate, stack in order P, A{x,y,z}, G{x,y,z}
# LPBF[N=5, f=15, fs=50]
# Within 2std of M=mean(mid 75%), LPBF[N=2, f=1, fs=50] on abs dist from M

# Use the orientation quaternion to project the raw accelerometer and gyroscope axes into the page space
def calibrate_sequence(sequence: np.array) -> np.array:
    # Extract the parts of the sequence
    rotate = quat.as_quat_array(sequence[:, 8:])
    accel = quat.from_vector_part(sequence[:, 2:5])
    gyro = quat.from_vector_part(sequence[:, 5:8])

    # Project onto the world axes (the page is xy, gravity is z)
    world_accel = rotate.conjugate() * accel * rotate
    world_gyro = rotate.conjugate() * gyro * rotate

    # pressure (untouched), wax, way, waz, wgx, wgy, wgz
    return np.column_stack([sequence[:, 1], quat.as_vector_part(world_accel), quat.as_vector_part(world_gyro)])

# Low-pass a sequence using a parametrised butterworth filter
def filter_sequence(sequence: np.array, cutoff: float = 15.0) -> np.array:
    lowpass = signal.butter(N=5, Wn=cutoff, fs=50, btype="lowpass", output="sos")
    return signal.sosfiltfilt(lowpass, sequence, axis=0)

# Some sequences were started a bit early or ended a bit late, so have noise from the pen moving
# Extract the middle of the sequence, find the spread of this middle bit, then cutoff the beginning
# and end when the running average goes within the range
def trim_sequence(sequence: np.array, slice: int = 8, tolerance: float = 2.0, cutoff: float = 1.0) -> np.array:
    # Get slice
    low = len(sequence) // slice
    high = (slice - 1) * low

    # Calc stats
    mean = np.mean(sequence[low:high], axis=0)
    deviation = tolerance * np.std(sequence[low:high], axis=0)

    # Lowpass the absolute value of the signal (running average)
    lowpass = signal.butter(N=2, Wn=cutoff, fs=50, btype="lowpass", output="sos")
    amplitude = signal.sosfiltfilt(lowpass, np.abs(sequence - mean), axis=0)

    # Find the first and last indices where the running average goes within the valid region
    start = next(x for x, val in enumerate(amplitude) if np.all(val < deviation))
    end = len(sequence) - next(x for x, val in enumerate(reversed(amplitude)) if np.all(val < deviation))

    return sequence[start:end] ## return the middle bit without the noise at the start/end

if __name__ == "__main__":
    file_info, key_info = load_file_keys("Series/00-Raw")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) # ,[])

    print("Processing files...")

    count = 0
    for key in data:
        count += 1

        calibrated = calibrate_sequence(data[key])
        filtered = filter_sequence(calibrated)
        trimmed = trim_sequence(filtered)

        # print(data[key].shape, calibrated.shape, filtered.shape, trimmed.shape)

        print(f"Saving file {count:03}/{len(data)}", end='\r')
        save_sequence(f"10-Calibrated-{CONFIG}", key, filtered, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        save_sequence(f"10-Filtered-{CONFIG}", key, filtered, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        save_sequence(f"10-Trimmed-{CONFIG}", key, filtered, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

    print(f"Finished processing {count} files.")
