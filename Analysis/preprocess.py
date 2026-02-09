#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import quaternion as quat

from loader import load_file_keys, load_handwriting, save_sequence
from visualise import show_sequence

def calibrate_sequence(sequence: np.array) -> np.array:
    rotate = quat.as_quat_array(sequence[:, 8:])
    accel = quat.from_vector_part(sequence[:, 2:5])
    gyro = quat.from_vector_part(sequence[:, 5:8])

    world_accel = rotate.conjugate() * accel * rotate
    world_gyro = rotate.conjugate() * gyro * rotate

    # p, ax, ay, az, gx, gy, gz
    return np.column_stack([sequence[:, 1], quat.as_vector_part(world_accel), quat.as_vector_part(world_gyro)])

def filter_sequence(sequence: np.array) -> np.array:
    lowpass = signal.butter(N=5, Wn=15, fs=50, btype="lowpass", output="sos")
    return signal.sosfiltfilt(lowpass, sequence, axis=0)

def trim_sequence(sequence: np.array) -> np.array:
    SPLIT = 8

    low = len(sequence) // SPLIT
    high = (SPLIT - 1) * low

    mean = np.mean(sequence[low:high], axis=0)
    deviation = 2 * np.std(sequence[low:high], axis=0)

    lowpass = signal.butter(N=5, Wn=1, fs=50, btype="lowpass", output="sos")
    amplitude = signal.sosfiltfilt(lowpass, np.abs(sequence - mean), axis=0)

    start = next(x for x, val in enumerate(amplitude) if np.all(val < deviation))
    end = len(sequence) - next(x for x, val in enumerate(reversed(amplitude)) if np.all(val < deviation))

    return sequence[start:end] ## return the middle bit that starts with

if __name__ == "__main__":
    file_info, key_info = load_file_keys("Raw")

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

        # print(f"Drawing file {count:03}/{len(data)}", end='\r')
        # print(key)

        print(f"Saving file {count:03}/{len(data)}", end='\r')
        save_sequence("Calibrated", key, filtered, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        save_sequence("Filtered", key, filtered, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        save_sequence("Trimmed", key, filtered, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

    print(f"Finished processing {count} files.")
