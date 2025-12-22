#! /usr/bin/env python

import numpy as np
import scipy.signal as signal
import quaternion as quat

import matplotlib.pyplot as plt

from loader import load_file_keys, load_handwriting, save_sequence

def calibrate_sequence(sequence: np.array) -> np.array:
    rotate = quat.as_quat_array(sequence[:, 8:])
    accel = quat.from_vector_part(sequence[:, 2:5])
    gyro = quat.from_vector_part(sequence[:, 5:8])

    world_accel = rotate.conjugate() * accel * rotate
    world_gyro = rotate.conjugate() * gyro * rotate

    return np.column_stack([sequence[:, 1], quat.as_vector_part(world_accel), quat.as_vector_part(world_gyro)])

def filter_sequence(sequence: np.array) -> np.array:
    lowpass = signal.butter(N=5, Wn=15, fs=50, btype="lowpass", output="sos")
    return signal.sosfiltfilt(lowpass, sequence, axis=0)

if __name__ == "__main__":
    file_info, key_info = load_file_keys("Raw")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info, [])

    print("Processing files...")

    count = 0
    for key in data:
        count += 1

        calibrated = calibrate_sequence(data[key])
        filtered = filter_sequence(calibrated)

        print(f"Saving file {count:03}/{len(data)}", end='\r')
        save_sequence("Preprocessed", key, filtered, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

    print(f"Finished processing {count} files.")
    # plt.figure(figsize=(18, 12))
    # plt.subplot(2, 2, 1)
    # plt.plot(data[key][:, 1])
    # plt.subplot(2, 2, 3)
    # plt.specgram(data[key][:, 1], NFFT=16, noverlap=12)

    # plt.subplot(2, 2, 2)
    # plt.plot(filter_sequence(data[key])[:, 1])
    # plt.subplot(2, 2, 4)
    # plt.specgram(filter_sequence(data[key])[:, 1], NFFT=16, noverlap=12)

    # plt.tight_layout()
    # plt.savefig("figure")

    # for key in data:
    #     calibrated = calibrate_sequence(data[key])
    #     filter_sequence(calibrated)





    
