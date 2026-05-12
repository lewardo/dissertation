#! /usr/bin/env python

import numpy as np
import quaternion as quat
import scipy.signal as signal

def calibrate_sequence(sequence: np.array, args: dict) -> np.array:
    if args.preprocess_calibrate:
        orient = quat.as_quat_array(sequence[..., 8:])
        accel = quat.from_vector_part(sequence[..., 2:5])
        gyro = quat.from_vector_part(sequence[..., 5:8])

        world_accel = orient.conjugate() * accel * orient
        world_gyro = orient.conjugate() * gyro * orient

        # world vectros
        return np.column_stack([sequence[:, 1], quat.as_vector_part(world_accel), quat.as_vector_part(world_gyro)])
    else:
        return sequence[..., 1:8]

def filter_sequence(sequence: np.array, args: dict) -> np.array:
    lowpass = signal.butter(N=5, Wn=args.preprocess_cutoff, fs=50, btype="lowpass", output="sos")
    return signal.sosfiltfilt(lowpass, sequence, axis=0) # filtered

def trim_sequence(sequence: np.array, args: dict) -> np.array:
    if not args.preprocess_trim:
        return sequence
    
    low = len(sequence) // slice
    high = (slice - 1) * low

    mean = np.mean(sequence[low:high], axis=0)
    deviation = 2 * np.std(sequence[low:high], axis=0)

    lowpass = signal.butter(N=2, Wn=1, fs=50, btype="lowpass", output="sos")
    amplitude = signal.sosfiltfilt(lowpass, np.abs(sequence - mean), axis=0)

    start = next(x for x, val in enumerate(amplitude) if np.all(val < deviation))
    end = len(sequence) - next(x for x, val in enumerate(reversed(amplitude)) if np.all(val < deviation))

    return sequence[start:end] # return the middle bit
