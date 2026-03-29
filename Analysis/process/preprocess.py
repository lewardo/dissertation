#! /usr/bin/env python

import numpy as np
import quaternion as quat
import scipy.signal as signal

# Use the orientation quaternion to project the raw accelerometer and gyroscope axes into the page space
def calibrate_sequence(sequence: np.array, calibrate: bool = 0) -> np.array:
    if calibrate:
        # Extract the parts of the sequence
        orient = quat.as_quat_array(sequence[..., 8:])
        accel = quat.from_vector_part(sequence[..., 2:5])
        gyro = quat.from_vector_part(sequence[..., 5:8])

        # Project onto the world axes (the page is xy, gravity is z)
        world_accel = orient.conjugate() * accel * orient
        world_gyro = orient.conjugate() * gyro * orient

        # pressure (untouched), wax, way, waz, wgx, wgy, wgz
        return np.column_stack([sequence[:, 1], quat.as_vector_part(world_accel), quat.as_vector_part(world_gyro)])
    else:
        # Return just the pressure and 6DOF untouched
        return sequence[..., 1:8]

# Low-pass a sequence using a parametrised butterworth filter
def filter_sequence(sequence: np.array, cutoff: float = 15.0) -> np.array:
    lowpass = signal.butter(N=5, Wn=cutoff, fs=50, btype="lowpass", output="sos")
    return signal.sosfiltfilt(lowpass, sequence, axis=0)

# Some sequences were started a bit early or ended a bit late, so have noise from the pen moving
# Extract the middle of the sequence, find the spread of this middle bit, then cutoff the beginning
# and end when the running average goes within the range
def trim_sequence(sequence: np.array, trim: bool = False) -> np.array:
    if not trim:
        return sequence
    
    low = len(sequence) // slice
    high = (slice - 1) * low

    # Calc stats
    mean = np.mean(sequence[low:high], axis=0)
    deviation = 2 * np.std(sequence[low:high], axis=0)

    # Lowpass the absolute value of the signal (running average)
    lowpass = signal.butter(N=2, Wn=1, fs=50, btype="lowpass", output="sos")
    amplitude = signal.sosfiltfilt(lowpass, np.abs(sequence - mean), axis=0)

    # Find the first and last indices where the running average goes within the valid region
    start = next(x for x, val in enumerate(amplitude) if np.all(val < deviation))
    end = len(sequence) - next(x for x, val in enumerate(reversed(amplitude)) if np.all(val < deviation))

    return sequence[start:end] ## return the middle bit without the noise at the start/end
