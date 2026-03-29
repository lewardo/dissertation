#! /usr/bin/env python

import numpy as np

def augment_sequence(sequence: np.array, jerk: int = 0, mags: int = 0) -> np.array:
    # Extract the right axes
    accel, gyro = sequence[..., 1:4], sequence[..., 4:7]
    
    # Caluculate the jerk from the linear acceleration data
    accel_jerks = np.diff(accel, axis=0, prepend=[accel[0]])

    # Caluclate the acceleration power, and the gyro energy
    accel_power = np.sqrt(np.sum(accel ** 2, axis=1))
    gyro_energy = np.sum(gyro ** 2, axis=1)

    # conditionally stack the extra features onto the sequence
    augmented = sequence
    augmented = np.column_stack([sequence, accel_jerks]) if jerk else augmented
    augmented = np.column_stack([augmented, accel_power, gyro_energy]) if mags else augmented

    return augmented

