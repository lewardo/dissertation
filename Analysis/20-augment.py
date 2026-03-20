#! /usr/bin/env python

import numpy as np

from util.loader import load_file_keys, load_handwriting, save_sequence

CONFIG = 0

# Pipeline 0
# No augmentation

# Pipeline 1
# Series, acceleration power (pythag), gyroscope energy (sum of sq)

# Pipeline 2
# Series, acceleration jerks, acceleration power, gyroscope energy

def augment_sequence(sequence: np.array) -> np.array:
    accel, gyro = sequence[:, 1:4], sequence[:, 4:7]
    
    accel_jerks = np.diff(accel, axis=0, prepend=accel[0])
    accel_power = np.sqrt(np.sum(accel ** 2, axis=1))
    gyro_energy = np.sum(gyro ** 2, axis=1)

    # difference for correlation measures
    return np.column_stack([
        sequence, 
        accel_jerks,
        accel_power, 
        gyro_energy
    ])

if __name__ == "__main__":
    file_info, key_info = load_file_keys(f"Series/10-Trimmed-0")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) # ,[])

    print("Processing files...")

    count = 0
    for key in data:
        count += 1

        # augmented = augment_sequence(data[key])
        augmented = data[key]

        print(f"Saving file {count:03}/{len(data)}", end='\r')
        save_sequence(f"Series/20-Augmented-{CONFIG}", key, augmented, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"
                                                                        # , "accel_power", "gyro_energy"
                                                                        ])

    print(f"Finished processing {count} files.")
