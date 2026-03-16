#! /usr/bin/env python

import numpy as np

from Analysis.util.loader import load_file_keys, load_handwriting, save_sequence

def augment_sequence(sequence: np.array) -> np.array:
    accel, gyro = sequence[:, 1:4], sequence[:, 4:7]
    
    accel_power = np.sqrt(np.sum(accel ** 2, axis=1))
    gyro_energy = np.sum(gyro ** 2, axis=1)

    # difference for correlation measures
    return np.column_stack([sequence, accel_power, gyro_energy])

if __name__ == "__main__":
    file_info, key_info = load_file_keys("Series/Trimmed")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) # ,[])

    print("Processing files...")

    count = 0
    for key in data:
        count += 1

        augmented = augment_sequence(data[key])

        print(f"Saving file {count:03}/{len(data)}", end='\r')
        save_sequence("Series/Augmented", key, augmented, ["pressure", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "accel_power", "gyro_energy"])

    print(f"Finished processing {count} files.")
