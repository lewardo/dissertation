#! /usr/bin/env python

import os, re, csv
import numpy as np

ALPHABET = 0
PANGRAM = 1
RICKROLL = 2
CHRISTMAS = 3

def load_file_keys(path: str) -> dict:
    print("Loading file keys...")

    file_regex = "P(\\d{3})_T(\\d{3})_(abc|dog|rick|xmas)_pss-(\\d{2})_fss-(\\d{2})(?:_\\d{8})?\\.csv"
    directory = os.path.join(os.path.curdir, os.pardir, "Data", path)

    count = 0
    file_keys = []
    file_names = []
    for file in os.listdir(directory):
        count += 1
        print(f"Loading key {count:03}/{len(os.listdir(directory))}", end="\r")

        if re.match(file_regex, file):
            file_key = tuple([
                int(e) if e.isdigit() else ["abc", "dog", "rick", "xmas"].index(e)
                for e in re.match(file_regex, file).groups()
            ])

            file_keys.append(file_key)
            file_names.append(os.path.join(directory, file))
    
    file_info = (np.array(file_names), np.array(file_keys))
    file_keys = tuple(np.array(file_keys).T)

    print(f"Finished loading {len(file_names)} keys.")
    
    return file_info, file_keys # return the keys

def load_handwriting(file_info: tuple, filter: np.array = None) -> np.array:
    file_names, file_keys = file_info

    if filter is None:
        filter = np.ones_like(file_names, dtype=bool)
    else:
        filter = np.all(filter, axis=0)

    filtered_names = np.array(file_names)[filter]
    filtered_keys = np.array(file_keys)[filter]

    print("Loading data files...")

    filtered_data = {}
    count = 0

    for file_path, file_key in zip(filtered_names, filtered_keys):
        count += 1
        print(f"Loading file {count:03}/{len(filter)}", end="\r")

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            filtered_data[tuple(file_key)] = np.array(list(reader)[1:], dtype=float)

    print(f"Finished loading {count} files.")
    return filtered_data

def save_sequence(path: str, file_key: tuple, sequence: np.array, header: list[str]) -> str:
    participant, trial, descriptor, pss, rest = file_key

    directory = os.path.join(os.path.curdir, os.pardir, "Data", path)
    filename = f"P{participant:03d}_T{trial:03d}_{["abc", "dog", "rick", "xmas"][descriptor]}_pss-{pss:02d}_fss-{rest:02d}.csv"

    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows((f"{v:.6f}" for v in row) for row in sequence)
    
    return filename
