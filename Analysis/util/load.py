#! /usr/bin/env python

import os, re
import numpy as np
import pandas as pd

def load_trial_keys(path: str) -> np.array:
    file_regex = "P(\\d{3})_T(\\d{3})_(abc|dog|rick|xmas)_pss-(\\d{2})_fss-(\\d{2})(?:_\\d{8})?\\.csv"
    directory = os.path.join(os.path.curdir, os.pardir, "Data", path)

    file_info, file_keys = [], []
    for file in os.listdir(directory):
        if re.match(file_regex, file):
            file_name = os.path.join(directory, file)
            file_key = tuple([
                int(e) if e.isdigit() else ["abc", "dog", "rick", "xmas"].index(e)
                for e in re.match(file_regex, file).groups()
            ])

            file_info.append(file_name)
            file_keys.append(file_key)

    return file_info, np.array(file_keys).transpose()

def filter_trials(file_names: np.array, file_keys: np.array, filters: np.array = None) -> pd.DataFrame:
    if len(filters) == 0:
        filter = np.ones_like(file_names, dtype=bool)
    else:
        filter = np.all(filters, axis=0)

    filtered_names = list(np.array(file_names)[filter])
    filtered_keys = [tuple(key) for key in np.array(file_keys).T[filter]]

    return filtered_names, filtered_keys
