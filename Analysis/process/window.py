#! /usr/bin/env python

import numpy as np    

from numpy.lib.stride_tricks import sliding_window_view

from util.classes import generate_window_key

def window_sequence(series: np.array, key: tuple, window: int) -> tuple:
    if window == 0 or window > series.shape[0]:
        # return np.array([series.transpose()]), [key]
        return np.array([]), [key]

    windows = sliding_window_view(series, window_shape=window, axis=0)[::window // 2, ...]
    window_keys = [generate_window_key(key, i) for i, win in enumerate(windows)]

    return windows, window_keys
    