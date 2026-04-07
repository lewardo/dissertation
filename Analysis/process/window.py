#! /usr/bin/env python

import numpy as np    

from numpy.lib.stride_tricks import sliding_window_view

from util.classes import generate_window_key

def window_sequence(series: np.array, key: tuple, args: dict) -> tuple:
    if not args.window_data:
        return np.array([series.transpose()]), [key]
    
    if args.window_size > series.shape[0]:
        return np.array([]), [key]

    windows = sliding_window_view(series, window_shape=args.window_size, axis=0)[::args.window_hop, ...]
    window_keys = [generate_window_key(key, i) for i, win in enumerate(windows)]

    return windows, window_keys
    