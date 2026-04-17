#! /usr/bin/env python

import pandas as pd
import tsfresh as tsf

def extract_features(dataset: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    print("Extracting features...")
    customFCParameters = {
        "fft_coefficient": [{"coeff": c, "attr": "abs"} for c in [1, 2, 5, 10, 25, 50, 75]],
        "fft_aggregated": [{"aggtype": "skew"}, {"aggtype": "kurtosis"}],
        
        "sample_entropy": None,
        "approximate_entropy": [
            {"m": 2, "r": 0.1}, 
            {"m": 2, "r": 0.5}
        ],
        "permutation_entropy": [{"tau": 1, "dimension": 3}, {"tau": 1, "dimension": 5}],
        "binned_entropy": [{"max_bins": 10}],
        "fourier_entropy": [{"bins": 10}],
        
        # Peak Analysis - Capturing hesitations
        "number_peaks": [{"n": 5}, {"n": 10}],
        "number_cwt_peaks": [{"n": 5}],

        "linear_trend": [
            {"attr": "slope"}, 
            {"attr": "intercept"}, 
            {"attr": "stderr"}
        ],

        "energy_ratio_by_chunks": [
            {"num_segments": 2, "segment_focus": 0}, # First half
            {"num_segments": 2, "segment_focus": 1}  # Second half
        ],

        "autocorrelation": [
            {"lag": 1}, 
            {"lag": 5}, 
            {"lag": 10}
        ],
        
        # Statistical - Capturing intensity
        "standard_deviation": None,
        "root_mean_square": None,
        "mean_abs_change": None,
        "variation_coefficient": None
    }

    extracted_features = tsf.extract_features(
        dataset,
        column_id='id',
        column_sort='time',
        default_fc_parameters=customFCParameters,
        n_jobs=4
    )

    return extracted_features.loc[classes.index]
