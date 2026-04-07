#! /usr/bin/env python

import pandas as pd
import tsfresh as tsf

# from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.feature_extraction import EfficientFCParameters

def extract_features(dataset: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    print("Extracting features...")
    feature_subset = [
        'sum_values',
        'abs_energy',
        'mean_abs_change',
        'mean_change',
        'median',
        'mean',
        'standard_deviation',
        'skewness',
        'kurtosis',
        'root_mean_square',
        'longest_strike_below_mean',
        'longest_strike_above_mean',
        'count_above_mean',
        'count_below_mean',
        # 'benford_correlation',
        # 'time_reversal_asymmetry_statistic',
        # 'c3',
        # 'cid_ce',
        # 'symmetry_looking',
        # 'large_standard_deviation',
        # 'quantile',
        # 'autocorrelation',
        # 'agg_autocorrelation',
        # 'partial_autocorrelation',
        # 'number_cwt_peaks',
        'number_peaks',
        'binned_entropy',
        'index_mass_quantile',
        'cwt_coefficients',
        'fft_coefficient',
        'fft_aggregated',
        # 'lempel_ziv_complexity',
        'fourier_entropy',
        'permutation_entropy'
    ]

    full_efficient_settings = EfficientFCParameters()
    customFCParameters = {
        feature: full_efficient_settings[feature] 
        for feature in feature_subset 
        if feature in full_efficient_settings
    }

    extracted_features = tsf.extract_features(
        dataset,
        column_id='id',
        column_sort='time',
        default_fc_parameters=customFCParameters,
        n_jobs=4
    )

    return extracted_features.loc[classes.index]
