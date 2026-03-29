#! /usr/bin/env python

import pandas as pd
import tsfresh as tsf

import tsfresh.utilities.dataframe_functions as df_funcs

def extract_features(dataset: pd.DataFrame, targets: pd.Series, groups: pd.Series):
    extracted_features = tsf.extract_features(dataset, column_id='id', column_sort='time')
    imputed_features = df_funcs.impute(extracted_features)

    selected_features = tsf.select_features(imputed_features, targets)

    selected_targets = targets.loc[selected_features.index]
    selected_groups = groups.loc[selected_features.index]

    return selected_features, selected_targets, selected_groups
