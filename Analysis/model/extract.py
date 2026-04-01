#! /usr/bin/env python

import pandas as pd
import tsfresh as tsf

from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

def extract_features(dataset: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    print("Extracting features...")
    extracted_features = tsf.extract_features(
        dataset,
        column_id='id',
        column_sort='time',
        default_fc_parameters=MinimalFCParameters()
    )

    return extracted_features.loc[classes.index]
