#! /usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# from skrebate import ReliefF

def model_series(features: pd.DataFrame, classes: pd.Series, groups: pd.Series):
    folds = StratifiedGroupKFold(n_splits=4)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', SelectKBest(f_classif, k=100)),
        ('selector', RFECV(
            estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
            # min_features_to_select=10,
            step=0.05
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=50, 
            max_depth=5,
            # class_weight='balanced',
            n_jobs=-1,
        ))
    ])

    trial_ids = features.index.to_series()
    cv_results = cross_validate(
        pipeline, features, classes, 
        groups=groups,
        cv=folds,
        scoring={
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        },
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1,
        verbose=3
    )

    return pipeline, cv_results, features, classes, groups, folds, trial_ids
