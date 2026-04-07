#! /usr/bin/env python

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def model_series(features: pd.DataFrame, classes: pd.Series, groups: pd.Series):
    # Build the pipeline
    folds = StratifiedGroupKFold(n_splits=4)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', SelectKBest(f_classif, k=250)),
        ('selector', RFECV(
            estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
            step=0.05
        )),
        ('classifier', RandomForestClassifier(n_estimators=1000, n_jobs=-1))
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
