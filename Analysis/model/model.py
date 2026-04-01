#! /usr/bin/env python

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def model_series(features: pd.DataFrame, classes: pd.Series, groups: pd.Series):
    # Build the pipeline
    folds = StratifiedGroupKFold(n_splits=4)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', RFE(RandomForestClassifier(n_jobs=-1))),
        ('classifier', RandomForestClassifier(n_estimators=1000, n_jobs=-1))
    ])

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
        verbose=True
    )

    return cv_results, features, classes, groups, folds
