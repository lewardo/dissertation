#! /usr/bin/env python

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}

def model_series(dataset: pd.DataFrame, classes: pd.Series, groups: pd.Series):
    dataset_dummy = pd.DataFrame(index=classes.index)
    groups = groups.loc[classes.index]

    # Initialise the augmenter
    augmenter = RelevantFeatureAugmenter(column_id='id', column_sort='time', default_fc_parameters=EfficientFCParameters())
    augmenter.set_timeseries_container(dataset)

    # Initialise the classifier
    classifier = RandomForestClassifier(n_estimators=1000, random_state=69, n_jobs=-1)

    # Build the pipeline
    group_fold = StratifiedGroupKFold(n_splits=4)
    pipeline = Pipeline([
        ('augmenter', augmenter),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    cv_results = cross_validate(
        pipeline, dataset_dummy, classes, 
        groups=groups,
        cv=group_fold,
        scoring=scoring_metrics,
        return_train_score=True
        # return_estimator=True
    )

    # clf = cv_results['estimator'][0].named_steps['classifier']
    # feature_names = cv_results['estimator'][0].named_steps['augmenter'].feature_names_

    # importance_df = pd.Series(
    #     clf.feature_importances_, 
    #     index=feature_names
    # ).sort_values(ascending=False)

    # print(importance_df.head(10))

    return pd.DataFrame(cv_results)

def print_report(cv_report: pd.DataFrame):
    cv_report.index = [f"Fold {i+1}" for i in range(len(cv_report))]

    print("--- Fold-by-Fold Performance ---")
    print(cv_report.round(3))

    mean_row = cv_report.mean().to_frame().T
    mean_row.index = ['Mean']

    std_row = cv_report.std().to_frame().T
    std_row.index = ['Std Dev']

    full_report = pd.concat([cv_report, mean_row, std_row])
    columns_to_show = ['fit_time', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    print("\n--- Final CV Summary Report ---")
    print(full_report[columns_to_show].round(3))