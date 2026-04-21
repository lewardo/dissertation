#! /usr/bin/env python

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.base import TransformerMixin, BaseEstimator

# from skrebate import ReliefF

class SeriesReshaper(BaseEstimator, TransformerMixin):
    def __init__(self, timesteps=250, n_features=10):
        self.timesteps = timesteps
        self.n_features = n_features
    def fit(self, X, y=None): return self
    def transform(self, X):
        # Reshapes flattened input back to (samples, 250, 7)
        return X.reshape(-1, self.timesteps, self.n_features)

def create_cnn_model(meta):
    model = Sequential([
        # First Layer: Captures micro-movements (tremors)
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(250, 10)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Second Layer: Captures broader stroke patterns
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        # 3 classes: Low, Medium, High Stress
        Dense(3, activation='softmax') 
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def model_series_cnn(features_raw, classes, groups, trial_ids):
    # NOTE: features_raw must be the windows flattened to 2D for the Pipeline
    # Shape: (N_windows, 250*7)
    print("*" * 50 + f"{features_raw.shape}")
    
    folds = StratifiedGroupKFold(n_splits=4)
    
    # We wrap the Keras model to work in the pipeline
    clf = KerasClassifier(
        model=create_cnn_model,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_split=0.1,
        class_weight="balanced" # Vital for your imbalanced Class 1
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Scaler works on flattened 2D data
        ('reshaper', SeriesReshaper(timesteps=250, n_features=10)),
        ('classifier', clf)
    ])

    # trial_ids = features_raw.index.to_series()
    cv_results = cross_validate(
        pipeline, features_raw, classes, 
        groups=groups,
        cv=folds,
        scoring=['accuracy', 'f1_weighted'],
        return_train_score=True,
        return_estimator=True,
        n_jobs=1 # Deep learning usually doesn't play well with n_jobs=-1
    )
    
    return pipeline, cv_results, features_raw, classes, groups, folds, trial_ids

def model_series(features: pd.DataFrame, classes: pd.Series, groups: pd.Series):
    # Build the pipeline
    folds = StratifiedGroupKFold(n_splits=4)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', SelectKBest(f_classif, k=100)),
        ('selector', RFE(
            estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
            n_features_to_select=10,
            step=1
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=500, 
            max_depth=5,
            class_weight='balanced',
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
