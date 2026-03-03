#! /usr/bin/env python

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from time import strftime

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

import scipy.signal as signal
import quaternion as quat

from joblib import dump, load

from loader import load_file_keys, load_handwriting, save_sequence, ALPHABET
from visualise import show_sequence

# Normalise test scores to range [0,1]
def get_key_target(key: tuple[int]) -> list[int]:
    pss = key[3] # / 40
    rest = key[4] # / 52

    low = [pss < 14, rest < 21]
    medium = [14 <= pss <= 26, 21 <= rest <= 35]
    high = [pss > 26, rest > 35]

    return [low, medium, high]

def split_data(data: dict) -> tuple:
    train_indices, test_indices = train_test_split(list(range(len(data))))

    train_data = np.array([value.flatten() for (key, value) in data.items() if key[0] in train_indices])
    train_target = np.array([get_key_target(key) for key in data if key[0] in train_indices])

    test_data = np.array([value.flatten() for (key, value) in data.items() if key[0] in test_indices])
    test_target = np.array([get_key_target(key) for key in data if key[0] in test_indices])

    return train_data, test_data, train_target, test_target

def attempt_train_classifier(train_data: np.ndarray, train_target: np.ndarray, shape: list[int]) -> MLPClassifier:
    print("Training classifier")
    
    # compensate for class priors with sample weights
    target_classes = np.argmax(train_target, axis=1)
    classes, class_counts = np.unique_counts(target_classes)

    target_weights = [1 / class_counts[sample] for sample in target_classes]

    return MLPClassifier(
        hidden_layer_sizes=shape, 
        learning_rate='adaptive', 
        max_iter=int(1e15), 
        activation='logistic',
        solver='adam'
    ).fit(
        X=train_data, 
        y=train_target, 
        sample_weight=target_weights
    )

def train_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, shape: list[int], N: int):
    current_score = 0
    for i in range(N):
        classifier_candidate = attempt_train_classifier(X_train, y_train, shape)
        classifier_score = classifier_candidate.score(X=X_test, y=y_test)

        if current_score < classifier_score:
            classifier, current_score = classifier_candidate, classifier_score
    
    return classifier

if __name__ == "__main__":    
    model_shape = [64, 32, 16]
    data_filters = "all"

    file_info, key_info = load_file_keys("Features")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info)
    
    train_data, test_data, train_target, test_target = split_data(data)

    print("Training stress model")
    stress_classifier = train_classifier(
        X_train=train_data,
        y_train=train_target[..., 0],
        X_test=test_data,
        y_test=test_target[..., 0],
        shape=model_shape,
        N=5
    )

    print("Training fatigue model")
    fatigue_classifier = train_classifier(
        X_train=train_data,
        y_train=train_target[..., 1],
        X_test=test_data,
        y_test=test_target[..., 1],
        shape=model_shape,
        N=5
    )

    stress_pred = stress_classifier.predict(test_data)
    print(confusion_matrix(test_target[..., 0].argmax(axis=1), stress_pred.argmax(axis=1)))
    print(classification_report(test_target[..., 0].argmax(axis=1), stress_pred.argmax(axis=1), zero_division=0))

    fatigue_pred = fatigue_classifier.predict(test_data)
    print(confusion_matrix(test_target[..., 1].argmax(axis=1), stress_pred.argmax(axis=1)))
    print(classification_report(test_target[..., 1].argmax(axis=1), fatigue_pred.argmax(axis=1), zero_division=0))

    descriptor = "x".join([str(dim) for dim in model_shape]) + data_filters
    stress_accuracy = stress_classifier.score(test_data, test_target[..., 0])
    fatigue_accuracy = fatigue_classifier.score(test_data, test_target[..., 1])
    timestamp = strftime("%m%d%H%M")

    dump(stress_classifier, f'models/stress_model_Pspl_{timestamp}_{descriptor}_{int(100*stress_accuracy)}.gz')
    dump(fatigue_classifier, f'models/fatigue_model_Pspl_{timestamp}_{descriptor}_{int(100*fatigue_accuracy)}.gz')
