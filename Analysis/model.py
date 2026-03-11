#! /usr/bin/env python

import numpy as np

from time import strftime

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from joblib import dump

from loader import load_file_keys, load_handwriting

# Normalise test scores to range [0,1]
def get_key_target(key: tuple[int], mode: int) -> list[int]:
    if mode == 'stress':
        pss = key[3] # / 40
        return [pss < 14, 14 <= pss <= 26, pss > 26]
    
    if mode == 'fatigue':
        rest = key[4] # / 52
        return [rest < 21, 21 <= rest <= 35,rest > 35]

    return [0, 0, 0]

def split_data(data: dict, mode: str = 'stress') -> tuple:
    participant_keys = np.unique_values([key[0] for key in data])
    participant_targets = np.array([
        next(get_key_target(key, mode) for key in data if key[0] == participant_key) 
        for participant_key in participant_keys
    ])

    participant_classes = np.argmax(participant_targets, axis=1)

    # use indices bc we need to map with get key target and stuff
    train_indices, test_indices = train_test_split(participant_keys, stratify=participant_classes)

    train_data = np.array([value.flatten() for (key, value) in data.items() if key[0] in train_indices])
    train_target = np.array([get_key_target(key, mode) for key in data if key[0] in train_indices])

    test_data = np.array([value.flatten() for (key, value) in data.items() if key[0] in test_indices])
    test_target = np.array([get_key_target(key, mode) for key in data if key[0] in test_indices])

    return train_data, test_data, train_target, test_target

    # data_features = np.array([value.flatten() for (key, value) in data.items()])
    # data_target = np.array([get_key_target(key, mode) for key in data])

    # return train_test_split(data_features, data_target, stratify=data_classes)

def get_sample_weights(train_target: np.ndarray) -> np.ndarray:
    classes = np.argmax(train_target, axis=1)
    counts = np.unique_counts(classes)

    return [np.max(counts) / counts.counts[sample] for sample in classes]

def attempt_train_classifier(train_data: np.ndarray, train_target: np.ndarray, shape: list[int]) -> MLPClassifier:    
    return MLPClassifier(
        hidden_layer_sizes=shape, 
        activation='logistic',
        max_iter=int(1e12), 
        learning_rate='adaptive',
        solver='adam',
        tol=1e-9
    ).fit(
        X=train_data,
        y=train_target,
        sample_weight=get_sample_weights(train_target)
    )

def train_classifier(data: dict, mode: str, shape: list[int], N: int):
    classifier = None
    current_score = -1

    for i in range(N):
        print(f"Attempting model {i+1} of {N}")

        train_data, test_data, train_target, test_target = split_data(data, mode=mode)

        classifier_candidate = attempt_train_classifier(train_data, train_target, shape)
        classifier_score = classifier_candidate.score(
            X=test_data, 
            y=test_target
        )

        print(f"\tScored {classifier_score}")
        if current_score < classifier_score:
            classifier, current_score = classifier_candidate, classifier_score

    prediction = classifier.predict(test_data)

    print("Classifier confusion matrix\n", confusion_matrix(test_target.argmax(axis=1), prediction.argmax(axis=1)))
    print("Classifier report\n", classification_report(test_target.argmax(axis=1), prediction.argmax(axis=1), zero_division=0))
    
    return classifier, current_score

if __name__ == "__main__":
    model_shape = [64, 32]
    data_filters = "all-aug"
    attempts = 4

    file_info, key_info = load_file_keys("Features_Augmented")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info)
    
    print(f"Training stress model")
    stress_classifier, stress_score = train_classifier(data, mode='stress', shape=model_shape, N=attempts)

    print("Training fatigue model")
    fatigue_classifier, fatigue_score = train_classifier(data, mode='fatigue', shape=model_shape, N=attempts)

    descriptor = "x".join([str(dim) for dim in model_shape])
    timestamp = strftime("%m%d%H%M")

    dump(stress_classifier, f'models/stress_model_Tspl_{timestamp}_{data_filters}_{descriptor}_{int(stress_score * 100)}.gz')
    dump(fatigue_classifier, f'models/fatigue_model_Tspl_{timestamp}_{data_filters}_{descriptor}_{int(fatigue_score * 100)}.gz')
