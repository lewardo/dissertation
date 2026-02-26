#! /usr/bin/env python

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import scipy.signal as signal
import quaternion as quat

from joblib import dump, load

from loader import load_file_keys, load_handwriting, save_sequence, PANGRAM
from visualise import show_sequence

# Normalise test scores to range [0,1]
def get_key_target(key: tuple[int]) -> list[int]:
    pss = key[3] # / 40
    rest = key[4] # / 52

    # pss_class = [pss < 14, 14 <= pss <= 26, 26 < pss]
    # rest_class = [rest < 21, 21 <= rest <= 35, 35 < rest]

    # return [pss_class, rest_class]

    low = [pss < 14, rest < 21]
    medium = [14 <= pss <= 26, 21 <= rest <= 35]
    high = [pss > 26, rest > 35]

    return [low, medium, high]

def train_classifier(data_train: np.ndarray, target_train: np.ndarray) -> MLPClassifier:
    print("Training classifier")
    return MLPClassifier(
        hidden_layer_sizes=[64, 64], 
        learning_rate='adaptive', 
        max_iter=int(1e12), 
        activation='logistic',
        solver='adam'
    ).fit(data_train, target_train)

if __name__ == "__main__":    
    file_info, key_info = load_file_keys("Features")

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info) #trial_desc == PANGRAM
    
    dataset = np.array([value.flatten() for value in data.values()])
    target = np.array([get_key_target(key) for key in data])

    data_train, data_test, target_train, target_test = train_test_split(dataset, target)

    print("Training stress model")
    stress_accuracy = 0
    for i in range(10):
        stress_candidate = train_classifier(data_train, target_train[..., 0])
        stress_score = stress_candidate.score(data_test, target_test[..., 0])

        if stress_accuracy < stress_score:
            stress_accuracy = stress_score
            stress_classifier = stress_candidate

    print("Training stress model")
    fatigue_accuracy = 0
    for i in range(10):
        fatigue_candidate = train_classifier(data_train, target_train[..., 1])
        fatigue_score = fatigue_candidate.score(data_test, target_test[..., 1])

        if fatigue_accuracy < fatigue_score:
            fatigue_accuracy = fatigue_score
            fatigue_classifier = fatigue_candidate

    # classifier = load('model.gz')
    # classifier = load('model.gz')

    print(f"Stress test accuracy: {stress_accuracy}")
    print(f"Fatigue test accuracy: {fatigue_accuracy}")

    stress_pred = stress_classifier.predict(data_test)
    stress_conf = confusion_matrix(target_test[..., 0].argmax(axis=1), stress_pred.argmax(axis=1))

    fatigue_pred = fatigue_classifier.predict(data_test)
    fatigue_conf = confusion_matrix(target_test[..., 1].argmax(axis=1), stress_pred.argmax(axis=1))

    layout = "64x64"
    dump(stress_classifier, f'models/stress_model_{layout}_{int(stress_accuracy)}.gz')
    dump(fatigue_classifier, f'models/fatigue_model_{layout}_{int(fatigue_accuracy)}.gz')

    print(stress_conf, fatigue_conf)