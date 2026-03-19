#! /usr/bin/env python

import numpy as np

from time import strftime

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, f1_score

from joblib import dump

from util.loader import load_file_keys, load_handwriting

FEATURES = "Features/30-Standard-1"
FILTERS = "all24"

SHAPE = [128]
OPTIMISER = 'adam'
ACTIVATION = 'relu'

MODEL_CONFIG = "all24-p0-a1-e1"
ATTEMPTS = 10

# Normalise test scores to range [0,1]
def get_key_target(key: tuple[int], mode: int) -> list[int]:
    if mode == 'stress':
        pss = key[3] # / 40
        return [pss < 14, 14 <= pss <= 26, pss > 26]
    
    if mode == 'fatigue':
        rest = key[4] # / 52
        return [rest < 21, 21 <= rest <= 35, rest > 35]

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
    
    train_data = np.array([value.reshape(-1) for (key, value) in data.items() if key[0] in train_indices])
    train_target = np.array([get_key_target(key, mode) for key in data if key[0] in train_indices])

    test_data = np.array([value.reshape(-1) for (key, value) in data.items() if key[0] in test_indices])
    test_target = np.array([get_key_target(key, mode) for key in data if key[0] in test_indices])

    return train_data, test_data, train_target, test_target

def get_sample_weights(train_target: np.ndarray) -> np.ndarray:
    classes = np.argmax(train_target, axis=1)
    counts = np.unique_counts(classes)

    return [np.max(counts) / counts.counts[sample] for sample in classes]

def check_targets_stratified(train_target: np.ndarray, test_target: np.ndarray) -> bool:
    train_classes = np.unique(np.argmax(train_target, axis=1))
    test_classes = np.unique(np.argmax(test_target, axis=1))

    return train_classes.shape == test_classes.shape

def train_classifier(data: dict, mode: str, shape: list[int], N: int):
    classifier = None
    confusion, report = "", ""
    score = -1

    for i in range(N):
        print(f"Attempting model {i+1} of {N}")

        train_data, test_data, train_target, test_target = split_data(data, mode=mode)

        while not check_targets_stratified(train_target, test_target):
            train_data, test_data, train_target, test_target = split_data(data, mode=mode)
            continue

        classifier_candidate = MLPClassifier(
            hidden_layer_sizes=shape, 
            activation=ACTIVATION,
            # verbose=True,

            ## LGFBS
            # max_iter=int(2e2), 
            # solver='lbfgs',

            ## SGD
            solver=OPTIMISER,
            learning_rate='adaptive',
            # learning_rate_init=5e-3,
            tol=1e-10,
            max_iter=int(1e9)
        ).fit(
            X=train_data,
            y=train_target,
            sample_weight=get_sample_weights(train_target)
        )
        
        candidate_output = classifier_candidate.predict(test_data).argmax(axis=1)
        target_output = test_target.argmax(axis=1)

        test_weights = get_sample_weights(test_target)
        candidate_score = f1_score(
            target_output, 
            candidate_output, 
            average='micro',
            sample_weight=test_weights
        )

        print(f"\tScored {candidate_score}")
        if score < candidate_score:
            classifier, score = classifier_candidate, candidate_score
            confusion = confusion_matrix(target_output, candidate_output, sample_weight=test_weights)
            report = classification_report(target_output, candidate_output, sample_weight=test_weights)
    
    feedback = '\n'.join([
        f"{mode} model report ({'x'.join([str(dim) for dim in shape])}-{OPTIMISER}-{ACTIVATION}, {MODEL_CONFIG})\n", 
        f"{score} test accuracy ({FEATURES})\n",
        f"Classifier confusion matrix",
        f"{confusion}\n",
        f"Classifier report", 
        f"{report}"])
    print(feedback)

    return classifier, feedback

def save_classifier(classifier: MLPClassifier, feedback: str, mode: str):
    name = f"models/40-{mode}_Pspl_{strftime("%m%d%H%M")}_{MODEL_CONFIG}_{'x'.join([str(dim) for dim in SHAPE])}_{OPTIMISER}-{ACTIVATION}"
    with open(f"{name}_report.txt", 'w') as f:
        f.write(feedback)
    dump(classifier, f"{name}_model.gz", compress=('gzip', 9))

if __name__ == "__main__":
    file_info, key_info = load_file_keys(FEATURES)

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info)
    
    print(f"Training stress model")
    stress_model = train_classifier(data, mode='stress', shape=SHAPE, N=ATTEMPTS)
    save_classifier(*stress_model, 'stress')

    print("Training fatigue model")
    fatigue_model = train_classifier(data, mode='fatigue', shape=SHAPE, N=ATTEMPTS)
    save_classifier(*fatigue_model, 'fatigue')