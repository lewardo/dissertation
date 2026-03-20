#! /usr/bin/env python

import numpy as np

from time import strftime

# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from joblib import dump

from util.loader import load_file_keys, load_handwriting

FEATURES = "Features/30-Extracted-A1-1" # simplest features
FILTERS = "all24"

ESTIMATORS = 200

# Normalise test scores to range [0,1]
def get_key_target(key: tuple[int], mode: int) -> list[int]:
    assert(mode in ['stress', 'fatigue'], "invalid mode")
    if mode == 'stress':
        pss = key[3] # / 40
        return [pss < 14, 14 <= pss <= 26, pss > 26]
    if mode == 'fatigue':
        rest = key[4] # / 52
        return [rest < 21, 21 <= rest <= 35, rest > 35]

def train_forest_classifier(data: dict, mode: str, N: int):
    confusion, report = "", ""
    best, scores = -1, []

    participants = np.array([key[0] for key in data.keys()])

    data_samples = np.array([value.reshape(-1) for value in data.values()])
    data_targets = np.array([get_key_target(key, mode) for key in data.keys()])

    group_fold = GroupKFold(n_splits=6)

    for train_index, test_index in group_fold.split(data_samples, data_targets, groups=participants):
        train_data, test_data = data_samples[train_index], data_samples[test_index]
        train_target, test_target = data_targets[train_index], data_targets[test_index]

        classifier = RandomForestClassifier(
            n_estimators=N, 
            class_weight='balanced'
            # random_state=42
        ).fit(
            X=train_data,
            y=train_target
        )
        
        output = classifier.predict(test_data).argmax(axis=1)
        target = test_target.argmax(axis=1)

        
        score = f1_score(target, output, average='macro')
        scores.append(score)
        if score > best:
            best = score
            confusion = confusion_matrix(target, output)
            report = classification_report(target, output)
    
    feedback = '\n'.join([
        f"{mode} model report ({ESTIMATORS}-RF)\n", 
        f"{score} test accuracy ({FEATURES} {FILTERS})\n",
        f"Classifier confusion matrix",
        f"{confusion}\n",
        f"Classifier report", 
        f"{report}\n"])
    print(feedback)

    return feedback, score
    
def save_classifier(feedback: str, score: float, mode: str):
    name = f"models/{mode}/40-{mode}_{strftime("%m%d%H%M")}_{int(100*score)}"
    with open(f"{name}_report.txt", 'w') as f:
        f.write(feedback)
    # dump(classifier, f"{name}_model.gz", compress=('gzip', 9))

if __name__ == "__main__":
    file_info, key_info = load_file_keys(FEATURES)

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info)
    
    print(f"Training stress model")
    stress_model = train_forest_classifier(data, mode='stress', N=ATTEMPTS)
    save_classifier(*stress_model, 'stress')

    print("Training fatigue model")
    fatigue_model = train_classifier(data, mode='fatigue', shape=SHAPE, N=ATTEMPTS)
    save_classifier(*fatigue_model, 'fatigue')