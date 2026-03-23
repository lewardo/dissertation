#! /usr/bin/env python

import numpy as np
from scipy import stats

from time import strftime

# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from joblib import dump

from util.loader import load_file_keys, load_handwriting

FEATURES = "Features/30-Extracted-A2-1" # simplest features
FILTERS = "all24"

ESTIMATORS = 200

# Normalise test scores to range [0,1]
def get_key_target(key: tuple[int], mode: int) -> list[int]:
    assert mode in ['stress', 'fatigue']

    pss = key[3] # / 40
    rest = key[4] # / 52
    if mode == 'stress':
        if pss < 14:
            return 0
        if pss < 27:
            return 1
    if mode == 'fatigue':
        if rest < 21:
            return 0
        if rest < 36:
            return 1
    return 2

def train_forest_classifier(data: dict, mode: str, N: int):
    scores = []

    participants = np.array([key[0] for key in data.keys()])
    data_samples = np.array([value.reshape(-1) for value in data.values()])
    data_targets = np.array([get_key_target(key, mode) for key in data.keys()])

    group_fold = GroupKFold(n_splits=6)

    all_targets = []
    all_outputs = []

    for train_index, test_index in group_fold.split(data_samples, data_targets, groups=participants):
        train_data, test_data = data_samples[train_index], data_samples[test_index]
        train_target, test_target = data_targets[train_index], data_targets[test_index]

        classifier = RandomForestClassifier(
            n_estimators=N, 
            class_weight='balanced',
            random_state=69
        ).fit(
            X=train_data,
            y=train_target
        )
        
        output = classifier.predict(test_data)     
        score = f1_score(test_target, output, average='macro', zero_division=0.0)
        scores.append(score)

        all_targets.extend(test_target)
        all_outputs.extend(output)
    
    mean_score = np.mean(scores)
    global_confusion = confusion_matrix(all_targets, all_outputs)
    global_report = classification_report(all_targets, all_outputs, zero_division=0.0)

    feedback = '\n'.join([
        f"--- {mode.upper()} MODEL REPORT ({ESTIMATORS}-RF) ---", 
        f"Average CV F1-Score: {mean_score:.4f}",
        f"Best Single Fold F1: {np.max(scores):.4f}",
        f"Worst Single Fold F1: {np.min(scores):.4f}\n",
        f"Global Confusion Matrix (Across all 6 folds):",
        f"{global_confusion}\n",
        f"Global Classification Report:", 
        f"{global_report}\n"
    ])

    print(feedback)
    return feedback, mean_score
    
def train_forest_classifier_voting(data: dict, mode: str, N: int):
    X_windows, y_windows, groups_windows, trial_keys = [], [], [], []
    
    for key, trial_windows in data.items():
        target = get_key_target(key, mode)
        participant = key[0]
        
        for window in trial_windows:
            X_windows.append(window)
            y_windows.append(target)
            groups_windows.append(participant)
            trial_keys.append(key)
            
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    groups_windows = np.array(groups_windows)
    
    group_fold = GroupKFold(n_splits=6)
    
    trial_targets, trial_predictions = [], []
    for train_index, test_index in group_fold.split(X_windows, y_windows, groups=groups_windows):
        X_train, y_train = X_windows[train_index], y_windows[train_index]
        X_test, y_test = X_windows[test_index], y_windows[test_index]
        
        test_keys = [trial_keys[i] for i in test_index]

        classifier = RandomForestClassifier(
            n_estimators=N, 
            class_weight='balanced_subsample',
            max_features='log2',
            random_state=42
        ).fit(X_train, y_train)
        
        window_predictions = classifier.predict(X_test)
        unique_test_keys = list(set(test_keys))
        
        for target_key in unique_test_keys:
            trial_preds = [pred for pred, k in zip(window_predictions, test_keys) if k == target_key]
            majority_vote = stats.mode(trial_preds, keepdims=True)[0][0]
            
            trial_targets.append(get_key_target(target_key, mode))
            trial_predictions.append(majority_vote)
            
    mean_score = f1_score(trial_targets, trial_predictions, average='macro', zero_division=0.0)
    global_confusion = confusion_matrix(trial_targets, trial_predictions)
    global_report = classification_report(trial_targets, trial_predictions, zero_division=0.0)
    
    feedback = '\n'.join([
        f"--- {mode.upper()} MODEL REPORT ({N}-RF with Majority Voting) ---", 
        f"Trial-Level F1-Score: {mean_score:.4f}",
        f"Global Confusion Matrix:",
        f"{global_confusion}\n",
        f"Global Classification Report:", 
        f"{global_report}\n"
    ])
    
    print(feedback)
    return feedback, mean_score

def save_classifier(feedback: str, score: float, mode: str):
    name = f"models/rf/{mode}/40-{mode}_{strftime("%m%d%H%M")}_{int(100*score)}"
    with open(f"{name}_report.txt", 'w') as f:
        f.write(feedback)
    # dump(classifier, f"{name}_model.gz", compress=('gzip', 9))

if __name__ == "__main__":
    file_info, key_info = load_file_keys(FEATURES)

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info)
    
    print(f"Training stress model")
    stress_model = train_forest_classifier_voting(data, mode='stress', N=ESTIMATORS)
    save_classifier(*stress_model, 'stress')

    print("Training fatigue model")
    fatigue_model = train_forest_classifier_voting(data, mode='fatigue', N=ESTIMATORS)
    save_classifier(*fatigue_model, 'fatigue')