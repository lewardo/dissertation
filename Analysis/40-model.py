#! /usr/bin/env python

import numpy as np

from time import strftime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from util.loader import load_file_keys, load_handwriting

# feature parameters
FEATURES = "Features/30-Extracted-A2-2W"
FILTERS = "all24" # []

# model parameters
MODEL = 'rf'

# mlp parameters
LAYERS = [64, 64]
ACTIVATION = 'relu'
SOLVER = 'adam'

# rf parameters
ESTIMATORS = 500

def classifier_instance():
    if MODEL == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=LAYERS,
            activation=ACTIVATION,
            solver=SOLVER,
            learning_rate='adaptive',
            max_iter=int(1e4),
            random_state=69
        )
    if MODEL == 'rf':
        return RandomForestClassifier(
            n_estimators=ESTIMATORS, 
            class_weight='balanced_subsample',
            max_features='log2',
            random_state=69
        )
    # if MODEL == 'xgb':
    #     return 
    
def classifier_description():
    if MODEL == 'mlp':
        dims = 'x'.join([str(dim) for dim in LAYERS])
        return f"mlp-{dims}-{ACTIVATION} ({SOLVER})"
    if MODEL == 'rf':
        return f"rf-{ESTIMATORS}"

# Classify test scores
def get_key_target(key: tuple[int], mode: str) -> int:
    # PSS-10 boundaries
    if mode == 'stress':
        if key[3] < 14:
            return 0
        if key[3] < 27:
            return 1
    
    # REST boundaries
    if mode == 'fatigue':
        if key[4] < 21:
            return 0
        if key[4] < 36:
            return 1
    
    # Otherwise
    return 2
    
def train_forest_classifier(data: dict, mode: str):
    data_participants = np.array([key[0] for key in data.keys()])

    data_samples = np.array([value.reshape(-1) for value in data.values()])
    data_targets = np.array([get_key_target(key, mode) for key in data.keys()])
    
    group_fold = StratifiedGroupKFold(n_splits=6)
    
    test_targets, test_predictions = np.array([]), np.array([])
    for train_index, test_index in group_fold.split(data_samples, data_targets, groups=data_participants):
        train_data, test_data = data_samples[train_index], data_samples[test_index]
        train_target, test_target = data_targets[train_index], data_targets[test_index]

        classifier = classifier_instance().fit(train_data, train_target)
        prediction = classifier.predict(test_data)
        
        test_targets = np.append(test_targets, test_target)
        test_predictions = np.append(test_predictions, prediction)
    
    print(test_targets.shape, test_predictions.shape)
    mean_score = f1_score(test_targets, test_predictions, average='macro', zero_division=0.0)
    global_confusion = confusion_matrix(test_targets, test_predictions)
    global_report = classification_report(test_targets, test_predictions, zero_division=0.0)
    
    feedback = '\n'.join([
        f"--- {mode.upper()} MODEL REPORT ({classifier_description()}) ---", 
        f"Trained on {FEATURES} {FILTERS}, split into 250/100 chunks",
        f"F1-Score: {mean_score:.4f}\n",
        f"Global Confusion Matrix:",
        f"{global_confusion}\n",
        f"Global Classification Report:", 
        f"{global_report}\n"
    ])
    
    print(feedback)
    return feedback, mean_score

def save_classifier(feedback: str, score: float, mode: str):
    name = f"models/40-{mode}-{MODEL}_{strftime("%m%d%H%M")}_{int(100 * score + 0.5)}"
    with open(f"{name}_report.txt", 'w') as f:
        f.write(feedback)
    # dump(classifier, f"{name}_model.gz", compress=('gzip', 9))

if __name__ == "__main__":
    file_info, key_info = load_file_keys(FEATURES)

    participant_id, trial_no, trial_desc, pss_score, rest_score = key_info
    data = load_handwriting(file_info)
    
    print(f"Training stress model")
    stress_model = train_forest_classifier(data, mode='stress')
    save_classifier(*stress_model, 'stress')

    print("Training fatigue model")
    fatigue_model = train_forest_classifier(data, mode='fatigue')
    save_classifier(*fatigue_model, 'fatigue')