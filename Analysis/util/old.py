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