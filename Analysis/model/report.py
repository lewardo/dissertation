import json, os
import pandas as pd
import sklearn as sk

from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report

def generate_full_report(pipeline, cv_results, features, classes, groups, group_fold, trial_ids):
    cv_report = pd.DataFrame(cv_results)
    cv_report.index = [f"Fold {i+1}" for i in range(len(cv_report))]
    
    # Drop the estimator objects so we can do math
    numeric_report = cv_report.drop(columns=['estimator'])
    
    # Add a column to check for Overfitting
    if 'train_accuracy' in numeric_report.columns:
        numeric_report['overfit_gap'] = numeric_report['train_accuracy'] - numeric_report['test_accuracy']
    
    # Add Mean and Std Dev
    mean_row = numeric_report.mean().to_frame().T
    mean_row.index = ['Mean']
    std_row = numeric_report.std().to_frame().T
    std_row.index = ['Std Dev']
    
    summary_df = pd.concat([numeric_report, mean_row, std_row])
    
    print("# Model config")
    sk.set_config(display='text')
    print(str(pipeline))

    print("# Model stats")
    cols_to_show = ['fit_time', 'test_accuracy', 'test_f1', 'overfit_gap']
    print(summary_df[cols_to_show].round(3))
    print("-" * 65)

    participant_data = []
    all_targets = []
    all_predictions = []
    
    # Recreate the splits to see who was in the test set for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(group_fold.split(features, classes, groups=groups)):
        model = cv_results['estimator'][fold_idx]
        
        test_data = features.iloc[test_idx]
        test_target = classes.iloc[test_idx]
        groups_test = groups.iloc[test_idx]
        
        # Generate predictions
        test_predictions = model.predict(test_data)
        
        all_targets.extend(test_target.tolist())
        all_predictions.extend(test_predictions.tolist())
        
        # Calculate accuracy for each participant in this test fold
        for participant in groups_test.unique():
            mask = groups_test == participant
            part_acc = accuracy_score(test_target[mask], test_predictions[mask])
            
            participant_data.append({
                'Participant': participant,
                'Fold': fold_idx + 1,
                'Accuracy': part_acc,
                'Num_Trials': mask.sum()
            })
            
    part_df = pd.DataFrame(participant_data).set_index('Participant').sort_values(by='Accuracy')
    
    print("\n# Per-participant performance")
    print(part_df.round(3))
    print("-" * 65)
    
    print("\n# Global classification report (all folds)")
    print(classification_report(all_targets, all_predictions))
    print("-" * 65)

    print("\n## Feature Selection Stability (RFE)")
    
    # Track how many times each feature was selected across the 4 folds
    feature_counts = pd.Series(0, index=features.columns)
    
    for model in cv_results['estimator']:
        # RFE stores selected features in a boolean mask called 'support_'
        reduced_mask = model.named_steps['reducer'].get_support()
        selected_mask = model.named_steps['selector'].support_
        selected_features = features.columns[reduced_mask][selected_mask]
        feature_counts[selected_features] += 1
        
    # Features selected in all 4 folds are your most robust predictors!
    stable_features = feature_counts[feature_counts == len(cv_results['estimator'])].index.tolist()
    
    print(f"Total features originally: {len(features.columns)}")
    print(f"Features selected in ALL folds: {len(stable_features)}")
    print(f"Stable Features: {stable_features}")

    all_trial_true = []
    all_trial_pred = []
    participant_results = []

    # Recreate the splits
    for fold_idx, (train_idx, test_idx) in enumerate(group_fold.split(features, classes, groups=groups)):
        model = cv_results['estimator'][fold_idx]
        
        # Test data for this fold
        test_data = features.iloc[test_idx]
        test_target = classes.iloc[test_idx]
        ids_test = trial_ids.iloc[test_idx]  # The 'id' column for these windows
        parts_test = groups.iloc[test_idx]   # Participant IDs
        
        # 1. Get Probabilities (Window-level)
        # Result is an array: [num_windows, num_classes]
        window_probs = model.predict_proba(test_data)
        prob_cols = [f"prob_{c}" for c in model.classes_]
        
        # 2. Create a temporary DataFrame for aggregation
        results_df = pd.DataFrame(window_probs, columns=prob_cols, index=test_data.index)
        results_df['trial_id'] = ids_test.values
        results_df['participant'] = parts_test.values
        results_df['true_class'] = test_target.values

        # 3. Aggregate by Trial ID
        # We take the mean of probabilities and the 'first' of the true_class/participant
        trial_summary = results_df.groupby('trial_id').agg({
            **{col: 'mean' for col in prob_cols},
            'true_class': 'first',
            'participant': 'first'
        })

        # 4. Determine final prediction for the trial (Argmax of mean probabilities)
        # We find which column (class) has the highest mean probability
        trial_summary['pred_class'] = trial_summary[prob_cols].idxmax(axis=1).str.replace('prob_', '').astype(test_target.dtype)

        # Store for global metrics
        all_trial_true.extend(trial_summary['true_class'].tolist())
        all_trial_pred.extend(trial_summary['pred_class'].tolist())

        # 5. Store Participant Accuracy (Trial-level)
        for part in trial_summary['participant'].unique():
            part_subset = trial_summary[trial_summary['participant'] == part]
            acc = accuracy_score(part_subset['true_class'], part_subset['pred_class'])
            participant_results.append({
                'Participant': part,
                'Fold': fold_idx + 1,
                'Trial_Accuracy': acc,
                'Num_Trials': len(part_subset)
            })

    # --- Generate Final Outputs ---
    trial_df = pd.DataFrame(participant_results).set_index('Participant').sort_values('Trial_Accuracy')
    
    print("# TRIAL-LEVEL REPORT")
    print(f"\nTotal Trials Evaluated: {len(all_trial_true)}")
    print(f"Global Trial-Level Accuracy: {accuracy_score(all_trial_true, all_trial_pred):.3f}")
    
    print("\n## Classification Report (Per Trial):")
    print(classification_report(all_trial_true, all_trial_pred))
    
    return pipeline, summary_df, part_df, trial_df, stable_features

def save_report_to_file(args_dict: dict, pipeline: dict, summary_df: pd.DataFrame, part_df: pd.DataFrame, trial_df: pd.DataFrame, stable_features: list):
    # 1. Generate a default filename with a timestamp if one isn't provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(os.path.curdir, "reports", f"modelling_report_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Modelling report\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("# Running config\n")
        f.write("-" * 70 + "\n")
        f.write(json.dumps(vars(args_dict), indent=4)) 
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("# Pipeline config\n")
        f.write("-" * 70 + "\n")
        f.write(str(pipeline)) 
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("# CV Result stats\n")
        f.write("-" * 70 + "\n")
        f.write(summary_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")
        
        f.write("# Per-participant performance\n")
        f.write("-" * 70 + "\n")
        f.write(part_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("# Per-trial performance\n")
        f.write("-" * 70 + "\n")
        f.write(trial_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")
        
        f.write("# Stable features\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total features stable across ALL folds: {len(stable_features)}\n\n")
        f.write("Features list:\n")
        for feature in stable_features:
            f.write(f" - {feature}\n")
            
    print(f"Report saved to: {filename}")