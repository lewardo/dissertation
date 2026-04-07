import json, os
import pandas as pd
import sklearn as sk
import numpy as np

from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_full_report(pipeline, cv_results, features, classes, groups, group_fold, trial_ids):
    cv_report = pd.DataFrame(cv_results)
    cv_report.index = [f"Fold {i+1}" for i in range(len(cv_report))]
    
    numeric_report = cv_report.drop(columns=['estimator'])
    
    if 'train_accuracy' in numeric_report.columns:
        numeric_report['overfit_gap'] = numeric_report['train_accuracy'] - numeric_report['test_accuracy']
    
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
    
    for fold_idx, (train_idx, test_idx) in enumerate(group_fold.split(features, classes, groups=groups)):
        model = cv_results['estimator'][fold_idx]
        
        test_data = features.iloc[test_idx]
        test_target = classes.iloc[test_idx]
        groups_test = groups.iloc[test_idx]
        
        test_predictions = model.predict(test_data)
        
        all_targets.extend(test_target.tolist())
        all_predictions.extend(test_predictions.tolist())
        
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
    
    feature_counts = pd.Series(0, index=features.columns)
    
    for model in cv_results['estimator']:
        reduced_mask = model.named_steps['reducer'].get_support()
        selected_mask = model.named_steps['selector'].support_
        selected_features = features.columns[reduced_mask][selected_mask]
        feature_counts[selected_features] += 1
        
    stable_features = feature_counts[feature_counts == len(cv_results['estimator'])].index.tolist()
    
    print(f"Total features originally: {len(features.columns)}")
    print(f"Features selected in ALL folds: {len(stable_features)}")
    print(f"Stable Features: {stable_features}")

    all_trial_true = []
    all_trial_pred = []
    all_trial_conf = []
    all_trial_results = []
    participant_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(group_fold.split(features, classes, groups=groups)):
        model = cv_results['estimator'][fold_idx]
        
        test_data = features.iloc[test_idx]
        test_target = classes.iloc[test_idx]
        ids_test = trial_ids.iloc[test_idx]
        parts_test = groups.iloc[test_idx]
        
        window_probs = model.predict_proba(test_data)
        prob_cols = [f"prob_{c}" for c in model.classes_]
        
        results_df = pd.DataFrame(window_probs, columns=prob_cols, index=test_data.index)
        results_df['trial_id'] = ids_test.values
        results_df['participant'] = parts_test.values
        results_df['true_class'] = test_target.values

        trial_summary = results_df.groupby('trial_id').agg({
            **{col: 'mean' for col in prob_cols},
            'true_class': 'first',
            'participant': 'first',
            'trial_id': 'count' # This counts the number of windows
        }).rename(columns={'trial_id': 'window_count'})

        mean_probs = trial_summary[prob_cols]
        
        trial_summary['pred_class'] = mean_probs.idxmax(axis=1).str.replace('prob_', '').astype(test_target.dtype)
        trial_summary['confidence'] = mean_probs.max(axis=1)
        
        sorted_probs = np.sort(mean_probs.values, axis=1)
        trial_summary['margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        trial_summary['is_correct'] = trial_summary['pred_class'] == trial_summary['true_class']

        all_trial_true.extend(trial_summary['true_class'].tolist())
        all_trial_pred.extend(trial_summary['pred_class'].tolist())
        all_trial_results.append(trial_summary)

        for part in trial_summary['participant'].unique():
            part_subset = trial_summary[trial_summary['participant'] == part]
            participant_results.append({
                'Participant': part,
                'Fold': fold_idx + 1,
                'Trial_Accuracy': accuracy_score(part_subset['true_class'], part_subset['pred_class']),
                'Avg_Confidence': part_subset['confidence'].mean(),
                'Num_Trials': len(part_subset)
            })

    full_trial_results = pd.concat(all_trial_results)
    
    labels = sorted(classes.unique())
    cm = confusion_matrix(all_trial_true, all_trial_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])

    conf_stats = {
        'overall_avg_conf': full_trial_results['confidence'].mean(),
        'correct_avg_conf': full_trial_results[full_trial_results['is_correct']]['confidence'].mean(),
        'incorrect_avg_conf': full_trial_results[~full_trial_results['is_correct']]['confidence'].mean(),
        'avg_margin': full_trial_results['margin'].mean()
    }

    print("# TRIAL-LEVEL SUMMARY")
    print(f"Global Trial Accuracy: {accuracy_score(all_trial_true, all_trial_pred):.3f}")
    print(f"Avg Confidence (Correct): {conf_stats['correct_avg_conf']:.3f}")
    print(f"Avg Confidence (Incorrect): {conf_stats['incorrect_avg_conf']:.3f}")
    print("\n## Trial Confusion Matrix:")
    print(cm_df)

    trial_df = pd.DataFrame(participant_results).set_index('Participant').sort_values('Trial_Accuracy')
    
    return pipeline, summary_df, part_df, trial_df, stable_features, cm_df, conf_stats

def save_report_to_file(args_dict: dict, pipeline: dict, summary_df: pd.DataFrame, part_df: pd.DataFrame, trial_df: pd.DataFrame, stable_features: list, cm_df: pd.DataFrame, conf_stats: dict):
    # 1. Generate a default filename with a timestamp if one isn't provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    filename = os.path.join(os.path.curdir, "reports", f"modelling_report_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODELLING REPORT\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("# RUNNING CONFIG\n")
        f.write("-" * 70 + "\n")
        f.write(json.dumps(vars(args_dict), indent=4)) 
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("# MODELLING CONFIG\n")
        f.write("-" * 70 + "\n")
        f.write(str(pipeline)) 
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("# CV RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(summary_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")
        
        f.write("# PER-PARTICIPANT PERFORMANCE (GLOBAL)\n")
        f.write("-" * 70 + "\n")
        f.write(part_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")
        
        f.write("# STABLE FEATURES (GLOBAL)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total features stable across ALL folds: {len(stable_features)}\n\n")
        f.write("Features list:\n")
        for feature in stable_features:
            f.write(f" - {feature}\n")
        f.write("\n" + "=" * 70 + "\n\n")

        f.write("# TRIAL-BASED STATS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Prediction Confidence: {conf_stats['overall_avg_conf']:.4f}\n")
        f.write(f"Avg Confidence when Correct:   {conf_stats['correct_avg_conf']:.4f}\n")
        f.write(f"Avg Confidence when Incorrect: {conf_stats['incorrect_avg_conf']:.4f}\n")
        f.write(f"Average Prediction Margin:     {conf_stats['avg_margin']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n\n")

        f.write("# TRIAL CONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")
        f.write(cm_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")

        f.write("# PER-PARTICIPANT PERFORMANCE (TRIAL-LEVEL)\n")
        f.write("-" * 70 + "\n")
        f.write(trial_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")
            
    print(f"Report saved to: {filename}")