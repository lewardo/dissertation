import os, datetime, json
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_full_report(pipeline, cv_results, features, classes, groups, group_fold, trial_ids):
    """
    Reporting adjusted for CNN (NumPy input) and removal of RFE feature stability.
    """
    cv_report = pd.DataFrame(cv_results)
    cv_report.index = [f"Fold {i+1}" for i in range(len(cv_report))]
    
    # Standard CV stats
    numeric_report = cv_report.drop(columns=['estimator'])
    if 'train_accuracy' in numeric_report.columns:
        numeric_report['overfit_gap'] = numeric_report['train_accuracy'] - numeric_report['test_accuracy']
    
    summary_df = pd.concat([numeric_report, numeric_report.mean().to_frame().T.rename(index={0:'Mean'}), 
                            numeric_report.std().to_frame().T.rename(index={0:'Std Dev'})])
    
    print("# Model config")
    print(str(pipeline))

    # Data collection for trial-level analysis
    all_trial_true = []
    all_trial_pred = []
    all_trial_results = []
    participant_results = []
    
    # group_fold.split requires (X, y, groups)
    for fold_idx, (train_idx, test_idx) in enumerate(group_fold.split(features, classes, groups=groups)):
        model = cv_results['estimator'][fold_idx]
        
        # NumPy indexing for features, Pandas indexing for Series
        test_data = features[test_idx] 
        test_target = classes.iloc[test_idx]
        ids_test = trial_ids.iloc[test_idx]
        parts_test = groups.iloc[test_idx]
        
        # Get probabilities from KerasClassifier
        window_probs = model.predict_proba(test_data)
        prob_cols = [f"prob_{c}" for c in model.classes_]
        
        # Build window-level result table
        results_df = pd.DataFrame(window_probs, columns=prob_cols, index=test_target.index)
        results_df['trial_id'] = ids_test.values
        results_df['participant'] = parts_test.values
        results_df['true_class'] = test_target.values

        # Aggregate Windows -> Trial (Mean Probability)
        trial_summary = results_df.groupby('trial_id').agg({
            **{col: 'mean' for col in prob_cols},
            'true_class': 'first',
            'participant': 'first'
        })
        
        # Predict based on average window probability
        mean_probs = trial_summary[prob_cols]
        trial_summary['pred_class'] = mean_probs.idxmax(axis=1).str.replace('prob_', '').astype(int)
        trial_summary['confidence'] = mean_probs.max(axis=1)
        
        # Margin: Difference between top two classes
        sorted_probs = np.sort(mean_probs.values, axis=1)
        trial_summary['margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
        trial_summary['is_correct'] = trial_summary['pred_class'] == trial_summary['true_class']

        all_trial_true.extend(trial_summary['true_class'].tolist())
        all_trial_pred.extend(trial_summary['pred_class'].tolist())
        all_trial_results.append(trial_summary)

        # Per-participant trial accuracy
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
    
    # Global trial stats
    labels = sorted(classes.unique())
    cm = confusion_matrix(all_trial_true, all_trial_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])

    conf_stats = {
        'overall_avg_conf': full_trial_results['confidence'].mean(),
        'correct_avg_conf': full_trial_results[full_trial_results['is_correct']]['confidence'].mean(),
        'incorrect_avg_conf': full_trial_results[~full_trial_results['is_correct']]['confidence'].mean(),
        'avg_margin': full_trial_results['margin'].mean()
    }

    trial_df = pd.DataFrame(participant_results).set_index('Participant').sort_values('Trial_Accuracy')
    
    # We return None for stable_features because it's a CNN
    return pipeline, summary_df, None, trial_df, None, cm_df, conf_stats

def save_report_to_file(args_dict, pipeline, summary_df, part_df, trial_df, stable_features, cm_df, conf_stats):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    filename = os.path.join("reports", f"cnn_modelling_report_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CNN MODELLING REPORT (RAW SENSOR INPUT)\n")
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("# RUNNING CONFIG\n")
        f.write(json.dumps(vars(args_dict), indent=4) + "\n\n")

        f.write("# CV RESULTS (WINDOW-LEVEL)\n")
        f.write(summary_df.to_string() + "\n\n")

        f.write("# TRIAL-BASED STATS (AGGREGATED WINDOWS)\n")
        f.write(f"Average Prediction Confidence: {conf_stats['overall_avg_conf']:.4f}\n")
        f.write(f"Avg Confidence when Correct:   {conf_stats['correct_avg_conf']:.4f}\n")
        f.write(f"Avg Confidence when Incorrect: {conf_stats['incorrect_avg_conf']:.4f}\n")
        f.write(f"Average Prediction Margin:     {conf_stats['avg_margin']:.4f}\n\n")

        f.write("# TRIAL CONFUSION MATRIX\n")
        f.write(cm_df.to_string() + "\n\n")

        f.write("# PER-PARTICIPANT PERFORMANCE (TRIAL-LEVEL)\n")
        f.write(trial_df.to_string() + "\n")
            
    print(f"CNN Report saved to: {filename}")