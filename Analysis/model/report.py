import os, json

import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_dataset_stats(features, classes, groups, trial_ids):
    stats = {
        'n_samples': len(classes),
        'n_participants': groups.nunique(),
        'n_trials': trial_ids.nunique(),
        'class_distribution': classes.value_counts(normalize=True).to_dict(),
        'samples_per_part_mean': groups.value_counts().mean(),
        'samples_per_part_std': groups.value_counts().std()
    }
    return stats

def analyze_variability(features, groups):
    if not isinstance(features, pd.DataFrame):
        return None
    
    part_means = features.groupby(groups).mean()
    between_var = part_means.var()
    part_vars = features.groupby(groups).var()
    within_var = part_vars.mean()
    
    variability_df = pd.DataFrame({
        'Between_Subject_Var': between_var,
        'Within_Subject_Var': within_var,
        'B/W_Ratio': between_var / within_var.clip(lower=1e-6)
    }).sort_values(by='B/W_Ratio', ascending=False)
    
    return variability_df

def get_detailed_cv_summary(cv_results):
    cv_report = pd.DataFrame(cv_results).drop(columns=['estimator'])
    cv_report.index = [f"Fold {i+1}" for i in range(len(cv_report))]
    
    if 'train_accuracy' in cv_report.columns and 'test_accuracy' in cv_report.columns:
        cv_report['overfit_gap'] = cv_report['train_accuracy'] - cv_report['test_accuracy']
    
    mean_row = cv_report.mean().to_frame().T
    mean_row.index = ['Mean']
    std_row = cv_report.std().to_frame().T
    std_row.index = ['Std Dev']
    
    return pd.concat([cv_report, mean_row, std_row])

def get_prediction_results(cv_results, features, classes, groups, trial_ids, group_fold):
    all_targets, all_predictions = [], []
    all_trial_results = []
    participant_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(group_fold.split(features, classes, groups=groups)):
        model = cv_results['estimator'][fold_idx]
        
        X_test = features.iloc[test_idx] if isinstance(features, pd.DataFrame) else features[test_idx]
        y_test, g_test, t_test = classes.iloc[test_idx], groups.iloc[test_idx], trial_ids.iloc[test_idx]
        
        y_pred = model.predict(X_test)
        all_targets.extend(y_test.tolist())
        all_predictions.extend(y_pred.tolist())
        
        probs = model.predict_proba(X_test)
        prob_cols = [f"prob_{c}" for c in model.classes_]
        
        res_df = pd.DataFrame(probs, columns=prob_cols, index=y_test.index)
        res_df['trial_id'], res_df['participant'], res_df['true_class'] = t_test.values, g_test.values, y_test.values

        trial_summary = res_df.groupby('trial_id').agg({
            **{col: 'mean' for col in prob_cols},
            'true_class': 'first', 'participant': 'first', 'trial_id': 'count'
        }).rename(columns={'trial_id': 'window_count'})

        trial_summary['pred_class'] = trial_summary[prob_cols].idxmax(axis=1).str.replace('prob_', '').astype(y_test.dtype)
        trial_summary['confidence'] = trial_summary[prob_cols].max(axis=1)
        
        sorted_p = np.sort(trial_summary[prob_cols].values, axis=1)
        trial_summary['margin'] = sorted_p[:, -1] - sorted_p[:, -2]
        trial_summary['is_correct'] = trial_summary['pred_class'] == trial_summary['true_class']
        all_trial_results.append(trial_summary)

    full_trials = pd.concat(all_trial_results)
    
    for part in full_trials['participant'].unique():
        p_sub = full_trials[full_trials['participant'] == part]
        participant_metrics.append({
            'Participant': part,
            'Trial_Accuracy': accuracy_score(p_sub['true_class'], p_sub['pred_class']),
            'Avg_Confidence': p_sub['confidence'].mean(),
            'Total_Trials': len(p_sub)
        })

    return {
        'window_y_true': all_targets,
        'window_y_pred': all_predictions,
        'trial_data': full_trials,
        'participant_df': pd.DataFrame(participant_metrics).set_index('Participant').sort_values('Trial_Accuracy')
    }

def get_stable_features(cv_results, features):
    feature_counts = pd.Series(0, index=features.columns)
    try:
        for model in cv_results['estimator']:
            # Compatibility for Pipeline(reducer -> selector) or simple selector
            if 'reducer' in model.named_steps and 'selector' in model.named_steps:
                mask = model.named_steps['reducer'].get_support()
                sub_mask = model.named_steps['selector'].get_support()
                selected = features.columns[mask][sub_mask]
            elif 'selector' in model.named_steps:
                selected = features.columns[model.named_steps['selector'].get_support()]
            else:
                continue
            feature_counts[selected] += 1
        return feature_counts[feature_counts == len(cv_results['estimator'])].index.tolist()
    except:
        return []

def generate_full_report(pipeline, cv_results, features, classes, groups, group_fold, trial_ids, args_dict=None):    
    results = {
        'dataset_stats': get_dataset_stats(features, classes, groups, trial_ids),
        'variability': analyze_variability(features, groups),
        'cv_summary': get_detailed_cv_summary(cv_results),
        'preds': get_prediction_results(cv_results, features, classes, groups, trial_ids, group_fold),
        'stable_features': get_stable_features(cv_results, features),
        'pipeline_str': str(pipeline),
        'args_dict': vars(args_dict)
    }

    print("\n" + "="*30 + " MODEL SUMMARY " + "="*30)
    print(results['cv_summary'][['test_accuracy', 'test_f1']].round(3))
    print(f"\nGlobal Trial Accuracy: {accuracy_score(results['preds']['trial_data']['true_class'], results['preds']['trial_data']['pred_class']):.3f}")
    
    return results

def save_report_to_file(report_bundle):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    filename = os.path.join("reports", f"comprehensive_modelling_report_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"COMPREHENSIVE MODELLING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        if report_bundle['args_dict']:
            f.write("# RUN CONFIGURATION\n" + "-" * 30 + "\n")
            f.write(json.dumps(report_bundle['args_dict'], indent=4) + "\n\n")
        
        f.write("# PIPELINE STRUCTURE\n" + "-" * 30 + "\n")
        f.write(report_bundle['pipeline_str'] + "\n\n")

        f.write("# DATASET PROFILE\n" + "-" * 30 + "\n")
        for k, v in report_bundle['dataset_stats'].items():
            f.write(f"{k}: {v}\n")
        
        if report_bundle['variability'] is not None:
            f.write("\n# VARIABILITY (Top 15 Features by B/W Ratio)\n")
            f.write(report_bundle['variability'].head(15).to_string() + "\n\n")

        f.write("# CROSS-VALIDATION SUMMARY\n" + "-" * 30 + "\n")
        f.write(report_bundle['cv_summary'].to_string() + "\n\n")

        f.write("# WINDOW-LEVEL CLASSIFICATION REPORT\n" + "-" * 30 + "\n")
        f.write(classification_report(report_bundle['preds']['window_y_true'], report_bundle['preds']['window_y_pred']))
        f.write("\n")

        tr = report_bundle['preds']['trial_data']
        f.write("# TRIAL-LEVEL PERFORMANCE\n" + "-" * 30 + "\n")
        f.write(f"Global Trial Accuracy: {accuracy_score(tr['true_class'], tr['pred_class']):.4f}\n")
        f.write(f"Avg Confidence:        {tr['confidence'].mean():.4f}\n")
        f.write(f"Avg Margin:            {tr['margin'].mean():.4f}\n\n")
        
        labels = sorted(tr['true_class'].unique())
        cm = confusion_matrix(tr['true_class'], tr['pred_class'], labels=labels)
        f.write("## Trial Confusion Matrix\n")
        f.write(pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels]).to_string() + "\n\n")

        f.write("# PER-PARTICIPANT BREAKDOWN (TRIAL-LEVEL)\n" + "-" * 30 + "\n")
        f.write(report_bundle['preds']['participant_df'].to_string() + "\n\n")

        f.write("# FEATURE STABILITY\n" + "-" * 30 + "\n")
        f.write(f"Stable in 100% of folds ({len(report_bundle['stable_features'])} features):\n")
        f.write(", ".join(report_bundle['stable_features']) + "\n")

    print(f"Report saved to: {filename}")