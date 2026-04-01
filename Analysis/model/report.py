import json, os
import pandas as pd

from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report

def generate_full_report(cv_results, features, classes, groups, group_fold):
    # =========================================================================
    # PART 1: Fold-by-Fold, Global Stats & Overfitting Check
    # =========================================================================
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
    
    print("## 1. Fold-by-Fold & Global Statistics")
    cols_to_show = ['fit_time', 'test_accuracy', 'test_f1', 'overfit_gap']
    print(summary_df[cols_to_show].round(3))
    print("-" * 65)

    # =========================================================================
    # PART 2: Performance by Participant & Confusion Matrix
    # =========================================================================
    participant_data = []
    all_y_true = []
    all_y_pred = []
    
    # Recreate the splits to see who was in the test set for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(group_fold.split(features, classes, groups=groups)):
        model = cv_results['estimator'][fold_idx]
        
        X_test = features.iloc[test_idx]
        y_test = classes.iloc[test_idx]
        groups_test = groups.iloc[test_idx]
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        
        # Calculate accuracy for each participant in this test fold
        for participant in groups_test.unique():
            mask = groups_test == participant
            part_acc = accuracy_score(y_test[mask], y_pred[mask])
            
            participant_data.append({
                'Participant': participant,
                'Fold': fold_idx + 1,
                'Accuracy': part_acc,
                'Num_Trials': mask.sum()
            })
            
    part_df = pd.DataFrame(participant_data).set_index('Participant').sort_values(by='Accuracy')
    
    print("\n## 2. Per-Participant Performance (Worst to Best)")
    print(part_df.round(3))
    print("-" * 65)
    
    print("\n## 3. Aggregated Classification Report (All Folds Combined)")
    print(classification_report(all_y_true, all_y_pred))
    print("-" * 65)

    # =========================================================================
    # PART 3: RFE Feature Selection Stability
    # =========================================================================
    print("\n## 4. Feature Selection Stability (RFE)")
    
    # Track how many times each feature was selected across the 4 folds
    feature_counts = pd.Series(0, index=features.columns)
    
    for model in cv_results['estimator']:
        # RFE stores selected features in a boolean mask called 'support_'
        selected_mask = model.named_steps['selector'].support_
        selected_features = features.columns[selected_mask]
        feature_counts[selected_features] += 1
        
    # Features selected in all 4 folds are your most robust predictors!
    stable_features = feature_counts[feature_counts == len(cv_results['estimator'])].index.tolist()
    
    print(f"Total features originally: {len(features.columns)}")
    print(f"Features selected in ALL folds: {len(stable_features)}")
    print(f"Top 10 Stable Features: {stable_features[:10]}")
    
    return summary_df, part_df, stable_features

def save_report_to_file(args_dict: dict, summary_df: pd.DataFrame, part_df: pd.DataFrame, stable_features: list):
    # 1. Generate a default filename with a timestamp if one isn't provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(os.path.curdir, "reports", f"modeling_report_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("                    COMPREHENSIVE MODELING REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Part 1: Running Configuration
        f.write("## 1. RUNNING CONFIGURATION (Command Line Args)\n")
        f.write("-" * 70 + "\n")
        # indent=4 makes it look like a beautifully spaced Python dictionary
        f.write(json.dumps(args_dict, indent=4)) 
        f.write("\n\n" + "=" * 70 + "\n\n")
        
        # Part 2: Global Stats
        f.write("## 2. FOLD-BY-FOLD & GLOBAL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(summary_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")
        
        # Part 3: Per-Participant Stats
        f.write("## 3. PER-PARTICIPANT PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(part_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n\n")
        
        # Part 4: RFE Stable Features
        f.write("## 4. RFE STABLE FEATURES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total features stable across ALL folds: {len(stable_features)}\n\n")
        f.write("Features list:\n")
        for feature in stable_features:
            f.write(f" - {feature}\n")
            
    print(f"🎉 Success! Report saved to: {filename}")