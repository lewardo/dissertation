#! /usr/bin/env python

import numpy as np
import pandas as pd
import pickle as pkl

from util.parse import parse_arguments
from util.load import load_trial_keys, filter_trials
from util.classes import get_key_class, get_key_participant

from process.preprocess import calibrate_sequence, filter_sequence, trim_sequence
from process.augment import augment_sequence, get_augmented_labels
from process.window import window_sequence

from model.extract import extract_features
from model.model import model_series, model_series_cnn
from model.report import generate_full_report, save_report_to_file

bad_participants = [0]

if __name__ == "__main__":
    arguments = parse_arguments()
    # intro & methodology
    print("Loading file keys")
    trial_info, all_keys = load_trial_keys("Raw")
    participant, attempt, script, stress, fatigue = all_keys
    file_names, trial_keys = filter_trials(trial_info, all_keys, [
        np.isin(participant, bad_participants, invert=True),
        # script == 0,
    ])

    print("Preprocessing trial data")
    by_participant = {}
    for file_name, trial_key in zip(file_names, trial_keys):
        trial_series = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype='float')
        
        trial_series = calibrate_sequence(trial_series, args=arguments)
        trial_series = filter_sequence(trial_series, args=arguments)
        trial_series = trim_sequence(trial_series, args=arguments)

        by_participant.setdefault(trial_key[0], []).append((trial_key, trial_series))

    print("Augmenting and windowing data")
    windows_raw, window_ids, window_trial_mappings = [], [], []
    trial_frames, trial_targets, trial_participants = [], {}, {}
    for participant_id, participant_series in by_participant.items():
        participant_data = np.concatenate([series for key, series in participant_series], axis=0)
        participant_mean = participant_data.mean(axis=0)
        participant_std = participant_data.std(axis=0).clip(min=1e-8)

        for trial_key, trial_series in participant_series:  
            trial_series = (trial_series - participant_mean) / participant_std      
            trial_series = augment_sequence(trial_series, args=arguments)
            # rank sum test
            trial_windows, window_keys = window_sequence(trial_series, trial_key, args=arguments)

            for window, window_key in zip(trial_windows, window_keys):
                if window.shape[1] != 250:
                    continue

                N, T = window.shape
                flattened = window.transpose().flatten()

                windows_raw.append(flattened)
                window_ids.append(hash(window_key))
                window_trial_mappings.append(trial_key)

                trial_target = get_key_class(window_key, mode=arguments.affect)
                trial_targets[hash(window_key)] = trial_target
                trial_participants[hash(window_key)] = get_key_participant(window_key)

                # trial_frame = pd.DataFrame(window.transpose(), columns=get_augmented_labels(arguments))

                # trial_frame['id'] = hash(window_key)
                # trial_frame['time'] = np.arange(T)

                # trial_frames.append(trial_frame)
    # features_raw = np.array(raw_windows)

    trials_classes = pd.Series(trial_targets)
    trials_participants = pd.Series(trial_participants)

    # trials_dataframe = pd.concat(trial_frames, ignore_index=True)

    valid_trials = trials_classes.notna()
    valid_ids = trials_classes[valid_trials].index

    trials_dataframe = np.array([win for win, wid in zip(windows_raw, window_ids) if wid in valid_ids])

    trials_classes = trials_classes[valid_trials].astype(int)
    trials_participants = trials_participants[valid_trials]

    trial_ids = pd.Series(
        [trial for trial, wid in zip(window_trial_mappings, window_ids) if wid in valid_ids],
        index=trials_classes.index # Keep the hashed window IDs as the index
    )

    print(trials_dataframe)

    print("Modelling data and cross-validating")
    model = model_series_cnn(trials_dataframe, trials_classes, trials_participants, trial_ids)

    print("Generating report")
    report = generate_full_report(*model)
    save_report_to_file(arguments, *report)
