#! /usr/bin/env python

import numpy as np
import pandas as pd

from util.parse import parse_arguments
from util.load import load_trial_keys, filter_trials
from util.classes import get_key_class, get_key_participant

from process.preprocess import calibrate_sequence, filter_sequence, trim_sequence
from process.augment import augment_sequence
from process.window import window_sequence

from model.extract import extract_features
from model.model import model_series, print_report

if __name__ == "__main__":
    arguments = parse_arguments()

    print("Loading file keys")
    trial_info, all_keys = load_trial_keys("Raw")
    participant, attempt, script, stress, fatigue = all_keys
    file_names, trial_keys = filter_trials(trial_info, all_keys, [participant > 0])

    print("Loading trial data")
    trial_frames, trial_targets, trial_participants = [], {}, {}
    for file_name, trial_key in zip(file_names, trial_keys):
        trial_series = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype='float')
        
        trial_series = calibrate_sequence(trial_series, calibrate=arguments.preprocess_calibrate)
        trial_series = filter_sequence(trial_series, cutoff=arguments.preprocess_cutoff)
        trial_series = trim_sequence(trial_series, trim=arguments.preprocess_trim)
        
        trial_series = augment_sequence(trial_series, jerk=arguments.augment_jerk, mags=arguments.augment_mags)
        
        trial_windows, window_keys = window_sequence(trial_series, trial_key, window=arguments.window_size)
        
        for window, window_key in zip(trial_windows, window_keys):
            N, T = window.shape

            trial_participants[hash(window_key)] = get_key_participant(window_key)

            trial_target = get_key_class(window_key, mode=arguments.affect)
            trial_targets[hash(window_key)] = trial_target

            trial_frame = pd.DataFrame(window.transpose(), columns=['p', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])

            trial_frame['id'] = hash(window_key)
            trial_frame['time'] = np.arange(T)

            trial_frames.append(trial_frame)

    trials_dataframe = pd.concat(trial_frames, ignore_index=True)
    trials_classes = pd.Series(trial_targets)
    trials_participants = pd.Series(trial_participants)

    print(trials_dataframe)

    # print("Extracting relevant features")
    # features, classes, groups = extract_features(trials_dataframe, trials_classes, trials_participants)
    
    print("Modelling data and cross-validating")
    cv_report = model_series(trials_dataframe, trials_classes, trials_participants)

    print("Finalising report")
    print_report(cv_report)