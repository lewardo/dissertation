#! /usr/bin/env python

import pickle as pkl

from util.parse import parse_arguments

from model.model import model_series
from model.report import generate_full_report, save_report_to_file

if __name__ == "__main__":
    arguments = parse_arguments()

    print("Loading saved features")
    with open('features_sj3.pkl', 'rb') as feature_file:
        trials_features, trials_classes, trials_participants = pkl.load(feature_file)

    print("Modelling data and cross-validating")
    model = model_series(trials_features, trials_classes, trials_participants)

    print("Generating report")
    report = generate_full_report(*model, arguments)
    save_report_to_file(report)
