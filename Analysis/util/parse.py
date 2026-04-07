#! /usr/bin/env python

import argparse as ap

def parse_arguments():
    parser = ap.ArgumentParser(description="Data Analysis and Modelling Script")

    parser.add_argument("-c", "--preprocess-calibrate", type=bool, default=False, help="Preprocessing calibration")
    parser.add_argument("-f", "--preprocess-cutoff", type=int, default=20, help="Preprocessing filer cutoff")
    parser.add_argument("-t", "--preprocess-trim", type=bool, default=False, help="Preprocessing trimming")

    parser.add_argument("-j", "--augment-jerk", type=bool, default=False, help="Augmentation jerk")
    parser.add_argument("-m", "--augment-mags", type=bool, default=False, help="Augmentation magnitudes")

    parser.add_argument("-w", "--window-data", type=bool, default=False, help="Window size")
    parser.add_argument("-s", "--window-size", type=int, default=250, help="Window size")
    parser.add_argument("-p", "--window-hop", type=int, default=125, help="Window hop")

    parser.add_argument("-a", "--affect", type=str, required=True, help="Mode, 'stress' or 'fatigue'")

    return parser.parse_args()