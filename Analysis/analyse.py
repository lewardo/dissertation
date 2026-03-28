#! /usr/bin/env python

import argparse

import numpy as np
import pandas as pd
import tsfresh as tsf

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Analysis and Modelling Script")
    
    # Required Arguments
    parser.add_argument("-p", "--preprocess", type=int, default=0, help="Preprocessing config")
    parser.add_argument("-a", "--augment", type=int, default=0, help="Augmentation config")
    parser.add_argument("-f", "--features", type=int, default=0, help="Feature set from data")

    parser.add_argument("-c", "--classifier", type=str, default="rf", help="Classifier model architecture")
    
    return parser.parse_args()