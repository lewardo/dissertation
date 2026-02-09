#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def show_sequence(sequence: np.array, indices: list[int] = []) -> np.array:
    if indices != []:
        sequence = sequence.T[indices].T

    plt.figure(figsize=(12,12))
    plt.tight_layout()

    plt.xlabel("Samples")
    plt.ylabel("Amp")
    plt.plot(np.linspace(0, len(sequence)) * 0.02, sequence)

    plt.show()

def show_accel(sequence: np.array) -> np.array:
    plt.figure(figsize=(12,12))
    plt.tight_layout()

    plt.xlabel("Samples")
    plt.ylabel("")
    plt.plot(np.linspace(0, len(sequence)) * 0.02, sequence)