#! /usr/bin/env python

stress_boundaries = [14, 27] # 14, 27
fatigue_boundaries = [21, 36] # 21, 36
biclass = False

def get_key_class(key: tuple, mode: str) -> int:
    # PSS-10 boundaries
    if mode == 'stress':
        if key[3] < stress_boundaries[0]: # 14
            return 0
        if key[3] < stress_boundaries[1]: # 27
            return None if biclass else 1
    
    # REST boundaries
    if mode == 'fatigue':
        if key[4] < fatigue_boundaries[0]:
            return 0
        if key[4] < fatigue_boundaries[1]: # 36
            return None if biclass else 1
    
    # Otherwise
    return 1 if biclass else 2

def get_key_participant(key: tuple) -> int:
    return key[0]

def generate_window_key(key: tuple, index: int) -> tuple:
    p, a, d, s, f = key
    new_a = 100 * a + index
    return p, new_a, d, s, f