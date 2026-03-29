#! /usr/bin/env python

def get_key_class(key: tuple, mode: str) -> int:
    # PSS-10 boundaries
    if mode == 'stress':
        if key[3] < 14: # 14
            return 0
        if key[3] < 27: # 27
            return 1
    
    # REST boundaries
    if mode == 'fatigue':
        if key[4] < 21:
            return 0
        if key[4] < 36: # 36
            return 1
    
    # Otherwise
    return 2

def get_key_participant(key: tuple) -> int:
    return key[0]

def generate_window_key(key: tuple, index: int) -> tuple:
    p, a, d, s, f = key
    new_a = 100 * a + index
    return p, new_a, d, s, f