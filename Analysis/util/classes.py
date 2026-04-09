#! /usr/bin/env python

def get_key_class(key: tuple, mode: str) -> int:
    # PSS-10 boundaries
    if mode == 'stress':
        # return key[3] >= 22
        if key[3] < 14: # 14
            return 0
        if key[3] < 27: # 27
            return None
    
    # REST boundaries
    if mode == 'fatigue':
        # return key[4] >= 22
        if key[4] < 21:
            return 0
        if key[4] < 36: # 36
            return None
    
    # Otherwise
    return 1

def get_key_participant(key: tuple) -> int:
    return key[0]

def generate_window_key(key: tuple, index: int) -> tuple:
    p, a, d, s, f = key
    new_a = 100 * a + index
    return p, new_a, d, s, f