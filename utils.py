import random

import numpy as np


def pad(arr, value=0):
    return np.pad(arr, 1, mode='constant', constant_values=value)


def generate_board(n, shape, fill_value, x=None, y=None):
    board = np.zeros(shape, dtype=int)
    mask = np.ones_like(board, dtype=bool)
    if x is not None:
        mask[max(x-1, 0):x+2, max(y-1, 0):y+2] = False
    place_l = random.sample(np.argwhere(mask).tolist(), k=n)
    places = np.array(place_l).T
    places = (places[0], places[1])
    for x, y in place_l:
        board[max(x-1, 0):x+2, max(y-1, 0):y+2] += 1
    board[places] = fill_value
    return board, places
