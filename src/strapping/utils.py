import numpy as np


def repeat_array(data: np.ndarray, repeat: int):
    return np.vstack([data]*repeat)


def repeat_rows(data: np.ndarray, repeat: int):
    return np.repeat(data, repeats=repeat, axis=0)
