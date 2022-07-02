import numpy as np


def pooled_std(group1: np.ndarray, group2: np.ndarray):
    """
    Calculates pooled standard deviation.
    :param group1: the first data array
    :param group2: the second data array
    :returns: computed value
    """
    s1, s2 = np.std(group1, axis=0), np.std(group2, axis=0)
    n1, n2 = group1.shape[0], group2.shape[0]
    return np.sqrt(((n1 - 1) * s1 + (n2 - 1) ** s2) / (n1 + n2 - 2))
