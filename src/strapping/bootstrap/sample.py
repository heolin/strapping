from typing import Callable

import numpy as np

from strapping.utils import repeat_array, repeat_rows


def sample(data: np.ndarray, iterations: int = 100, aggrfunc: Callable = np.mean) -> np.ndarray:
    """
    Used to sample an aggregated value for a given dataset using bootstrapping mechanism.

    :param data: input dataset in a form of 2d array (even 1D should be transform into a column vector).
    :param iterations: number of iterations for bootstrapping mechanism.
    :param aggrfunc: an aggregation function applied to sampled data.
    :returns: a vector containing sampled values
    """
    batch_count, columns_count = data.shape
    idx = np.random.randint(batch_count, size=batch_count * iterations)
    sampled_data = data[idx].reshape(batch_count, iterations, columns_count)
    aggregated = np.apply_along_axis(aggrfunc, 0, sampled_data)
    return aggregated


def sample_diffs(test_data: np.ndarray, control_data: np.ndarray,
                 iterations: int, aggrfunc: Callable = np.mean) -> np.ndarray:
    """
    Used to sample an aggregated value of differences between two given datasets using bootstrapping mechanism.
    After sampling perform additional folds on data to increase the number of tested differences to interations^2.

    :param test_data: first dataset in a form of 2d array (even 1D should be transform into a column vector).
    :param control_data: second dataset in a form of 2d array
    :param iterations: number of iterations for bootstrapping mechanism
    :param aggrfunc: an aggregation function applied to sampled data
    :returns: a vector containing sampled values
    """
    assert test_data.shape == control_data.shape

    test_samples = sample(test_data, iterations, aggrfunc)
    control_samples = sample(control_data, iterations, aggrfunc)

    test_repeated = repeat_array(test_samples, iterations)
    control_repeated = repeat_rows(control_samples, iterations)
    diffs = (test_repeated - control_repeated)
    return diffs
