from typing import Tuple

import numpy as np


def confidence_intervals(
    data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes confidence intervals based on provided data.

    :data: a 2D array containing sampled values for each metric
    :returns: a tuple containing values for quantiles and mean (0.05, mean, 0.95)
    """
    q05 = np.quantile(data, 0.05, axis=0)
    mean = data.mean(axis=0)
    q95 = np.quantile(data, 0.95, axis=0)
    return q05, mean, q95


def percentage_confidence_intervals(
    data: np.ndarray,
    control_mean: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a percentage confidence intervals for provided data.

    :data: a 2D array containing sampled values for each metric
    :control_mean: a mean value for each metric, used as a normalization factor
    :returns: a tuple containing percentage values for quantiles and mean (0.05, mean, 0.95)
    """
    q05, mean, q95 = confidence_intervals(data)
    return q05 / control_mean, mean / control_mean, q95 / control_mean
