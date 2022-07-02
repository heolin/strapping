import numpy as np
import pytest

from strapping.stats import confidence_intervals, percentage_confidence_intervals


@pytest.mark.parametrize(
    "array,excepted_q05,excepted_mean,excepted_q95", [
        (
            np.array([
                [4, 20],
                [1, 40],
                [3, 30],
                [2, 30],
            ]),
            np.array([1.15, 21.5]),
            np.array([2.5, 30.0]),
            np.array([3.85, 38.5])
        ),
    ]
)
def test_confidence_intervals(array: np.ndarray, excepted_q05: np.ndarray,
                              excepted_mean: np.ndarray, excepted_q95: np.ndarray):
    q05, mean, q95 = confidence_intervals(array)
    assert np.allclose(q05, excepted_q05)
    assert np.allclose(mean, excepted_mean)
    assert np.allclose(q95, excepted_q95)


@pytest.mark.parametrize(
    "array,control_mean,excepted_q05,excepted_mean,excepted_q95", [
        (
                np.array([
                    [4, 20],
                    [1, 40],
                    [3, 30],
                    [2, 30],
                ]),
                np.array([3.5, 30]),
                np.array([0.32857143, 0.71666667]),
                np.array([0.71428571, 1.]),
                np.array([1.1, 1.28333333])
        ),
    ]
)
def test_percentage_confidence_intervals(array: np.ndarray, control_mean: np.ndarray, excepted_q05: np.ndarray,
                                         excepted_mean: np.ndarray, excepted_q95: np.ndarray):
    q05, mean, q95 = percentage_confidence_intervals(array, control_mean)
    assert np.allclose(q05, excepted_q05)
    assert np.allclose(mean, excepted_mean)
    assert np.allclose(q95, excepted_q95)
