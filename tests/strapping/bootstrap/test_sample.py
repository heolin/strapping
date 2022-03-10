from typing import Callable

import pytest
import numpy as np

from src.strapping.bootstrap import sample, sample_diffs


@pytest.mark.parametrize(
    "array,excepted,iterations,aggrfunc", [
        (
            np.array([
                [4, 20],
                [1, 40],
                [3, 30],
                [2, 30],
            ]),
            np.array([[2.25, 30.0], [2.5, 27.5]]),
            2,
            np.mean
        ),
        (
            np.array([
                [4, 20],
                [1, 40],
                [3, 30],
                [2, 30],
            ]),
            np.array([[1.29903811, 8.29156198]]),
            1,
            np.std
        )
    ]
)
def test_sample(array: np.ndarray, excepted: np.ndarray,
                iterations: int, aggrfunc: Callable, freeze_random: None):
    result = sample(array, iterations=iterations, aggrfunc=aggrfunc)
    assert np.allclose(result, excepted)
    assert result.shape == (iterations, array.shape[1])


@pytest.mark.parametrize(
    "a1,a2,excepted,iterations,repeat,aggrfunc", [
        (
            np.array([[4, 20], [1, 40], [2, 30]]),
            np.array([[4, 100], [1, 120], [3, 110]]),
            np.array([
                [-1.0, -73.33333333],
                [-2.66666667, -63.33333333],
                [-0.33333333, -80.0],
                [-2.0, -70.0]
            ]),
            2,
            2,
            np.mean
        ),
        (
            np.array([[4, 20], [1, 40], [2, 30]]),
            np.array([[4, 100], [1, 120], [3, 110]]),
            np.array([[0.47140452, 4.71404521]]),
            1,
            1,
            np.std
        )
    ]
)
def test_sample_diffs(a1: np.ndarray, a2: np.ndarray, excepted: np.ndarray,
                      iterations: int, repeat: int, aggrfunc: Callable, freeze_random: None):
    result = sample_diffs(a1, a2, iterations=iterations, aggrfunc=aggrfunc, repeat=repeat)
    assert np.allclose(result, excepted)
    assert result.shape == (iterations * repeat, a1.shape[1])
