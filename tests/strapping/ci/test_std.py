from typing import Union

import numpy as np
import pytest

from strapping.stats import pooled_std


@pytest.mark.parametrize(
    "x1,x2,expected", [
        (
            np.array([1, 2, 3, 4]),
            np.array([10, 20, 30]),
            7.62
        ),
        (
            np.array([
                [1, 2, 3, 4],
                [4, 6, 3, 10],
                [5, 8, 3, 5],
            ]),
            np.array([
                [10, 20, 30, 40],
                [20, 40, 60, 50]
            ]),
            np.array([1.21097018, 1.41290204, 0.57735027, 1.4432993])
        ),
    ]
)
def test_pooled_std(x1: np.ndarray, x2: np.ndarray, expected: Union[np.ndarray, float]):
    _std = pooled_std(x1, x2)
    if isinstance(expected, float):
        assert np.round(_std, 2) == expected
    else:
        assert np.allclose(_std, expected)
