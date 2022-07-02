import numpy as np
import pytest

from strapping.stats import pooled_std


@pytest.mark.parametrize(
    "x1,x2,expected", [
        (
            np.array([1, 2, 3, 4]),
            np.array([10, 20, 30]),
            7.62
        )
    ]
)
def test_pooled_std(x1: np.ndarray, x2: np.ndarray, expected: float):
    assert np.round(pooled_std(x1, x2), 2) == expected
