import numpy as np
import pytest

from src.strapping.utils import repeat_array, repeat_rows


@pytest.mark.parametrize(
    "array,excepted,repeat",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 2], [3, 4], [1, 2], [3, 4]]),
            2
        ),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]]),
            3
        )
    ]
)
def test_repeat_array(array: np.ndarray, excepted: np.ndarray, repeat: int):
    assert np.array_equal(repeat_array(array, repeat), excepted)


@pytest.mark.parametrize(
    "array,excepted,repeat",
    [
        (
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 2], [1, 2], [3, 4], [3, 4]]),
                2
        ),
        (
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]),
                3
        )
    ]
)
def test_repeat_rows(array: np.ndarray, excepted: np.ndarray, repeat: int):
    assert np.array_equal(repeat_rows(array, repeat), excepted)
