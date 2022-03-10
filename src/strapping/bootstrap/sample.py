from typing import Callable

import numpy as np

from src.strapping.utils import repeat_array, repeat_rows


def sample(data: np.ndarray, iterations: int = 100, aggrfunc: Callable = np.mean) -> np.ndarray:
    batch_count, columns_count = data.shape
    idx = np.random.randint(batch_count, size=batch_count * iterations)
    sampled_data = data[idx].reshape(batch_count, iterations, columns_count)
    aggregated = np.apply_along_axis(aggrfunc, 0, sampled_data)
    return aggregated


def sample_diffs(test_data: np.ndarray, control_data: np.ndarray,
                 iterations: int, repeat: int, aggrfunc: Callable = np.mean) -> np.ndarray:
    test_samples = sample(test_data, iterations, aggrfunc)
    control_samples = sample(control_data, iterations, aggrfunc)

    test_repeated = repeat_array(test_samples, repeat)
    control_repeated = repeat_rows(control_samples, repeat)

    diffs = (test_repeated - control_repeated)
    return diffs
