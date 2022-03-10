import pytest
import numpy as np


@pytest.fixture
def freeze_random():
    np.random.seed(0)
