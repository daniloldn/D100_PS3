import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    rng = np.random.RandomState(0)  # reproducible
    X = rng.normal(0, 1, 1000)

    win = Winsorizer(lower_quantile, upper_quantile)
    win.fit(X)
    transformed = win.transform(X)

    # Compute expected bounds from original X for assertions
    q1 = np.quantile(X, lower_quantile)
    q2 = np.quantile(X, upper_quantile)

    # All values should be within [q1, q2]
    assert np.all(transformed >= q1 - 1e-12)
    assert np.all(transformed <= q2 + 1e-12)

    # (0, 1) should be a no-op
    if (lower_quantile, upper_quantile) == (0, 1):
        assert np.allclose(transformed, X)

    # If both quantiles are equal, array collapses to that quantile
    if lower_quantile == upper_quantile:
        assert np.allclose(transformed, q1)