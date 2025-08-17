from panelmodels.api import (
    FixedEffectsLinear,
    _FixedEffectsGLM
)
from dataclasses import dataclass
import numpy as np
import pytest

@dataclass
class TestCase:
    cls: _FixedEffectsGLM
    n_observations: int
    n_features: int
    coeffs: np.array
    atol: float
    rtol: float


CASES = (
    TestCase(FixedEffectsLinear, 1000, 10, np.arange(11, dtype=float), 1e-3, 0),
)

@pytest.fixture(params=CASES)
def case(request):
    return request.param

def test_estimates(case: TestCase):
    model = case.cls()

    group_ids = np.zeros(case.n_observations, dtype=int)
    X = np.random.normal(size=(case.n_observations, case.n_features))
    X_const = np.hstack([np.ones((case.n_observations, 1)), X])
    e = np.random.normal(scale = .01, size = case.n_observations)

    eta = X_const @ case.coeffs
    y = model._inv_link(eta) + e

    model.fit(X, y, group_ids)

    np.testing.assert_allclose(
        model.params.coeffs,
        case.coeffs,
        atol = case.atol,
        rtol = case.rtol
    )

if __name__ == "__main__":
    test_estimates(CASES[0])