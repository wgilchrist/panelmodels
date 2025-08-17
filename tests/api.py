from dataclasses import dataclass
import numpy as np
import pytest

from panelmodels.api import (
    _FixedEffectsGLM,
    FixedEffectsLinear,
    FixedEffectsPoisson,
    FixedEffectsBernoulli
)
from dgp import(
    BaseDGP,
    LinearDGP,
    PoissonDGP,
    BernoulliDGP
)

@dataclass
class TestCase:
    model: _FixedEffectsGLM
    dgp: BaseDGP
    n_observations: int
    n_features: int
    atol: float
    rtol: float

N, D = 1_000_000, 10
COEFFS = np.arange(D, dtype=float)
FE = np.array([0])
ATOL, RTOL = .05, 0.

CASES = (
    TestCase(
        FixedEffectsLinear(),
        LinearDGP(COEFFS, FE, scale=0.01),
        N, D, ATOL, RTOL
    ),
    TestCase(
        FixedEffectsPoisson(),
        PoissonDGP(COEFFS, FE),
        N, D, ATOL, RTOL
    ),
    TestCase(
        FixedEffectsBernoulli(),
        BernoulliDGP(COEFFS, FE),
        N, D, ATOL, RTOL
    )
)

@pytest.fixture(params=CASES)
def case(request):
    return request.param

def test_estimates(case: TestCase):

    X = np.random.normal(loc=0, scale=.1, size=(case.n_observations, case.n_features))
    group_ids = np.zeros(case.n_observations, dtype=int)
    y = case.dgp.draw(X, group_ids)    

    case.model.fit(X, y, group_ids)

    np.testing.assert_allclose(
        case.model.params.coeffs,
        case.dgp.coeffs,
        atol = case.atol,
        rtol = case.rtol
    )

if __name__ == "__main__":
    test_estimates(CASES[2])