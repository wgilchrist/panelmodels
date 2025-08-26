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
    __test__ = False

    model: _FixedEffectsGLM
    dgp: BaseDGP
    n_observations: int
    n_features: int
    atol_alpha: float
    rtol_alpha: float
    atol_beta: float
    rtol_beta: float

CASES = (
    # Test consistency
    TestCase(
        FixedEffectsLinear(),
        LinearDGP(np.array([1.]), np.array([1., 2.]), scale=0.01),
        1_000_000, 1,
        0., 1e-1,
        0., 1e-1
    ),
    TestCase(
        FixedEffectsPoisson(),
        PoissonDGP(np.array([1.]), np.array([1., 2.])),
        1_000_000, 1,
        0., 1e-1,
        0., 1e-1
    ),
    TestCase(
        FixedEffectsBernoulli(),
        BernoulliDGP(np.array([1.]), np.array([1., 2.])),
        1_000_000, 1,
        0., 1e-1,
        0., 1e-1
    ),

    # Test beta for realistic data
    TestCase(
        FixedEffectsLinear(),
        LinearDGP(
            np.arange(1, 11, dtype=float), 
            np.random.uniform(-10.,2., size=1_000),
            scale=0.01
        ),
        600_000, 10,
        np.inf, 0.,
        0., 0.1
    ),
    TestCase(
        FixedEffectsPoisson(),
        PoissonDGP(
            np.arange(1, 11, dtype=float), 
            np.random.uniform(-10.,2., size=1_000),
        ),
        600_000, 10,
        np.inf, 0.,
        0., 0.1
    ),
    TestCase(
        FixedEffectsBernoulli(),
        BernoulliDGP(
            np.arange(1, 11, dtype=float), 
            np.random.uniform(-10.,2., size=1_000),
        ),
        600_000, 10,
        np.inf, 0.,
        0., 0.1
    ),

    # Test L1 Regularisation
    TestCase(
        FixedEffectsLinear(l1 = .001),
        LinearDGP(
            np.arange(0, 2, dtype=float), 
            np.array([0]),
            scale=0.01
        ),
        1_000_000, 2,
        np.inf, 0.,
        0., 0.1
    ),
)

@pytest.fixture(params=CASES)
def case(request):
    return request.param

def test_estimates(case: TestCase):

    X = np.random.normal(loc=0, scale=.1, size=(case.n_observations, case.n_features))
    group_ids = np.random.randint(0, len(case.dgp.fe), size=(case.n_observations))
    y = case.dgp.draw(X, group_ids)    

    case.model.fit(X, y, group_ids)

    np.testing.assert_allclose(
        case.model.params.fe,
        case.dgp.fe,
        atol = case.atol_alpha,
        rtol = case.rtol_alpha
    )

    np.testing.assert_allclose(
        case.model.params.coeffs,
        case.dgp.coeffs,
        atol = case.atol_beta,
        rtol = case.rtol_beta
    )

if __name__ == "__main__":
    test_estimates(CASES[-1])