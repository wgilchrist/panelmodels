import numpy as np
from panelmodels.solver import fit_poisson_fe

def test_runs_and_shapes():
    rng = np.random.default_rng(0)
    n, p, m = 200, 5, 7
    X = rng.normal(size=(n, p))
    g = rng.integers(0, m, size=n)
    beta_true = np.array([0.3, -0.1, 0.0, 0.0, 0.2])
    alpha_true = rng.normal(scale=0.2, size=m)
    eta = X @ beta_true + alpha_true[g]
    y = rng.poisson(lam=np.exp(eta))

    beta, alpha, info = fit_poisson_fe(X, y, g, lambda_norm=0.5)
    assert beta.shape == (p,)
    assert alpha.shape == (m,)
    assert info["lambda_max"] > 0