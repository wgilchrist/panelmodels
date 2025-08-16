import numpy as np
import pytest
from panelmodels.sklearn_api import PoissonFixedEffects

def make_synthetic(rng, m_groups, n_per_group, p=8, beta_sparsity=3, beta_scale=0.3, alpha_scale=0.5):
    n = m_groups * n_per_group
    X = rng.normal(size=(n, p))
    beta_true = rng.normal(scale=beta_scale, size=p)
    if beta_sparsity > 0:
        beta_true[rng.choice(p, size=beta_sparsity, replace=False)] = 0.0
    base_ids = rng.choice(1_000_000_000, size=m_groups, replace=False)
    g = np.repeat(base_ids, n_per_group)
    alpha_true = rng.normal(loc=0.0, scale=alpha_scale, size=m_groups)
    eta = X @ beta_true + alpha_true[np.repeat(np.arange(m_groups), n_per_group)]
    y = rng.poisson(lam=np.exp(eta))
    return X, y, g, beta_true, alpha_true

def demean(v): return v - v.mean()

def test_api_fit_predict():
    rng = np.random.default_rng(1)
    n, p, m = 150, 4, 3
    X = rng.normal(size=(n, p))
    g = rng.integers(10, 10+m, size=n)
    beta_true = np.array([0.2, -0.4, 0.0, 0.1])
    alpha_true = np.array([0.1, -0.2, 0.3])
    eta = X @ beta_true + alpha_true[(g-10)]
    y = rng.poisson(np.exp(eta))
    est = PoissonFixedEffects(lambda_norm=0.8, l2=1e-3).fit(X, y, group_ids=g)
    mu = est.predict(X, group_ids=g)
    assert mu.shape == (n,)
    assert est.beta_.shape == (p,)

PARAMS = [(10,30),(1000,100),(10000,300)]

@pytest.mark.parametrize("m_groups,n_per_group", PARAMS)
def test_poisson_fe_lasso_parameter_recovery(m_groups, n_per_group):
    rng = np.random.default_rng(1234)
    X, y, g, beta_true, alpha_true = make_synthetic(rng, m_groups=m_groups, n_per_group=n_per_group, p=8, beta_sparsity=3)
    est = PoissonFixedEffects(lambda_norm=0.1, l2=1e-6, max_inner=400, verbose=False)
    est.fit(X, y, group_ids=g)
    assert est.beta_.shape == (X.shape[1],)
    assert est.alpha_.shape == (m_groups,)
    assert est.lambda_max_ > 0
    beta_hat = est.beta_
    beta_rmse = float(np.sqrt(np.mean((beta_hat - beta_true) ** 2)))
    beta_corr = float(np.corrcoef(beta_hat, beta_true)[0, 1])
    inv = est._group_encoder_.inverse_map
    first_seen = {}
    k = 0
    for gid in g:
        i = int(gid)
        if i not in first_seen:
            first_seen[i] = k
            k += 1
            if k == m_groups: break
    idx = np.array([first_seen[int(z)] for z in inv], dtype=int)
    a_true_c = demean(alpha_true[idx])
    a_hat_c = demean(est.alpha_)
    alpha_rmse = float(np.sqrt(np.mean((a_hat_c - a_true_c) ** 2)))
    alpha_corr = float(np.corrcoef(a_hat_c, a_true_c)[0, 1])
    if n_per_group >= 100:
        assert beta_corr > 0.6
        assert alpha_corr > 0.6
        assert beta_rmse < 0.15
    elif n_per_group >= 50:
        assert beta_corr > 0.4
        assert alpha_corr > 0.4
    else:
        assert beta_corr > 0.2
        assert alpha_corr > 0.2
    assert np.isfinite(beta_rmse) and np.isfinite(alpha_rmse)

def test_predict_shapes_and_unseen_groups():
    rng = np.random.default_rng(4321)
    X, y, g, *_ = make_synthetic(rng, m_groups=6, n_per_group=40, p=5, beta_sparsity=2)
    est = PoissonFixedEffects(lambda_norm=0.1, l2=1e-6)
    est.fit(X, y, group_ids=g)
    X_new = rng.normal(size=(20, X.shape[1]))
    seen = np.asarray(list(est._group_encoder_.forward_map.keys()), dtype=np.int64)
    unseen = np.asarray([999999, 888888, 777777], dtype=np.int64)
    pool = np.concatenate([seen, unseen]).astype(np.int64)
    g_new = rng.choice(pool, size=20, replace=True).astype(np.int64)
    mu = est.predict(X_new, group_ids=g_new, unknown_group_alpha=0.0)
    eta = est.decision_function(X_new, group_ids=g_new, unknown_group_alpha=0.0)
    assert mu.shape == (20,) and eta.shape == (20,)
    np.testing.assert_allclose(mu, np.exp(eta), rtol=1e-10, atol=0)