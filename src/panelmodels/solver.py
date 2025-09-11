import numpy as np
import numba as nb
from numba.types import FunctionType

@nb.njit
def _broadcast_alpha(alpha, group_ids):
    n = len(group_ids)
    fe = np.zeros(n, dtype=float)
    for i in range(n):
        gid = group_ids[i]
        fe[i] = alpha[gid]

    return fe

@nb.njit
def _init_alpha(y: np.ndarray, group_ids: np.ndarray, link: FunctionType, w: np.ndarray) -> np.ndarray:
    n = len(y)
    g = group_ids.max() + 1

    sum_wy = np.zeros(g)
    sum_w = np.zeros(g)

    for i in range(n):
        gid = group_ids[i]
        wi = w[i]
        sum_wy[gid] += wi * y[i]
        sum_w[gid] += wi

    for gid in range(g):
        if sum_w[gid] == 0.0:
            sum_w[gid] = 1.0

    av_y = sum_wy / sum_w
    return link(av_y)

@nb.njit
def _profile_alpha(
    alpha: np.ndarray,
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    inv_link: FunctionType,
    variance: FunctionType,
    w: np.ndarray
):  
    g = group_ids.max() + 1

    eta = _broadcast_alpha(alpha, group_ids) + X @ beta
    mu = inv_link(eta)
    W = var = variance(mu)
    Weff = W * w

    z = eta + (y - mu) / Weff
    target = z - X @ beta

    num = np.bincount(group_ids, weights=Weff * target, minlength=g)
    den = np.bincount(group_ids, weights=Weff, minlength=g)

    for gid in range(g):
        if den[gid] == 0.0:
            den[gid] == 1.0

    return num / den

@nb.njit
def _soft_threshold(x: np.ndarray, lambda_: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

@nb.njit
def _Hv(X, W, v):
    t = X @ v
    t *= W
    return X.T @ t

@nb.njit
def _estimate_L(X, W, iters=8):
    d = X.shape[1]
    v = np.ones(d) / np.sqrt(d)
    for _ in range(iters):
        w = _Hv(X, W, v)
        nrm = np.sqrt((w*w).sum())
        if nrm == 0.0:
            return 0.0
        v = w / nrm
    
    Hv = _Hv(X, W, v) # Rayleigh quotient
    return (v @ Hv)

@nb.njit
def _objective(eta, z, w_eff):
    r = eta - z
    return 0.5 * ((w_eff * r * r)).sum()

@nb.njit
def _profile_beta(
    alpha: np.array,
    beta: np.array,
    X: np.array,
    y: np.array,
    group_ids: np.array,
    inv_link: FunctionType,
    variance: FunctionType,
    l1: float,
    t_init: float,
    w: np.ndarray
):
    offset = _broadcast_alpha(alpha, group_ids)
    eta = offset + X @ beta
    mu = inv_link(eta)
    W = v = variance(mu)
    Weff = W * w
    z = eta + (y - mu) / W

    f = _objective(eta, z, Weff)
    grad = X.T @ (Weff * (eta - z))

    t = t_init
    if t <= 0.0:
        L = _estimate_L(X, Weff, 8)
        t = 1.0 / (L + 1e-12)

    # Backtracking line search (monotone)
    F_curr = f + l1 * np.abs(beta).sum()
    for _ in range(30):
        candidate = _soft_threshold(beta - t * grad, l1)
        s = candidate - beta

        eta_c = offset + X @ candidate
        f_c = _objective(eta_c, z, Weff)

        # Beck–Teboulle majorization
        q = f + grad @ s + 0.5 * (s @ s) / t
        if f_c <= q:
            beta = candidate
            t *= 1.1  # promote growth
            break
        t *= 0.5

    return beta, t

def fit_glm_fe(
    link: FunctionType,
    inv_link: FunctionType,
    variance: FunctionType,
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    sample_weights: np.ndarray,
    max_iters: int,
    tol: float,
    l1: float
):  
    n, d = X.shape
    w = sample_weights
    w_sum = w.sum()
    X_mu = (w[:, None] * X).sum(axis=0) / w_sum
    X_ctr = X - X_mu
    X_var = (w[:, None] * (X_ctr * X_ctr)).sum(axis=0) / w_sum
    X_sigma = np.sqrt(X_var)

    X_sigma[X_sigma == 0] = 1.0
    X_std = X_ctr / X_sigma

    t = 0.0
    g = group_ids.max() + 1
    beta = np.zeros(d, dtype=float)
    alpha = _init_alpha(y, group_ids, link, w)
    
    for iter_ in range(max_iters): 
        print(f"Iteration {iter_+1}/{max_iters}", end="\r")
        alpha_old = alpha
        beta_old = beta

        alpha = _profile_alpha(alpha, beta, X_std, y, group_ids, inv_link, variance, w)
        beta, t = _profile_beta(alpha, beta, X_std, y, group_ids, inv_link, variance, l1, 0.0, w)

        if np.maximum(
            np.max(np.abs(alpha - alpha_old)),
            np.max(np.abs(beta - beta_old))
        ) < tol:
            print(f"Stopped early: {iter_}/{max_iters}")
            break
    
    print()

    # unstandardise; handling var(X) = 0 gracefull
    alpha_out, beta_out = np.zeros_like(alpha), np.zeros_like(beta)
    beta_out[X_sigma > 0] = beta[X_sigma > 0] / X_sigma
    alpha_out = alpha - beta_out @ X_mu
    return alpha_out, beta_out