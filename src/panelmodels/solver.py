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
def _init_alpha(y: np.ndarray, group_ids: np.ndarray, link: FunctionType) -> np.ndarray:
    n = len(y)
    g = group_ids.max() + 1

    sum_y = np.zeros(g)
    count = np.zeros(g)

    for i in range(n):
        gid = group_ids[i]
        sum_y[gid] += y[i]
        count[gid] += 1

    av_y = sum_y / count
    return link(av_y)

@nb.njit
def _profile_alpha(
    alpha: np.array,
    beta: np.array,
    X: np.array,
    y: np.array,
    group_ids: np.array,
    inv_link: FunctionType,
    variance: FunctionType,
):  
    n, d = X.shape
    g = group_ids.max() + 1

    eta = _broadcast_alpha(alpha, group_ids) + X @ beta
    mu = inv_link(eta)
    var = variance(mu)

    score = np.zeros(g)
    info = np.zeros(g)

    for i in range(n):
        gid = group_ids[i]
        score[gid] += (y[i] - mu[i]) / n
        info[gid] += var[i] / n
    
    return alpha + score / info

@nb.njit
def _profile_alpha(
    alpha: np.array,
    beta: np.array,
    X: np.array,
    y: np.array,
    group_ids: np.array,
    inv_link: FunctionType,
    variance: FunctionType,
):  
    n, d = X.shape
    g = group_ids.max() + 1

    eta = _broadcast_alpha(alpha, group_ids) + X @ beta
    mu = inv_link(eta)
    W = var = variance(mu)

    z = eta + (y - mu) / W
    target = z - X @ beta

    num = np.bincount(group_ids, weights=W * target, minlength=g)
    den = np.bincount(group_ids, weights=W, minlength=g)

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
def _objective(eta, z, w):
    r = eta - z
    return 0.5 * ((w * r * r)).sum()

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
    t_init: float
):
    offset = _broadcast_alpha(alpha, group_ids)
    eta = offset + X @ beta
    mu = inv_link(eta)
    v = variance(mu)
    W = v
    z = eta + (y - mu) / W

    f = _objective(eta, z, W)
    grad = X.T @ (W * (eta - z))

    t = t_init
    if t <= 0.0:
        L = _estimate_L(X, W, 8)
        t = 1.0 / (L + 1e-12)

    # Backtracking line search (monotone)
    F_curr = f + l1 * np.abs(beta).sum()
    for _ in range(30):
        candidate = _soft_threshold(beta - t * grad, l1)
        s = candidate - beta

        eta_c = offset + X @ candidate
        f_c = _objective(eta_c, z, W)

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
    X: np.array,
    y: np.array,
    group_ids: np.array,
    max_iters: int,
    tol: float,
    l1: float
):  
    n, d = X.shape
    X_mu, X_sigma = X.mean(axis=0), X.std(axis=0)
    X_sigma[X_sigma == 0] = 1
    X_std = (X - X_mu) / X_sigma

    t = 0.0
    g = group_ids.max() + 1
    beta = np.zeros(d, dtype=float)
    alpha = _init_alpha(y, group_ids, link)
    
    for iter_ in range(max_iters): 

        print(f"Iteration {iter_+1}/{max_iters}", end="\r")

        alpha_old = alpha
        beta_old = beta

        alpha = _profile_alpha(alpha, beta, X_std, y, group_ids, inv_link, variance)
        beta, t = _profile_beta(alpha, beta, X_std, y, group_ids, inv_link, variance, l1, 0.0)

        if np.maximum(
            np.max(np.abs(alpha - alpha_old)),
            np.max(np.abs(beta - beta_old))
        ) < tol:
            print(f"Stopped early: {iter_}/{max_iters}")
            break
    
    print()

    # handle var(X) = 0 gracefull
    alpha_out, beta_out = np.zeros_like(alpha), np.zeros_like(beta)
    beta_out[X_sigma > 0] = beta[X_sigma > 0] / X_sigma
    alpha_out = alpha - beta_out @ X_mu
    return alpha_out, beta_out