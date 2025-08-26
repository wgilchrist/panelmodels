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
def _soft_threshold(x: np.ndarray, lambda_: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

@nb.njit
def _profile_beta(
    alpha: np.array,
    beta: np.array,
    X: np.array,
    y: np.array,
    group_ids: np.array,
    inv_link: FunctionType,
    variance: FunctionType,
    l1: float
):
    offset = _broadcast_alpha(alpha, group_ids)
    eta = offset + X @ beta
    mu = inv_link(eta)
    v = variance(mu)
    W = v

    z = eta + (y - mu) / W

    g = (X.T * W) @ (z - offset)
    h = (X.T * W) @ X
    
    beta = np.linalg.solve(h, g)  
    return _soft_threshold(beta, l1) 

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
    X_std = (X - X_mu) / X_sigma

    g = group_ids.max() + 1
    beta = np.zeros(d, dtype=float)
    alpha = np.zeros(g, dtype=float)
    
    for iter_ in range(max_iters): 

        alpha_old = alpha
        beta_old = beta

        alpha = _profile_alpha(alpha, beta, X_std, y, group_ids, inv_link, variance)
        beta = _profile_beta(alpha, beta, X_std, y, group_ids, inv_link, variance, l1)

        if np.maximum(
            np.max(np.abs(alpha - alpha_old)),
            np.max(np.abs(beta - beta_old))
        ) < tol:
            print(f"Stopped early: {iter_}/{max_iters}")
            break
        
    return (alpha - beta @ (X_mu / X_sigma)), beta / X_sigma