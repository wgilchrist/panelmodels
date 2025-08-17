import numpy as np
from numba.types import FunctionType

def fit_glm_fe(
        link: FunctionType,
        inv_link: FunctionType,
        variance: FunctionType,
        X: np.array,
        y: np.array,
        max_iters: int,
        tol: float
):  
    n, d = X.shape
    beta = np.zeros(d, dtype=float)
    
    for iter_ in range(max_iters):    
        eta = X @ beta
        mu = inv_link(eta)
        v = variance(mu)
        W = v

        z = eta + (y - mu) / W
        XTWX = (X.T * W) @ X
        XTWz = (X.T * W) @ z

        beta_new = np.linalg.solve(XTWX, XTWz)

        if np.max(np.abs(beta_new - beta)) < tol:
            print(f"Stopped early: {iter_}/{max_iters}")
            break

        beta = beta_new

    return beta