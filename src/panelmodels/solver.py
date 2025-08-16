import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    # Fallback: dummy decorators
    def njit(*args, **kwargs):
        def _wrap(fn): return fn
        return _wrap
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    NUMBA_AVAILABLE = False


# ----------------------------
# Utilities (Numba-friendly)
# ----------------------------

@njit(cache=True, fastmath=True)
def _soft_threshold_numba(x, lam):
    if x >  lam: return x - lam
    if x < -lam: return x + lam
    return 0.0

def _soft_threshold_py(x, lam):
    if x > lam: return x - lam
    if x < -lam: return x + lam
    return 0.0

_SOFT = _soft_threshold_numba if NUMBA_AVAILABLE else _soft_threshold_py


def _make_group_index(g, m=None):
    """
    Sorts by group to make groups contiguous and returns:
    order, inv, starts, ends, m
    """
    g = np.asarray(g, np.int64)
    if m is None:
        m = int(g.max()) + 1
    order = np.argsort(g, kind="stable")
    g_sorted = g[order]
    starts = np.empty(m, dtype=np.int64)
    ends   = np.empty(m, dtype=np.int64)

    # compute starts/ends
    # handle empty groups
    pos = 0
    for k in range(m):
        # move pos to first index with group k
        lo = pos
        while pos < g_sorted.size and g_sorted[pos] == k:
            pos += 1
        hi = pos
        starts[k] = lo
        ends[k]   = hi

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size, dtype=np.int64)
    return order, inv, starts, ends, m


@njit(cache=True, fastmath=True)
def _profile_alpha_sorted(X, beta, g_sorted, starts, ends, T, eps):
    """
    X, beta are in original (non-standardized) space here.
    g is sorted (monotonic non-decreasing), groups are contiguous.
    Return alpha (m,), eta0 (n,) but in sorted order.
    """
    n, p = X.shape
    m = T.size
    eta0 = np.zeros(n, dtype=np.float64)
    for j in range(p):
        bj = beta[j]
        if bj != 0.0:
            eta0 += bj * X[:, j]
    # S = sum_{i in group} exp(eta0_i)
    S = np.zeros(m, dtype=np.float64)
    for k in range(m):
        s = starts[k]; e = ends[k]
        ssum = 0.0
        for i in range(s, e):
            ssum += np.exp(eta0[i])
        S[k] = ssum
    alpha = np.zeros(m, dtype=np.float64)
    for k in range(m):
        # log T - log S (guarded)
        t = T[k] if T[k] > eps else eps
        s = S[k] if S[k] > eps else eps
        alpha[k] = np.log(t) - np.log(s)
    return alpha, eta0


@njit(cache=True, fastmath=True)
def _fe_weighted_demean_sorted(X, W, z, g_sorted, starts, ends, eps):
    """
    Weighted FE demeaning when rows are sorted by group:
    X_tilde_ij = X_ij - (sum_{i in g} W_i X_ij)/(sum_{i in g} W_i)
    z_tilde_i  = z_i  - (sum_{i in g} W_i z_i )/(sum_{i in g} W_i)
    """
    n, p = X.shape
    X_tilde = np.empty_like(X)
    z_tilde = np.empty_like(z)
    Wg = np.zeros(starts.size, dtype=np.float64)

    # Precompute group weight sums
    for k in range(starts.size):
        s = starts[k]; e = ends[k]
        wsum = 0.0
        for i in range(s, e):
            wsum += W[i]
        if wsum < eps: wsum = eps
        Wg[k] = wsum

    # For each column, compute group weighted means and subtract
    for j in range(p):
        # weighted group sums for this column
        for k in range(starts.size):
            s = starts[k]; e = ends[k]
            wsumx = 0.0
            for i in range(s, e):
                wsumx += W[i] * X[i, j]
            mean = wsumx / Wg[k]
            for i in range(s, e):
                X_tilde[i, j] = X[i, j] - mean

    # z_tilde
    for k in range(starts.size):
        s = starts[k]; e = ends[k]
        wsumz = 0.0
        for i in range(s, e):
            wsumz += W[i] * z[i]
        mean = wsumz / Wg[k]
        for i in range(s, e):
            z_tilde[i] = z[i] - mean

    return X_tilde, z_tilde, Wg


@njit(cache=True, fastmath=True)
def _calc_column_scales(X_tilde, W, eps=1e-12):
    n, p = X_tilde.shape
    s = np.empty(p, dtype=np.float64)
    for j in range(p):
        acc = 0.0
        for i in range(n):
            v = X_tilde[i, j]
            acc += W[i] * v * v
        s[j] = np.sqrt(acc) + eps
    return s


@njit(cache=True, fastmath=True, parallel=False)
def _rho_at_zero(X_tilde, z_tilde, W, s):
    # rho_j = sum W * (X_tilde_j/s_j) * z_tilde
    n, p = X_tilde.shape
    rho = np.empty(p, dtype=np.float64)
    for j in range(p):
        sj = s[j]
        acc = 0.0
        for i in range(n):
            acc += W[i] * (X_tilde[i, j] / sj) * z_tilde[i]
        rho[j] = acc
    return rho


@njit(cache=True, fastmath=True)
def _update_eta_from_beta(X, beta):
    n, p = X.shape
    eta0 = np.zeros(n, dtype=np.float64)
    for j in range(p):
        bj = beta[j]
        if bj != 0.0:
            for i in range(n):
                eta0[i] += bj * X[i, j]
    return eta0


@njit(cache=True, fastmath=True)
def _irls_cd_step(
    X,           # (n,p) sorted by group
    beta_std,    # (p,) in std space
    s,           # (p,) column scales from calibration
    g_sorted, starts, ends,
    l2,          # ridge on original beta
    lambda_calibrated,
    eps,
    max_inner,
    inner_tol
):
    """
    One IRLS + CD outer step (given current beta_std),
    returns next beta_std and info to evaluate convergence.
    """
    n, p = X.shape

    # Profile alphas for current beta
    beta_unstd = np.empty(p, dtype=np.float64)
    for j in range(p):
        beta_unstd[j] = beta_std[j] / s[j] if s[j] != 0.0 else 0.0

    # alpha, eta0 (sorted order)
    # First compute eta0 = X @ beta_unstd
    eta0 = _update_eta_from_beta(X, beta_unstd)
    # S, alpha
    m = starts.size
    S = np.zeros(m, dtype=np.float64)
    for k in range(m):
        s0 = starts[k]; e0 = ends[k]
        acc = 0.0
        for i in range(s0, e0):
            acc += np.exp(eta0[i])
        S[k] = acc if acc > eps else eps
    # We also need T; not passed in. We'll recompute outside this function instead.
    # (We avoid computing alpha here to keep this function generic.)
    return eta0  # used for convergence check


@njit(cache=True, fastmath=True)
def _compute_alpha_from_T_S(T, S, eps):
    m = T.size
    alpha = np.zeros(m, dtype=np.float64)
    for k in range(m):
        t = T[k] if T[k] > eps else eps
        s = S[k] if S[k] > eps else eps
        alpha[k] = np.log(t) - np.log(s)
    return alpha


@njit(cache=True, fastmath=True)
def _profile_alpha_fast(X, beta_unstd, g_sorted, starts, ends, T, eps):
    """
    Combined profile alpha with given unstandardized beta.
    """
    n, p = X.shape
    eta0 = _update_eta_from_beta(X, beta_unstd)
    # S
    m = T.size
    S = np.zeros(m, dtype=np.float64)
    for k in range(m):
        s0 = starts[k]; e0 = ends[k]
        acc = 0.0
        for i in range(s0, e0):
            acc += np.exp(eta0[i])
        S[k] = acc
    alpha = _compute_alpha_from_T_S(T, S, eps)
    return alpha, eta0


@njit(cache=True, fastmath=True)
def _cd_loop(
    X_tilde, z_tilde, W, s, beta_std, l2, lambda_calibrated, max_inner, inner_tol
):
    """
    Coordinate descent in standardized space with ridge on original beta.
    Returns beta_std (updated).
    """
    n, p = X_tilde.shape
    # Precompute l2 penalties in std space
    l2_std = np.empty(p, dtype=np.float64)
    for j in range(p):
        l2_std[j] = l2 / (s[j] * s[j]) if s[j] != 0.0 else 0.0

    # Build residual r_std = z_tilde - sum_j (X_tilde_j/s_j)*beta_std_j
    r_std = np.copy(z_tilde)
    for j in range(p):
        if beta_std[j] != 0.0:
            sj = s[j]
            invsj = 1.0 / sj
            for i in range(n):
                r_std[i] -= (X_tilde[i, j] * invsj) * beta_std[j]

    # Curvatures a_j = sum W * (X_tilde_j/s_j)^2 + l2_std_j
    a = np.empty(p, dtype=np.float64)
    for j in range(p):
        sj = s[j]
        invsj = 1.0 / sj
        acc = 0.0
        for i in range(n):
            xstd = X_tilde[i, j] * invsj
            acc += W[i] * xstd * xstd
        a[j] = acc + l2_std[j]

    for _ in range(max_inner):
        max_delta = 0.0
        for j in range(p):
            sj = s[j]
            invsj = 1.0 / sj
            # rho_j = sum W * xj_std * (r_std + xj_std * bj_old)
            acc = 0.0
            for i in range(n):
                xstd = X_tilde[i, j] * invsj
                acc += W[i] * xstd * (r_std[i] + xstd * beta_std[j])
            bj_old = beta_std[j]
            bj_new = _SOFT(acc, lambda_calibrated) / a[j]
            if bj_new != bj_old:
                delta = bj_new - bj_old
                # r_std -= xstd * delta
                for i in range(n):
                    r_std[i] -= (X_tilde[i, j] * invsj) * delta
                beta_std[j] = bj_new
                ad = abs(delta)
                if ad > max_delta:
                    max_delta = ad
        if max_delta < inner_tol * (1.0 + np.abs(beta_std).max()):
            break

    return beta_std


# ------------------------------------------
# Public API: fast Poisson FE + L1 (normalized)
# ------------------------------------------

def fit_poisson_fe(
    X, y, group_ids,
    lambda_norm=1.0,
    l2=0.0,
    beta0=None,
    max_outer=50,
    max_inner=100,
    tol=1e-6,
    inner_tol=1e-8,
    eps=1e-12,
    verbose=False
):
    """
    Fast Poisson FE with L1 on beta, using:
      - exact profiling of group fixed effects (alphas),
      - IRLS outer loop,
      - coordinate descent inner loop on standardized features,
      - one-shot calibration of column scales and λ_max at β=0,
      - Numba-accelerated weighted FE demeaning & reductions.

    Parameters
    ----------
    (same as your original)

    Returns
    -------
    beta : (p,)  coefficients in original scale
    alpha: (m,)  profiled group effects
    info : dict  diagnostics including lambda_max, lambda_used, scales, iters
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(group_ids, dtype=np.int64)
    n, p = X.shape
    m = int(g.max()) + 1

    # Sufficient stats for alpha profile
    T = np.bincount(g, weights=y, minlength=m).astype(np.float64)

    # Sort by group once for fast FE ops
    order, inv, starts, ends, m = _make_group_index(g, m)
    Xs = X[order]
    ys = y[order]
    gs = g[order]

    # Work arrays
    beta_std = np.zeros(p, dtype=np.float64) if beta0 is None else np.asarray(beta0, dtype=np.float64).copy()
    s = np.ones(p, dtype=np.float64)
    lambda_calibrated = None

    # -------- Calibration at beta=0 --------
    # alpha at beta=0
    beta_zero = np.zeros(p, dtype=np.float64)
    alpha0, eta0 = _profile_alpha_fast(Xs, beta_zero, gs, starts, ends, T, eps)
    # eta = eta0 + alpha[g]
    eta = eta0.copy()
    for k in range(m):
        s0 = starts[k]; e0 = ends[k]
        a = alpha0[k]
        for i in range(s0, e0):
            eta[i] += a
    mu = np.exp(eta)
    W = mu
    z = eta + (ys - mu) / np.maximum(mu, eps)

    # weighted FE demeaning
    X_tilde, z_tilde, Wg = _fe_weighted_demean_sorted(Xs, W, z, gs, starts, ends, eps)
    s = _calc_column_scales(X_tilde, W, eps=1e-12)

    # standardized correlation at beta=0 → lambda_max
    rho0 = _rho_at_zero(X_tilde, z_tilde, W, s)
    lambda_max = float(np.abs(rho0).max())
    lambda_calibrated = float(lambda_norm * lambda_max)

    if verbose:
        print(f"[calibration] lambda_max={lambda_max:.6g}  lambda_used={lambda_calibrated:.6g}")

    # -------- IRLS outer loop --------
    it = 0
    for it in range(max_outer):
        # Profile alpha at current beta (unstandardized)
        beta_unstd = beta_std / s
        alpha, eta0 = _profile_alpha_fast(Xs, beta_unstd, gs, starts, ends, T, eps)
        # eta = eta0 + alpha[g]
        eta = eta0.copy()
        for k in range(m):
            s0 = starts[k]; e0 = ends[k]
            a = alpha[k]
            for i in range(s0, e0):
                eta[i] += a

        mu = np.exp(eta)
        W = mu
        z = eta + (ys - mu) / np.maximum(mu, eps)

        # FE demeaning under current W, z
        X_tilde, z_tilde, Wg = _fe_weighted_demean_sorted(Xs, W, z, gs, starts, ends, eps)

        # Coordinate descent in std space
        beta_old_std = beta_std.copy()
        beta_std = _cd_loop(
            X_tilde, z_tilde, W, s, beta_std,
            l2=l2, lambda_calibrated=lambda_calibrated,
            max_inner=max_inner, inner_tol=inner_tol
        )

        # Convergence on linear predictor change (unstandardized)
        eta0_old = Xs @ (beta_old_std / s)
        eta0_new = Xs @ (beta_std      / s)
        if np.linalg.norm(eta0_new - eta0_old) < tol * (1.0 + np.linalg.norm(eta0_old)):
            if verbose:
                print(f"[converged] outer iters: {it+1}")
            break

    # Map back to original row order and compute final alpha
    beta = beta_std / s
    # final alpha in original (unsorted) indexing
    eta0_full_sorted = Xs @ beta
    S = np.zeros(m, dtype=np.float64)
    for k in range(m):
        s0 = starts[k]; e0 = ends[k]
        acc = 0.0
        for i in range(s0, e0):
            acc += np.exp(eta0_full_sorted[i])
        S[k] = acc
    alpha_sorted = _compute_alpha_from_T_S(T, S, eps)

    alpha = alpha_sorted  # (m,)
    info = {
        "outer_iters": int(it + 1),
        "lambda_norm": float(lambda_norm),
        "lambda_max": float(lambda_max),
        "lambda_used": float(lambda_calibrated),
        "l2": float(l2),
        "scales": s,
        "numba": bool(NUMBA_AVAILABLE),
    }
    return beta, alpha, info


# ------------------------------------------
# (Optional) Backward-compatible alias
# ------------------------------------------

def fit_poisson_fe_lasso_normalized(*args, **kwargs):
    """
    Backward-compatible name that now uses the fast implementation.
    """
    return fit_poisson_fe_lasso_normalized_fast(*args, **kwargs)
