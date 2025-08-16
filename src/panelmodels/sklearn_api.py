# file: poisson_fe_lasso/sklearn_api.py
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from dataclasses import dataclass

from .solver import fit_poisson_fe

@dataclass(frozen=True)
class _GroupEncoder:
    """Maps arbitrary int group ids to 0..m-1 and back."""
    forward_map: dict
    inverse_map: np.ndarray  # index -> original id

    @classmethod
    def fit(cls, group_ids: np.ndarray) -> "_GroupEncoder":
        unique = np.unique(group_ids.astype(np.int64))
        fwd = {int(g): i for i, g in enumerate(unique)}
        inv = unique
        return cls(fwd, inv)

    def transform(self, group_ids: np.ndarray) -> np.ndarray:
        out = np.empty_like(group_ids, dtype=np.int64)
        for i, g in enumerate(group_ids):
            out[i] = self.forward_map[int(g)]
        return out

    def try_transform_with_default(self, group_ids: np.ndarray, default_index: int = -1) -> np.ndarray:
        out = np.empty_like(group_ids, dtype=np.int64)
        for i, g in enumerate(group_ids):
            out[i] = self.forward_map.get(int(g), default_index)
        return out


class PoissonFixedEffects(BaseEstimator, RegressorMixin):
    """
    Scikit-learn style wrapper for Poisson FE with L1 penalty on beta.

    Parameters
    ----------
    lambda_norm : float, default=1.0
        Normalized L1 strength; 1.0 ≈ lambda_max at initialization.
    l2 : float, default=0.0
        Optional L2 on beta (elastic-net-like).
    beta0 : array-like of shape (n_features,), default=None
        Warm start for beta (in original feature scale).
    max_outer : int, default=50
    max_inner : int, default=100
    tol : float, default=1e-6
    inner_tol : float, default=1e-8
    eps : float, default=1e-12
    verbose : bool, default=False

    Notes
    -----
    - You must pass `group_ids` to `fit` (and to `predict` if predicting on new samples).
    - Unseen groups at predict time receive alpha=0 by default (i.e., baseline with no FE adjustment).
      You can override by passing `unknown_group_alpha` to `predict`.
    """

    def __init__(
        self,
        lambda_norm: float = 1.0,
        l2: float = 0.0,
        beta0=None,
        max_outer: int = 50,
        max_inner: int = 100,
        tol: float = 1e-6,
        inner_tol: float = 1e-8,
        eps: float = 1e-12,
        verbose: bool = False,
    ):
        self.lambda_norm = lambda_norm
        self.l2 = l2
        self.beta0 = beta0
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.inner_tol = inner_tol
        self.eps = eps
        self.verbose = verbose

    # ---- Required sklearn API ----
    def fit(self, X, y, *, group_ids) -> "PoissonFELasso":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        g_raw = np.asarray(group_ids, dtype=np.int64)

        if X.shape[0] != y.shape[0] or g_raw.shape[0] != y.shape[0]:
            raise ValueError("X, y, and group_ids must have the same number of rows/samples.")

        # remember feature meta
        self.n_features_in_ = X.shape[1]
        # feature_names_in_ only if X is a DataFrame with columns
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(list(X.columns), dtype=object)

        # map groups to 0..m-1
        self._group_encoder_ = _GroupEncoder.fit(g_raw)
        g = self._group_encoder_.transform(g_raw)

        # warm start handling
        beta0 = None
        if self.beta0 is not None:
            beta0 = np.asarray(self.beta0, dtype=np.float64)
            if beta0.shape != (self.n_features_in_,):
                raise ValueError(f"beta0.shape {beta0.shape} != (n_features,) {(self.n_features_in_,)}")

        beta, alpha, info = fit_poisson_fe(
            X=X,
            y=y,
            group_ids=g,
            lambda_norm=self.lambda_norm,
            l2=self.l2,
            beta0=beta0,
            max_outer=self.max_outer,
            max_inner=self.max_inner,
            tol=self.tol,
            inner_tol=self.inner_tol,
            eps=self.eps,
            verbose=self.verbose,
        )

        # store fitted params
        self.beta_ = beta              # (p,)
        self.alpha_ = alpha            # (m,)
        self.scales_ = info["scales"]  # column scales used internally
        self.lambda_max_ = info["lambda_max"]
        self.lambda_used_ = info["lambda_used"]
        self.outer_iters_ = info["outer_iters"]
        self.l2_used_ = info["l2"]
        self.classes_ = None  # for API parity; not used
        self.fitted_ = True
        return self

    def decision_function(self, X, *, group_ids, unknown_group_alpha: float = 0.0) -> np.ndarray:
        """Linear predictor η = X @ beta + alpha[group]."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected X with {self.n_features_in_} features; got {X.shape[1]}")
        g_raw = np.asarray(group_ids, dtype=np.int64)

        # map groups; unseen -> -1
        g = self._group_encoder_.try_transform_with_default(g_raw, default_index=-1)
        eta0 = X @ self.beta_

        # add alpha if seen, else fallback
        eta = eta0.copy()
        seen_mask = (g != -1)
        eta[seen_mask] += self.alpha_[g[seen_mask]]
        eta[~seen_mask] += unknown_group_alpha
        return eta

    def predict(self, X, *, group_ids, unknown_group_alpha: float = 0.0) -> np.ndarray:
        """Mean prediction μ = exp(η)."""
        eta = self.decision_function(X, group_ids=group_ids, unknown_group_alpha=unknown_group_alpha)
        return np.exp(eta)

    # ---- Utilities ----
    def _check_is_fitted(self):
        if not hasattr(self, "fitted_") or not self.fitted_:
            raise AttributeError("Estimator is not fitted yet. Call `fit` with appropriate arguments.")

    # get_params / set_params are inherited from BaseEstimator and work with __init__ signature
