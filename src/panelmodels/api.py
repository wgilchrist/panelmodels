from abc import ABC, abstractmethod
import numba as nb
from dataclasses import dataclass

import numpy as np

from .solver import fit_glm_fe

@dataclass
class FixedEffectsParams:
    fe: np.array
    coeffs: np.array

class _GroupMapper:

    def __init__(self):
        self._mapping = {}
        self._inv_mapping = {}
        self.max_id = 0

    def fit_transform(self, group_ids: np.array):
        self.inp_dtype = group_ids.dtype
        trfm_group_ids = np.empty(group_ids.shape, dtype=int)
        for pos, id_ in enumerate(group_ids):
            if id_ not in self._mapping:
                self._mapping[id_] = self.max_id
                self._inv_mapping[self.max_id] = id_
                self.max_id += 1
            
            trfm_group_ids[pos] = self._mapping[id_]
        
        return trfm_group_ids

    def transform(self, group_ids):
        if not hasattr(self, "inp_dtype"):
            raise ValueError("transform called before fit_transform")
        
        trfm_group_ids = np.empty(len(group_ids), dtype=int)
        for i, id_ in enumerate(group_ids):
            try:
                trfm_group_ids[i] = self._mapping[id_]
            except KeyError:
                raise KeyError(f"Unknown group id encountered: {id_!r}")

        return trfm_group_ids

    def inv_transform(self, trfm_group_ids):
        if not hasattr(self, "inp_dtype"):
            raise ValueError("inv_transform called before fit_transform")
        
        group_ids = np.empty(len(trfm_group_ids), self.inp_dtype)
        for i, id_ in enumerate(trfm_group_ids):
            try:
                group_ids[i] = self._inv_mapping[id_]
            except KeyError:
                raise KeyError(f"Unknown mapped id encountered: {id_!r}")

        return group_ids


class _FixedEffectsGLM(ABC):

    def __init__(self):
        self._group_mapper = _GroupMapper()
        self._params = None

    @abstractmethod
    def _link(mu):
        pass

    @abstractmethod
    def _inv_link(x):
        pass

    @abstractmethod
    def _variance(eta):
        pass

    @property
    def params(self) -> FixedEffectsParams:
        if self._params is not None:
            return self._params
        
        raise ValueError("Fit model before accessing parameters")
    
    def fit(self, X, y, group_ids, max_iters: int = 1000, tol:int = 1e-12):
        
        X = np.hstack([np.ones((len(X), 1)), X])
        
        beta = fit_glm_fe(
            self._link,
            self._inv_link,
            self._variance,
            X,
            y,
            max_iters,
            tol
        )
        fe, coeffs = beta[0], beta[1:]
        self._params = FixedEffectsParams(fe=fe, coeffs=coeffs)
    
    def predict(self, X, group_ids):
        fe, coeffs = self.params.fe, self.params.coeffs
        eta = fe[group_ids] + X @ coeffs
        return self._inv_link(eta)


class FixedEffectsLinear(_FixedEffectsGLM):

    def __init__(self):
        super().__init__()

    @staticmethod
    @nb.njit
    def _link(mu):
        return mu
    
    @staticmethod
    @nb.njit
    def _inv_link(eta):
        return eta
    
    @staticmethod
    @nb.njit
    def _variance(mu):
        return np.ones_like(mu)
    
class FixedEffectsPoisson(_FixedEffectsGLM):

    def __init__(self):
        super().__init__()

    @staticmethod
    @nb.njit
    def _link(mu):
        return np.log(np.maximum(mu, 1e-8))
    
    @staticmethod
    @nb.njit
    def _inv_link(eta):
        return np.exp(np.minimum(eta, 700))
    
    @staticmethod
    @nb.njit
    def _variance(mu):
        return np.maximum(mu, 1e-8)
    
class FixedEffectsBernoulli(_FixedEffectsGLM):

    def __init__(self):
        super().__init__()

    @staticmethod
    @nb.njit
    def _link(mu):
        mu_clipped = np.clip(mu, 1e-8, 1 - 1e-8)
        return np.log(mu_clipped / (1 - mu_clipped))
    
    @staticmethod
    @nb.njit
    def _inv_link(eta):
        eta_clipped = np.clip(eta, -700, 700)
        return 1 / (1 + np.exp(-eta_clipped))
    
    @staticmethod
    @nb.njit
    def _variance(mu):
        mu_clipped = np.clip(mu, 1e-8, 1 - 1e-8)
        return mu_clipped * (1-mu_clipped)
