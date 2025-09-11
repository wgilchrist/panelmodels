from abc import ABC, abstractmethod
import numba as nb
from dataclasses import dataclass

import numpy as np

from .solver import fit_glm_fe

@dataclass
class FixedEffectsParams:
    fe: np.array
    coeffs: np.array

class _GroupIDMapper:

    def __init__(self):
        self._map = {}
        self._inv_map = {}

    def fit_transform(self, group_ids: np.ndarray) -> None:
        n = len(group_ids)
        out = np.zeros(n, dtype=int)

        next_id = 0
        for i, gid in enumerate(group_ids):
            mapped_gid = self._map.get(gid)
            if mapped_gid is not None:
                out[i] = mapped_gid
            else:
                out[i] = next_id
                self._map[gid] = next_id
                self._inv_map[next_id] = gid
                next_id += 1

        return out
    
    def transform(self, group_ids: np.ndarray) -> np.ndarray:
        try:
            return np.array(
                [self._map[gid] for gid in group_ids],
                dtype=int
            )
        except KeyError as e:
            raise KeyError("New group id mapping encountered")
        
    def inv_transform(self, group_ids: np.ndarray) -> np.ndarray:
        try:
            return np.array(
                [self._inv_map[gid] for gid in group_ids],
                dtype=int
            )
        except KeyError as e:
            raise KeyError("New group id mapping encountered")

class _FixedEffectsGLM(ABC):

    def __init__(self, l1: float = 0.0):
        self.l1 = l1
        self._params = None
        self._map = _GroupIDMapper()

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
        if self._params is None:
            raise ValueError("Fit model before accessing parameters")
        return self._params
            
    def fit(self, X, y, group_ids, weights = None, max_iters: int = 200, tol: int = 1e-12):

        group_ids = self._map.fit_transform(group_ids)   

        if weights is None:
            weights = np.ones(group_ids.shape, dtype=float) 

        alpha, beta = fit_glm_fe(
            self._link,
            self._inv_link,
            self._variance,
            X,
            y,
            group_ids,
            weights,
            max_iters,
            tol,
            self.l1
        )
        self._params = FixedEffectsParams(fe=alpha, coeffs=beta)
    
    def predict(self, X, group_ids):
        group_ids = self._map.transform(group_ids)
        fe, coeffs = self.params.fe, self.params.coeffs
        eta = fe[group_ids] + X @ coeffs
        return self._inv_link(eta)


class FixedEffectsLinear(_FixedEffectsGLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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