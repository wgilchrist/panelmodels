from abc import ABC, abstractmethod
import numpy as np

class BaseDGP(ABC):

    def __init__(self, coeffs, fe, seed: int = 0):
        self.coeffs = coeffs
        self.fe = fe
        self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def _inv_link(self, mu):
        pass

    @abstractmethod
    def _distribution(self, mu):
        pass

    def draw(self, X, group_ids):
        n, d = X.shape
        assert d == len(self.coeffs)
        assert group_ids.max() + 1 <= len(self.fe)
        assert len(group_ids) == n

        eta = self.fe[group_ids] + X @ self.coeffs
        mu = self._inv_link(eta)
        return self._distribution(mu)

class LinearDGP(BaseDGP):
    def __init__(self, *args, scale=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
    def _inv_link(self, eta):
        return eta
    def _distribution(self, mu):
        return self.rng.normal(mu, self.scale)
    
class PoissonDGP(BaseDGP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _inv_link(self, eta):
        return np.clip(np.exp(eta), None, 1e32)
    def _distribution(self, mu):
        return self.rng.poisson(mu)
    
class BernoulliDGP(BaseDGP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _inv_link(self, eta):
        return 1 / (1 + np.exp(-eta))
    def _distribution(self, mu):
        return self.rng.binomial(1, mu) 