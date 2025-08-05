"""
Merton Jump-Diffusion FX Model implementation.
"""
import numpy as np
from typing import Dict, Any

class JumpDiffusionModel:
    """
    Merton jump-diffusion model for FX rates.
    """
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def simulate(self, S0: float, T: float, N: int) -> np.ndarray:
        dt = T / N
        mu = self.params.get('mu', 0.0)
        sigma = self.params.get('sigma', 0.1)
        lam = self.params.get('lam', 0.1)  # jump intensity
        m = self.params.get('m', 0.0)      # mean jump size
        v = self.params.get('v', 0.01)     # jump size volatility
        S = np.zeros(N + 1)
        S[0] = S0
        for t in range(1, N + 1):
            J = np.random.poisson(lam * dt)
            jump = np.sum(np.random.normal(m, v, J)) if J > 0 else 0.0
            dW = np.random.normal(0, np.sqrt(dt))
            S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + jump)
        return S
