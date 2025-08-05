import numpy as np
from fx_risk.models.heston_model import HestonModel

def test_heston_simulation():
    params = {
        'kappa': 2.0,
        'theta': 0.01,
        'sigma': 0.1,
        'rho': -0.7,
        'r': 0.0
    }
    model = HestonModel(params)
    S = model.simulate(S0=1.0, v0=0.01, T=1.0, N=252)
    assert S.shape == (253,)
    assert np.all(S > 0)
