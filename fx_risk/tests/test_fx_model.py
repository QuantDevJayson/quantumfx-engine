import numpy as np
from fx_risk.models.fx_model import SimpleGBMModel


def test_gbm_simulation():
    model = SimpleGBMModel({"mu": 0.01, "sigma": 0.05, "dt": 1 / 252})
    market_data = np.ones(100)
    simulated = model.simulate(market_data)
    assert simulated.shape == market_data.shape
    assert np.all(simulated > 0)


def test_gbm_calibration():
    model = SimpleGBMModel({"mu": 0.0, "sigma": 0.1, "dt": 1 / 252})
    historical = np.exp(np.cumsum(np.random.normal(0, 0.01, 252)))
    params = model.calibrate(historical)
    assert "mu" in params and "sigma" in params
