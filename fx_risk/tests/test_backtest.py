import numpy as np
from fx_risk.models.fx_model import SimpleGBMModel
from fx_risk.backtesting.backtest import Backtester, Calibrator


def test_backtester():
    model = SimpleGBMModel({"mu": 0.01, "sigma": 0.05, "dt": 1 / 252})
    historical = np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    backtester = Backtester(model)
    results = backtester.run(historical)
    assert "mse" in results


def test_calibrator():
    model = SimpleGBMModel({"mu": 0.0, "sigma": 0.1, "dt": 1 / 252})
    historical = np.exp(np.cumsum(np.random.normal(0, 0.01, 252)))
    calibrator = Calibrator(model)
    params = calibrator.calibrate(historical)
    assert "mu" in params and "sigma" in params
