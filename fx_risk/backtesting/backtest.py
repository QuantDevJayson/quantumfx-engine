"""
Back-testing and calibration utilities for FX risk models.
"""

import numpy as np
from typing import Dict, Any
import logging
from fx_risk.models.fx_model import FXRiskModel

logger = logging.getLogger(__name__)


class Backtester:
    """
    Back-test FX risk models against historical data.
    """

    def __init__(self, model: FXRiskModel) -> None:
        self.model = model

    def run(self, historical_data: np.ndarray) -> Dict[str, Any]:
        """
        Run back-test and return performance metrics.
        Args:
            historical_data (np.ndarray): Historical FX data.
        Returns:
            Dict[str, Any]: Performance metrics (e.g., MSE).
        """
        try:
            simulated = self.model.simulate(historical_data)
            mse = float(np.mean((simulated - historical_data) ** 2))
            return {"mse": mse}
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise


class Calibrator:
    """
    Calibrate FX risk models to historical data.
    """

    def __init__(self, model: FXRiskModel) -> None:
        self.model = model

    def calibrate(self, historical_data: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate the model to historical data.
        Args:
            historical_data (np.ndarray): Historical FX data.
        Returns:
            Dict[str, Any]: Calibrated parameters.
        """
        try:
            return self.model.calibrate(historical_data)
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise
