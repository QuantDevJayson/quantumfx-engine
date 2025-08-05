"""
Stress testing framework for FX risk models.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class StressTest:
    """
    Base class for FX stress tests.
    """

    def apply(self, fx_series: np.ndarray) -> np.ndarray:
        """
        Apply stress scenario to FX series. To be implemented by subclasses.
        Args:
            fx_series (np.ndarray): FX rate time series.
        Returns:
            np.ndarray: Stressed FX series.
        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("apply() must be implemented by subclasses.")


class ShockScenario(StressTest):
    """
    Apply a fixed shock to the FX rate.
    """

    def __init__(self, shock_pct: float) -> None:
        self.shock_pct = shock_pct

    def apply(self, fx_series: np.ndarray) -> np.ndarray:
        """
        Apply a fixed percentage shock to the FX series.
        Args:
            fx_series (np.ndarray): FX rate time series.
        Returns:
            np.ndarray: Shocked FX series.
        """
        try:
            shocked = fx_series * (1 + self.shock_pct)
            if np.any(shocked <= 0):
                logger.warning("Shocked FX series contains non-positive values.")
            return shocked
        except Exception as e:
            logger.error(f"Shock application failed: {e}")
            raise
