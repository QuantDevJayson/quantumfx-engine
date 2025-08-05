"""
BAU (Business-As-Usual) risk metrics for FX portfolios.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class BAUMetrics:
    """
    Compute standard risk metrics for FX portfolios.
    """

    @staticmethod
    def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Compute Value at Risk (VaR) at given confidence level.
        Args:
            returns (np.ndarray): Array of returns.
            alpha (float): Confidence level (default 0.05).
        Returns:
            float: Value at Risk.
        """
        try:
            var = float(np.percentile(returns, 100 * alpha))
            return var
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise

    @staticmethod
    def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Compute Expected Shortfall (ES) at given confidence level.
        Args:
            returns (np.ndarray): Array of returns.
            alpha (float): Confidence level (default 0.05).
        Returns:
            float: Expected Shortfall.
        """
        try:
            var = BAUMetrics.value_at_risk(returns, alpha)
            es = float(returns[returns <= var].mean())
            return es
        except Exception as e:
            logger.error(f"ES calculation failed: {e}")
            raise
