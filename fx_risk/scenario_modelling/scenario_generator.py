"""
Scenario generation for FX risk models.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """
    Base class for scenario generation.
    """

    def generate(self, base_fx: float, n_scenarios: int) -> np.ndarray:
        """
        Generate FX scenarios. To be implemented by subclasses.
        Args:
            base_fx (float): Base FX rate.
            n_scenarios (int): Number of scenarios to generate.
        Returns:
            np.ndarray: Generated FX scenarios.
        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("generate() must be implemented by subclasses.")


class HistoricalSimulationScenario(ScenarioGenerator):
    """
    Generate scenarios using historical simulation.
    """

    def __init__(self, historical_returns: np.ndarray) -> None:
        self.historical_returns = historical_returns

    def generate(self, base_fx: float, n_scenarios: int) -> np.ndarray:
        """
        Generate scenarios by resampling historical returns.
        Args:
            base_fx (float): Base FX rate.
            n_scenarios (int): Number of scenarios to generate.
        Returns:
            np.ndarray: Simulated FX scenarios.
        """
        try:
            idx = np.random.choice(len(self.historical_returns), n_scenarios)
            scenarios = base_fx * np.exp(self.historical_returns[idx])
            if np.any(scenarios <= 0):
                logger.warning("Generated scenarios contain non-positive values.")
            return scenarios
        except Exception as e:
            logger.error(f"Scenario generation failed: {e}")
            raise
