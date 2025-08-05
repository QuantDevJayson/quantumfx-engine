"""
Multi-currency FX portfolio support.
"""
import numpy as np
from typing import Dict, List

class FXPortfolio:
    def stress_test(self, fx_rates: Dict[str, float], shocks: Dict[str, float], base_ccy: str = "USD") -> float:
        """
        Apply stress shocks to FX rates and return stressed portfolio value.
        Args:
            fx_rates (Dict[str, float]): Current FX rates.
            shocks (Dict[str, float]): Shock (as decimal, e.g., 0.1 for +10%) for each currency.
            base_ccy (str): Base currency for valuation.
        Returns:
            float: Stressed portfolio value in base currency.
        """
        stressed_rates = fx_rates.copy()
        for ccy, shock in shocks.items():
            if ccy in stressed_rates:
                stressed_rates[ccy] *= (1 + shock)
        return self.value(stressed_rates, base_ccy)

    def scenario_analysis(self, fx_scenarios: Dict[str, np.ndarray], base_ccy: str = "USD") -> np.ndarray:
        """
        Run scenario analysis: compute portfolio value for each scenario path, parallelized for high performance.
        Args:
            fx_scenarios (Dict[str, np.ndarray]): Dict of currency to array of FX rates (shape: [n_scenarios]).
            base_ccy (str): Base currency for valuation.
        Returns:
            np.ndarray: Portfolio values for each scenario.
        """
        import concurrent.futures
        n_scenarios = None
        for arr in fx_scenarios.values():
            n_scenarios = len(arr)
            break
        if n_scenarios is None:
            return np.array([])

        def scenario_value(i):
            rates = {ccy: arr[i] for ccy, arr in fx_scenarios.items()}
            return self.value(rates, base_ccy)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            values = list(executor.map(scenario_value, range(n_scenarios)))
        return np.array(values)
    """
    Represents a portfolio of FX positions across multiple currencies.
    """
    def __init__(self, positions: Dict[str, float]):
        """
        Args:
            positions (Dict[str, float]): Mapping from currency code to notional amount.
        """
        self.positions = positions

    def value(self, fx_rates: Dict[str, float], base_ccy: str = "USD") -> float:
        """
        Calculate portfolio value in the base currency.
        Args:
            fx_rates (Dict[str, float]): Mapping from currency code to FX rate vs base_ccy.
            base_ccy (str): The base currency for valuation.
        Returns:
            float: Portfolio value in base currency.
        Raises:
            ValueError: If a required FX rate is missing.
        """
        value = 0.0
        for ccy, notional in self.positions.items():
            if ccy == base_ccy:
                value += notional
            elif ccy in fx_rates:
                value += notional * fx_rates[ccy]
            else:
                raise ValueError(f"FX rate for {ccy} vs {base_ccy} is missing.")
        return value

    def exposures(self, fx_rates: Dict[str, float], base_ccy: str = "USD") -> Dict[str, float]:
        """
        Return the value of each currency position in the base currency.
        Args:
            fx_rates (Dict[str, float]): Mapping from currency code to FX rate vs base_ccy.
            base_ccy (str): The base currency for valuation.
        Returns:
            Dict[str, float]: Currency exposures in base currency.
        """
        exposures = {}
        for ccy, notional in self.positions.items():
            if ccy == base_ccy:
                exposures[ccy] = notional
            elif ccy in fx_rates:
                exposures[ccy] = notional * fx_rates[ccy]
            else:
                exposures[ccy] = float('nan')
        return exposures

    def portfolio_var(self, returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Compute portfolio Value at Risk (VaR) at given confidence level.
        Args:
            returns (np.ndarray): Portfolio return series.
            alpha (float): Confidence level (default 0.05).
        Returns:
            float: Portfolio VaR.
        """
        return float(np.percentile(returns, 100 * alpha))

    def portfolio_expected_shortfall(self, returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Compute portfolio Expected Shortfall (ES) at given confidence level.
        Args:
            returns (np.ndarray): Portfolio return series.
            alpha (float): Confidence level (default 0.05).
        Returns:
            float: Portfolio ES.
        """
        var = self.portfolio_var(returns, alpha)
        return float(returns[returns <= var].mean())
