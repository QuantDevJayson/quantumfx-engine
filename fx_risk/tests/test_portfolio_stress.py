import numpy as np
from fx_risk.models.portfolio import FXPortfolio

def test_portfolio_stress_test():
    positions = {"EUR": 1000, "JPY": 200000}
    fx_rates = {"EUR": 1.1, "JPY": 0.007}
    shocks = {"EUR": -0.1, "JPY": 0.2}  # EUR -10%, JPY +20%
    portfolio = FXPortfolio(positions)
    stressed_value = portfolio.stress_test(fx_rates, shocks, base_ccy="USD")
    assert isinstance(stressed_value, float)
    assert stressed_value > 0

def test_portfolio_scenario_analysis():
    positions = {"EUR": 1000, "JPY": 200000}
    n = 5
    fx_scenarios = {
        "EUR": np.linspace(1.0, 1.2, n),
        "JPY": np.linspace(0.006, 0.008, n)
    }
    portfolio = FXPortfolio(positions)
    scenario_values = portfolio.scenario_analysis(fx_scenarios, base_ccy="USD")
    assert scenario_values.shape == (n,)
    assert np.all(scenario_values > 0)
