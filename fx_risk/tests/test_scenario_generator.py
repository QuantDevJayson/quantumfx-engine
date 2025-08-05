import numpy as np
from fx_risk.scenario_modelling.scenario_generator import HistoricalSimulationScenario


def test_historical_simulation_scenario():
    historical_returns = np.random.normal(0, 0.01, 1000)
    scenario_gen = HistoricalSimulationScenario(historical_returns)
    scenarios = scenario_gen.generate(1.0, 100)
    assert scenarios.shape == (100,)
    assert np.all(scenarios > 0)
