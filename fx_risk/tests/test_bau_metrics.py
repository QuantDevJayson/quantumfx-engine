import numpy as np
from fx_risk.bau_metrics.metrics import BAUMetrics


def test_value_at_risk():
    returns = np.random.normal(0, 0.01, 1000)
    var = BAUMetrics.value_at_risk(returns, 0.05)
    assert isinstance(var, float)


def test_expected_shortfall():
    returns = np.random.normal(0, 0.01, 1000)
    es = BAUMetrics.expected_shortfall(returns, 0.05)
    assert isinstance(es, float)
