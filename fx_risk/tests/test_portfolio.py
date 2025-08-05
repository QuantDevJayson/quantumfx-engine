from fx_risk.models.portfolio import FXPortfolio

def test_portfolio_value():
    positions = {"EUR": 1000, "JPY": 200000}
    fx_rates = {"EUR": 1.1, "JPY": 0.007}
    portfolio = FXPortfolio(positions)
    value = portfolio.value(fx_rates, base_ccy="USD")
    assert isinstance(value, float)
    assert value > 0
