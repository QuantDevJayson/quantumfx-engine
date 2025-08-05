"""
Simple REST API for FX risk model operations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fx_risk.models.heston_model import HestonModel
from fx_risk.models.portfolio import FXPortfolio
import numpy as np

app = FastAPI()

class HestonRequest(BaseModel):
    params: dict
    S0: float
    v0: float
    T: float
    N: int

@app.post("/simulate/heston")
def simulate_heston(req: HestonRequest):
    try:
        model = HestonModel(req.params)
        path = model.simulate(req.S0, req.v0, req.T, req.N)
        return {"path": path.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class PortfolioRequest(BaseModel):
    positions: dict
    fx_rates: dict
    base_ccy: str = "USD"

@app.post("/portfolio/value")
def portfolio_value(req: PortfolioRequest):
    try:
        portfolio = FXPortfolio(req.positions)
        value = portfolio.value(req.fx_rates, req.base_ccy)
        return {"value": value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
