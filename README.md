# QuantumFX: Modular, High-Performance FX Risk Analytics & Scenario Engine

### The next-generation open-source platform for quantitative FX risk modeling, scenario analysis, and regulatory stress testing.
---
Note: This is an extract of a live, ongoing project for advanced FX risk management. The codebase is actively developed and extended with new features, analytics, and integrations.

This project provides a highly modular, high-performance framework for developing, maintaining, and validating advanced FX risk models. Each domain (models, scenario_modelling, stress_testing, bau_metrics, backtesting, api, reporting, streamlit_app) is strictly separated for maintainability and scalability. Scenario analysis is parallelized for large-scale, real-time risk analytics.

## Why BAU Risk Metrics?
**Business-As-Usual (BAU) risk metrics** are essential for monitoring and managing FX risk under normal market conditions. They provide a quantitative foundation for day-to-day risk oversight, including measures such as Value at Risk (VaR) and Expected Shortfall. BAU metrics help institutions:
- Detect and control risk exposures before they escalate
- Satisfy regulatory requirements for ongoing risk management
- Benchmark performance and risk across portfolios
By integrating BAU metrics, QuantumFX ensures robust, real-time risk monitoring as a core part of the risk management workflow.

It includes:

- **Scenario Modelling**
- **Stress Testing**
- **BAU Risk Metrics**
- **Back-testing & Calibration**
- **Multi-currency Portfolio Support**
- **API Integration (FastAPI)**
- **Advanced Reporting & Visualization**

## Features
- Strictly modular, extensible architecture
- Production-ready code structure
- Back-testing and calibration for accuracy and compliance
- Designed for regulatory standards
- Portfolio-level stress testing and scenario analysis (parallelized for high performance)
- Live REST API endpoints for model simulation and portfolio valuation
- YAML-based configuration for flexible workflows

## Structure
- `fx_risk/models/` — Core model definitions (e.g., `fx_model.py`, `heston_model.py`), portfolio logic (`portfolio.py`), and configuration loader (`config_loader.py`).
- `fx_risk/scenario_modelling/` — Scenario generation and management (e.g., `scenario_generator.py`).
- `fx_risk/stress_testing/` — Stress test modules (e.g., `stress_test.py`).
- `fx_risk/bau_metrics/` — Business-as-usual risk metrics (e.g., `metrics.py`).
- `fx_risk/backtesting/` — Back-testing and calibration tools (e.g., `backtest.py`).
- `fx_risk/api/` — REST API server (`server.py`) for integration and automation.
- `fx_risk/reporting/` — Reporting and visualization utilities (e.g., `report.py`).
- `fx_risk/streamlit_app.py` — Interactive Streamlit visualization UI for simulation, portfolio analytics, and scenario analysis.
- `config.yaml` — Example YAML configuration for model parameters.
- `requirements.txt` — All Python dependencies for production deployment.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Explore all modules in `fx_risk/` (see structure above for file-level details).
3. Run tests: `pytest` (all modules are covered by unit tests for production reliability).
4. Launch the API: `uvicorn fx_risk.api.server:app --reload` (for RESTful integration).
5. Launch the Streamlit app for interactive visualization: `streamlit run fx_risk/streamlit_app.py`.
6. Edit `config.yaml` to customize model parameters for your use case.

## Supported FX Risk Models

QuantumFX supports a wide range of advanced and research-grade FX risk models. The following models are available for simulation and analytics in the Streamlit dashboard:

- Heston
- Black-Scholes
- Jump-Diffusion
- GARCH(1,1)
- SABR (coming soon)
- Regime-Switching (coming soon)
- Stochastic Local Volatility (SLV) (coming soon)
- Copula (coming soon)
- Neural SDE (coming soon)
- Rough Volatility (coming soon)

All models are accessible from the sidebar dropdown in the Streamlit app. Models marked "coming soon" are included for roadmap visibility and will be enabled in future releases.

## Screenshots

Below are screenshots of the QuantumFX Streamlit dashboard, demonstrating the model selection, analytics, and visualization features:


<img width="870" height="422" alt="quantumfx-model-scenario-engine" src="https://github.com/user-attachments/assets/8bbc811e-ec1a-4e58-825d-ca313201829b" />

---

<img width="870" height="422" alt="quantumfx-model-scenario-engine" src="https://github.com/user-attachments/assets/daeb181a-75dd-46da-b132-006fcf27e6b8" />

---

<img width="592" height="373" alt="quantumfx-simulated-garch" src="https://github.com/user-attachments/assets/609f0c74-cb67-4269-bb0f-a2ed41d96c53" />

---

<img width="875" height="419" alt="quantumfx-interpret" src="https://github.com/user-attachments/assets/fa3eca1b-c3da-4803-b9c6-956686b88796" />

---

<img width="704" height="450" alt="quantumfx-ewma-var-exceedances" src="https://github.com/user-attachments/assets/fa98af62-ddf5-4016-8142-8c2cd2df9d8a" />

---

## Contributing
This project is under active development by the author and contributors. Thus, new features and improvements are added regularly.

## Visualization
The Streamlit app (`fx_risk/streamlit_app.py`) provides an interactive UI for:
- Simulating FX model paths
- Valuing and analyzing multi-currency portfolios
- Running portfolio stress tests and scenario analysis
Launch with:
```
streamlit run fx_risk/streamlit_app.py
```


**Disclaimer**: Majority of features have been truncated by the original author for brevity and clarity.



**Feel Free to Contact Original Author:**

## Contact the Author

- **GitHub**: [QuantDevJayson](https://github.com/QuantDevJayson)  
- **PyPI**: [jayson.ashioya](https://pypi.org/user/jayson.ashioya)  
- **LinkedIn**: [Jayson Ashioya](https://www.linkedin.com/in/jayson-ashioya-c-082814176/)



