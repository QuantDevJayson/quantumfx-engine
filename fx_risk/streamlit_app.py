
"""
QuantumFX: Modular, High-Performance FX Risk Analytics & Scenario Engine.

Author: @QuantDevJayson
"""
# --- Ensure project root is in sys.path for local imports ---
import sys
import os
import pathlib
import streamlit as st
import numpy as np


# Add project root to sys.path if not already present
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# --- Modern Sidebar Layout ---
def sidebar_inputs():
    with st.sidebar:

        st.markdown("""
        <style>
        :root {
            --qfx-primary: #1a237e;
            --qfx-primary-light: #3949ab;
            --qfx-primary-xlight: #e8eaf6;
            --qfx-accent: #00bcd4;
        }
        .sidebar-title {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--qfx-primary);
            margin-bottom: 0.5em;
            letter-spacing: 1px;
            
            border-radius: 8px;
            padding: 0.3em 0;
            box-shadow: 0 2px 8px 0 rgba(26,35,126,0.04);
        }
        .sidebar-section-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 1.2em;
            margin-bottom: 0.5em;
            color: var(--qfx-primary-light);
            letter-spacing: 0.5px;
        }
        .sidebar-divider {
            border-top: 1px solid var(--qfx-primary-xlight);
            margin: 1em 0 1em 0;
        }
        .sidebar-content {
            background: var(--qfx-primary-xlight) !important;
            border-radius: 10px;
            padding-bottom: 1em;
        }
        textarea, .stTextInput > div > input {
            background: #f5f7fa !important;
            border: 1px solid var(--qfx-primary-light) !important;
            border-radius: 6px !important;
        }
        .stButton > button {
            background: var(--qfx-primary-light) !important;
            color: #fff !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600;
        }
        .stButton > button:hover {
            background: var(--qfx-primary) !important;
            color: #fff !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">QuantumFX</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Model Parameters</div>', unsafe_allow_html=True)
        model = st.selectbox(
            "Model",
            [
                "Heston",
                "Black-Scholes",
                "Jump-Diffusion",
                "GARCH(1,1)",
                "SABR",
                "Regime-Switching",
                "Stochastic Local Volatility (SLV)",
                "Copula",
                "Neural SDE",
                "Rough Volatility"
            ],
            key="sidebar_model_select",
            help="Choose the FX risk model to simulate."
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Portfolio</div>', unsafe_allow_html=True)
        positions = st.text_area("Positions (JSON)", '{"EUR": 1000, "JPY": 200000}', key="sidebar_positions", help="Enter your FX positions as a JSON dictionary.")
        fx_rates = st.text_area("FX Rates (JSON)", '{"EUR": 1.1, "JPY": 0.007}', key="sidebar_fx_rates", help="Current FX rates for each currency.")
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Stress Test</div>', unsafe_allow_html=True)
        shocks = st.text_area("FX Shocks (JSON, e.g. {\"EUR\": -0.1, \"JPY\": 0.2})", '{"EUR": -0.1, "JPY": 0.2}', key="sidebar_shocks", help="Apply shocks to FX rates for stress testing.")
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Scenario Analysis</div>', unsafe_allow_html=True)
        scenarios = st.text_area("FX Scenarios (JSON, e.g. {\"EUR\": [1.1,1.2], \"JPY\": [0.007,0.008]})", '{"EUR": [1.1, 1.2], "JPY": [0.007, 0.008]}', key="sidebar_scenarios", help="Provide scenario FX rates for each currency.")
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        with st.expander("Advanced Settings", expanded=False):
            st.caption("Configure advanced simulation and reporting options below.")
            var_window = st.number_input("Rolling VaR Window (days)", value=20, min_value=5, key="sidebar_var_window", help="Window size for rolling Value at Risk calculation.")
            st.caption("More advanced settings coming soon...")

    return {
        "model": model,
        "positions": positions,
        "fx_rates": fx_rates,
        "shocks": shocks,
        "scenarios": scenarios,
        "var_window": var_window,
    }

sidebar_state = sidebar_inputs()

from fx_risk.models.heston_model import HestonModel
from fx_risk.models.portfolio import FXPortfolio
from fx_risk.reporting.report import FXReport
from fx_risk.models.jump_diffusion import JumpDiffusionModel
from fx_risk.models.garch import GARCHModel


# --- Dashboard Layout ---
import plotly.graph_objs as go
from matplotlib import pyplot as plt

st.markdown("""
<style>
.main .block-container {padding-top: 2rem;}
.sidebar .sidebar-content {background-color: #f0f2f6;}
h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #1a237e !important;
    font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    letter-spacing: 0.5px;
}
.stMarkdown > div > span[style*="font-size:1.2em"] {
    color: #3949ab !important;
}
</style>
""", unsafe_allow_html=True)

st.title("QuantumFX: Modular, High-Performance FX Risk Analytics & Scenario Engine")
st.markdown(
    "<span style='font-size:1.2em'><b>The next-generation open-source platform for quantitative FX risk modeling, scenario analysis, and regulatory stress testing.</b></span>",
    unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "FX Model Simulation", "Portfolio Analytics", "Stress Testing", "Scenario Analysis"
])


# --- Advanced Model Options ---
ADVANCED_MODELS = [
    "Heston", "Black-Scholes", "Jump-Diffusion", "GARCH(1,1)",
    "SABR", "Regime-Switching", "Stochastic Local Volatility (SLV)", "Copula", "Neural SDE", "Rough Volatility"
]

with tab1:
    st.header("FX Model Simulation")
    # Use sidebar_state for model selection
    model_type = sidebar_state["model"]
    # Model parameter input (keep in tab for clarity, but only for selected model)
    if model_type == "Heston":
        kappa = st.number_input("kappa", value=2.0)
        theta = st.number_input("theta", value=0.01)
        sigma = st.number_input("sigma", value=0.1)
        rho = st.number_input("rho", value=-0.7)
        r = st.number_input("r", value=0.0)
        S0 = st.number_input("Initial FX Rate (S0)", value=1.0)
        v0 = st.number_input("Initial Variance (v0)", value=0.01)
        T = st.number_input("Time Horizon (years)", value=1.0)
        N = st.number_input("Steps", value=252)
        params = dict(kappa=kappa, theta=theta, sigma=sigma, rho=rho, r=r)
        model = HestonModel(params)
    elif model_type == "Black-Scholes":
        sigma = st.number_input("sigma", value=0.1)
        mu = st.number_input("mu", value=0.0)
        S0 = st.number_input("Initial FX Rate (S0)", value=1.0)
        T = st.number_input("Time Horizon (years)", value=1.0)
        N = st.number_input("Steps", value=252)
        class BlackScholesFX:
            def __init__(self, mu, sigma):
                self.mu = mu
                self.sigma = sigma
            def simulate(self, S0, T, N):
                dt = T / N
                returns = np.random.normal(self.mu * dt, self.sigma * np.sqrt(dt), int(N))
                path = S0 * np.exp(np.cumsum(returns))
                return np.insert(path, 0, S0)
        model = BlackScholesFX(mu, sigma)
    elif model_type == "Jump-Diffusion":
        mu = st.number_input("mu", value=0.0)
        sigma = st.number_input("sigma", value=0.1)
        lam = st.number_input("Jump Intensity (lambda)", value=0.1)
        m = st.number_input("Mean Jump Size (m)", value=0.0)
        v = st.number_input("Jump Volatility (v)", value=0.01)
        S0 = st.number_input("Initial FX Rate (S0)", value=1.0)
        T = st.number_input("Time Horizon (years)", value=1.0)
        N = st.number_input("Steps", value=252)
        params = dict(mu=mu, sigma=sigma, lam=lam, m=m, v=v)
        model = JumpDiffusionModel(params)
    elif model_type == "GARCH(1,1)":
        mu = st.number_input("mu", value=0.0)
        omega = st.number_input("omega", value=0.00001)
        alpha = st.number_input("alpha", value=0.05)
        beta = st.number_input("beta", value=0.9)
        sigma2 = st.number_input("Initial Variance (sigma^2)", value=0.01)
        S0 = st.number_input("Initial FX Rate (S0)", value=1.0)
        T = st.number_input("Time Horizon (years)", value=1.0)
        N = st.number_input("Steps", value=252)
        params = dict(mu=mu, omega=omega, alpha=alpha, beta=beta, sigma2=sigma2)
        model = GARCHModel(params)
    elif model_type == "SABR":
        st.info("SABR model coming soon. Please select another model.")
        model = None
    elif model_type == "Regime-Switching":
        st.info("Regime-Switching model coming soon. Please select another model.")
        model = None
    elif model_type == "Stochastic Local Volatility (SLV)":
        st.info("Stochastic Local Volatility model coming soon. Please select another model.")
        model = None
    elif model_type == "Copula":
        st.info("Copula model coming soon. Please select another model.")
        model = None
    elif model_type == "Neural SDE":
        st.info("Neural SDE model coming soon. Please select another model.")
        model = None
    elif model_type == "Rough Volatility":
        st.info("Rough Volatility model coming soon. Please select another model.")
        model = None
    else:
        st.warning("Please select a valid model.")
        model = None

    if model is not None and st.button("Simulate FX Path", key="sim_fx"):
        if model_type == "Heston":
            path = model.simulate(S0, v0, T, int(N))
        else:
            path = model.simulate(S0, T, int(N))
        st.subheader(f"Simulated {model_type} FX Path")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=path, mode="lines", name="FX Path"))
        fig.update_layout(xaxis_title="Time Step", yaxis_title="FX Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Summary Stats:", FXReport.summary_stats(path))

        # --- Risk Metrics & Advanced Plots ---
        window = sidebar_state["var_window"]
        returns = np.diff(np.log(path))
        if len(returns) >= window:
            # Rolling VaR
            rolling_var = [np.percentile(returns[max(0, i-window):i+1], 5) for i in range(window-1, len(returns))]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=rolling_var, mode="lines", name="Rolling VaR"))
            fig2.update_layout(title="Rolling 5% Value at Risk (VaR)", xaxis_title="Time Step", yaxis_title="VaR", template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

            # Rolling CVaR (Expected Shortfall)
            rolling_cvar = [np.mean([r for r in returns[max(0, i-window):i+1] if r <= np.percentile(returns[max(0, i-window):i+1], 5)]) for i in range(window-1, len(returns))]
            fig_cvar = go.Figure()
            fig_cvar.add_trace(go.Scatter(y=rolling_cvar, mode="lines", name="Rolling CVaR (ES)"))
            fig_cvar.update_layout(title="Rolling 5% Conditional VaR (ES)", xaxis_title="Time Step", yaxis_title="CVaR", template="plotly_white")
            st.plotly_chart(fig_cvar, use_container_width=True)

            # Rolling Exceedances (number of returns below VaR)
            exceedances = [np.sum(np.array(returns[max(0, i-window):i+1]) < np.percentile(returns[max(0, i-window):i+1], 5)) for i in range(window-1, len(returns))]

            # EWMA VaR
            lambda_ewma = 0.94
            ewma_var = []
            ewma_vol = np.std(returns[:window])
            ewma_var_indices = []
            for i in range(window-1, len(returns)):
                window_returns = returns[max(0, i-window):i+1]
                for r in window_returns:
                    ewma_vol = np.sqrt(lambda_ewma * ewma_vol**2 + (1-lambda_ewma) * r**2)
                var_val = np.percentile(window_returns, 5) * ewma_vol / np.std(window_returns) if np.std(window_returns) > 0 else 0
                ewma_var.append(var_val)
                ewma_var_indices.append(i)

            # Find exceedances (returns below EWMA VaR)
            exceedance_x = []
            exceedance_y = []
            for idx, var_val in zip(ewma_var_indices, ewma_var):
                if returns[idx] < var_val:
                    exceedance_x.append(idx)
                    exceedance_y.append(returns[idx])

            fig_ewma = go.Figure()
            fig_ewma.add_trace(go.Scatter(y=ewma_var, mode="lines", name="EWMA VaR"))
            # Add markers for exceedances
            if exceedance_x:
                fig_ewma.add_trace(go.Scatter(x=exceedance_x, y=exceedance_y, mode="markers", marker=dict(color="red", size=8, symbol="x"), name="Exceedances"))
            fig_ewma.update_layout(title="Rolling EWMA VaR (5%) with Exceedances", xaxis_title="Time Step", yaxis_title="EWMA VaR / Return", template="plotly_white")
            st.plotly_chart(fig_ewma, use_container_width=True)

        # Drawdown
        cummax = np.maximum.accumulate(path)
        drawdown = (path - cummax) / cummax
        fig3, ax3 = plt.subplots()
        ax3.plot(drawdown, color="crimson")
        ax3.set_title("Drawdown Curve")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Drawdown")
        st.pyplot(fig3, use_container_width=True)

        # Volatility, Sharpe, Tail Risk
        vol = np.std(returns) * np.sqrt(252)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else float('nan')
        tail_risk = np.percentile(returns, 10)
        st.metric("Annualized Volatility", f"{vol:.4f}")
        st.metric("Sharpe Ratio", f"{sharpe:.4f}")
        st.metric("10th Percentile (Tail Risk)", f"{tail_risk:.4f}")

        # --- Dynamic Interpretation ---
        st.markdown("---")
        st.subheader("How to Interpret These Results")
        interpretation = ""
        verdict = ""
        if model_type == "Heston":
            interpretation = (
                "The Heston model captures both drift and stochastic volatility. "
                "A high volatility or sharp drawdowns may indicate increased risk or market stress. "
                "Rolling VaR shows how risk evolves over time."
            )
            if vol > 0.2:
                verdict = "⚠️ High volatility detected. Consider risk mitigation or hedging."
            elif sharpe < 0.5:
                verdict = "⚠️ Low risk-adjusted return. Portfolio may be under-compensated for risk."
            else:
                verdict = "✅ Volatility and risk-adjusted return are within normal range."
        elif model_type == "Black-Scholes":
            interpretation = (
                "The Black-Scholes model assumes constant volatility and lognormal returns. "
                "Use this as a baseline for risk, but be aware it may underestimate tail risk and jumps."
            )
            if vol > 0.2:
                verdict = "⚠️ High volatility detected. Black-Scholes may not capture extreme events."
            elif sharpe < 0.5:
                verdict = "⚠️ Low risk-adjusted return. Consider alternative models."
            else:
                verdict = "✅ Portfolio risk and return are in a typical range for Black-Scholes."
        elif model_type == "Jump-Diffusion":
            interpretation = (
                "Jump-Diffusion models capture sudden FX rate jumps due to macro events. "
                "Frequent or large jumps in the path may signal event risk."
            )
            if vol > 0.25:
                verdict = "⚠️ Extreme volatility and jump risk detected. Consider stress testing."
            elif tail_risk < -0.05:
                verdict = "⚠️ Significant negative tail risk. Monitor for large losses."
            else:
                verdict = "✅ Jump risk is moderate."
        elif model_type == "GARCH(1,1)":
            interpretation = (
                "GARCH models capture volatility clustering and time-varying risk. "
                "Periods of high volatility may persist."
            )
            if vol > 0.2:
                verdict = "⚠️ Persistent high volatility. Consider dynamic hedging."
            elif sharpe < 0.5:
                verdict = "⚠️ Low risk-adjusted return. Portfolio may be exposed to volatility shocks."
            else:
                verdict = "✅ Volatility and risk-adjusted return are stable."
        else:
            interpretation = "Model not implemented yet."
            verdict = "Please select a supported model."
        st.info(interpretation)
        st.success(verdict)

        # Correlation Matrix
        st.subheader("Correlation Matrix (Upload Multiple FX Paths)")
        uploaded = st.file_uploader("Upload CSV of FX paths (columns = currencies)", type=["csv"], key="fx_corr")
        if uploaded is not None:
            import pandas as pd
            df = pd.read_csv(uploaded)
            corr = df.pct_change().corr()
            st.dataframe(corr)
            st.caption("Correlation matrix of FX returns")

with tab2:
    st.header("Portfolio Analytics")
    st.caption("Use the sidebar to enter your portfolio and FX rates, then click 'Value Portfolio'.")
    import json
    if st.button("Value Portfolio", key="main_value_portfolio"):
        try:
            pos = json.loads(sidebar_state["positions"])
            rates = json.loads(sidebar_state["fx_rates"])
            portfolio = FXPortfolio(pos)
            value = portfolio.value(rates)
            st.success(f"Portfolio Value: {value:.2f} (base currency)")
            st.write("Currency Exposures:", portfolio.exposures(rates))
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.header("Portfolio Stress Testing")
    st.caption("Use the sidebar to enter shocks and run a stress test on your portfolio.")
    import json
    if st.button("Run Stress Test", key="main_run_stress"):
        try:
            pos = json.loads(sidebar_state["positions"])
            rates = json.loads(sidebar_state["fx_rates"])
            shock_dict = json.loads(sidebar_state["shocks"])
            portfolio = FXPortfolio(pos)
            stressed_value = portfolio.stress_test(rates, shock_dict)
            st.warning(f"Stressed Portfolio Value: {stressed_value:.2f} (base currency)")
        except Exception as e:
            st.error(f"Error: {e}")

with tab4:
    st.header("Portfolio Scenario Analysis")
    st.caption("Use the sidebar to enter scenario FX rates and analyze your portfolio across scenarios.")
    import json
    if st.button("Run Scenario Analysis", key="main_run_scenario"):
        try:
            pos = json.loads(sidebar_state["positions"])
            scenario_dict = json.loads(sidebar_state["scenarios"])
            portfolio = FXPortfolio(pos)
            scenario_values = portfolio.scenario_analysis({k: np.array(v) for k, v in scenario_dict.items()})
            st.write("Portfolio Values per Scenario:", scenario_values)
            st.line_chart(scenario_values)
            # Scenario Distribution Histogram
            st.bar_chart(np.histogram(scenario_values, bins=20)[0])
            st.caption("Distribution of Portfolio Values Across Scenarios")
        except Exception as e:
            st.error(f"Error: {e}")
