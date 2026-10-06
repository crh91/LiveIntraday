import streamlit as st

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Live Edge Tracker", layout="wide")

import numpy as np
import matplotlib.pyplot as plt
import datetime
from zoneinfo import ZoneInfo
from GBM_sim import simulate_gbm_paths, evaluate_theta_retention_engine

MINUTES_PER_DAY = 385  # 9:15 AM to 3:40 PM, same as local script


# --- PASSWORD PROTECTION BLOCK ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        password = st.text_input("Enter Password", type="password")
        if password:
            if "password" not in st.secrets:
                st.error("Secret 'password' not found in Streamlit Cloud settings!")
                return False
            if password == st.secrets["password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Password incorrect")
        return False
    return True


if not check_password():
    st.stop()
# ---------------------------------

st.title("⚡ Live Intraday Edge Tracker")

# --- 1. Sidebar Inputs (mirrors the local GBM script) ---
st.sidebar.header("Market Inputs")
spot = st.sidebar.number_input("Spot (S0)", value=72900.0, step=1.0, format="%.2f")
strike = st.sidebar.number_input("Strike (K)", value=72900.0, step=50.0, format="%.2f")

st.sidebar.header("Volatilities")
hv = st.sidebar.number_input("Realized Vol (path generation)", value=0.1590, step=0.0001, format="%.4f")
intraday_iv = st.sidebar.number_input("Intraday IV (Greeks / hedge trigger)", value=0.1700, step=0.0001, format="%.4f")
eod_iv = st.sidebar.number_input("EOD IV (final settlement)", value=0.1725, step=0.0001, format="%.4f")

st.sidebar.header("DTE Window")
initial_dte = st.sidebar.number_input("Start DTE", value=2.3360, step=0.0010, format="%.4f")
eod_dte = st.sidebar.number_input("End DTE", value=2.0000, step=0.0010, format="%.4f")

st.sidebar.header("Simulation")
hedge_ratio = st.sidebar.number_input("Hedge Ratio (1.0 = full, 0.5 = partial)", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
paths_to_simulate = st.sidebar.number_input("Paths", value=2000, min_value=100, max_value=20000, step=500)

run = st.sidebar.button("▶ Run Simulation", type="primary")

# --- 2. Time info (IST) and trading window ---
now_ist = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
trading_window = initial_dte - eod_dte  # = sim_days
sim_minutes = int(trading_window * MINUTES_PER_DAY)

c1, c2, c3 = st.columns(3)
c1.metric("Current Time (IST)", now_ist.strftime("%I:%M %p"))
c2.metric("Trading Window (Start DTE − End DTE)", f"{trading_window:.4f} days")
c3.metric("Simulated Minutes", f"{sim_minutes}")

if trading_window <= 0:
    st.error("Start DTE must be greater than End DTE.")
    st.stop()
if sim_minutes < 2:
    st.error("Trading window is too small to simulate (less than 2 minutes).")
    st.stop()

# --- 3. Engine Execution ---
if run:
    with st.spinner("Running Monte Carlo Engine..."):
        simulated_paths, dt = simulate_gbm_paths(
            spot, hv, 0.0, trading_window, MINUTES_PER_DAY, int(paths_to_simulate)
        )
        net_pnls, theta_retentions, total_hedges, t0_vals = evaluate_theta_retention_engine(
            simulated_paths, strike, initial_dte, eod_dte, intraday_iv, hedge_ratio, eod_iv
        )

    avg_pnl, med_pnl = np.mean(net_pnls), np.median(net_pnls)
    avg_tr, med_tr = np.mean(theta_retentions) * 100, np.median(theta_retentions) * 100
    avg_h, med_h = np.mean(total_hedges), np.median(total_hedges)
    avg_t0, med_t0 = np.mean(t0_vals), np.median(t0_vals)
    win_rate = np.mean(net_pnls > 0) * 100

    # Theta retention measured against theta earned over the window only
    window_theta = avg_t0 * trading_window
    window_tr = (avg_pnl / window_theta * 100) if window_theta != 0 else 0.0

    st.success(f"Simulated {int(paths_to_simulate)} paths, {sim_minutes} minutes each, strike {strike:.0f}.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Net PnL", f"{avg_pnl:.2f} Pts")
    m2.metric("Win Rate", f"{win_rate:.1f}%")
    m3.metric("Theta Retention (full-day t0)", f"{avg_tr:.1f}%")
    m4.metric("Theta Retention (window)", f"{window_tr:.1f}%")

    st.subheader(f"Monte Carlo Results ({int(paths_to_simulate)} paths)")
    st.table({
        "Metric": ["Initial Theta (t0)", "Net PnL (Points)", "Theta Retention (TR) %", "Hedges Executed"],
        "Mean": [f"{avg_t0:.2f}", f"{avg_pnl:.2f}", f"{avg_tr:.1f}", f"{avg_h:.1f}"],
        "Median": [f"{med_t0:.2f}", f"{med_pnl:.2f}", f"{med_tr:.1f}", f"{med_h:.1f}"],
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(simulated_paths[:, :10], linewidth=1, alpha=0.8)
    ax.axhline(spot, color="black", linestyle="--")
    ax.set_title("First 10 Paths (1-Min Intervals)")
    ax.set_xlabel("Time (Minutes)")
    ax.set_ylabel("Spot Price")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.info("Set Start DTE, End DTE and the other inputs in the sidebar, then click **Run Simulation**.")
