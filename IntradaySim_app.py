import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# --- CLEAN IMPORT ---
# This pulls your functions directly from your GBM_sim.py file!
from GBM_sim import simulate_gbm_paths, evaluate_theta_retention_engine

st.set_page_config(page_title="Live Edge Tracker", layout="wide")

st.title("⚡ Live Intraday Edge Tracker")

# --- 1. Sidebar Inputs ---
st.sidebar.header("Live Inputs")
spot = st.sidebar.number_input("Live Spot Price", value=74250)

# UPDATED: Added format="%.4f" and changed step to 0.0001
hv = st.sidebar.number_input("Realized Volatility (HV)", value=0.1200, step=0.0001, format="%.4f")
intraday_iv = st.sidebar.number_input("Intraday IV", value=0.1700, step=0.0001, format="%.4f")
eod_iv = st.sidebar.number_input("Target EOD IV", value=0.1500, step=0.0001, format="%.4f")

st.sidebar.header("DTE Parameters")
full_days_left = st.sidebar.number_input("Full Days to Expiry (EOD DTE)", value=4.0, step=1.0)
day_pass_adj = st.sidebar.number_input("Day Pass Adjust", value=0.8, step=0.1)

# NEW: The additional day adjustment variable
day_adj_variable = st.sidebar.number_input("Day Adj Variable", value=0.0, step=0.1)

auto_refresh = st.sidebar.checkbox("Enable Live 60s Auto-Refresh", value=True)

# --- 2. Live Time & DTE Calculation ---
# Market Close = 15:40 (3:40 PM), Open = 9:15 AM
now = datetime.datetime.now()
market_close = now.replace(hour=15, minute=40, second=0, microsecond=0)
market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

if now > market_close:
    minutes_left = 0
elif now < market_open:
    minutes_left = 385
else:
    minutes_left = int((market_close - now).total_seconds() / 60)

# The time math you asked about
fractional_day_left = minutes_left / 385.0
sim_days = fractional_day_left  

# NEW DTE FORMULA exactly as you requested
initial_dte = full_days_left + (fractional_day_left * day_pass_adj) - day_adj_variable
eod_dte = full_days_left - day_adj_variable

st.write(f"**Current Time:** {now.strftime('%I:%M %p')} | **Minutes to Close:** {minutes_left}")
st.write(f"**Fractional Day Left:** {fractional_day_left:.4f} (Based on 385 mins total)")
st.write(f"**Effective Current DTE:** {initial_dte:.4f} days | **EOD DTE:** {eod_dte:.4f} days")

# --- 3. Engine Execution ---
if minutes_left > 0:
    strike = round(spot / 50) * 50  
    hedge_ratio = 1.0 
    paths_to_simulate = 1000
    
    with st.spinner("Running Monte Carlo Engine..."):
        # Run the imported functions!
        simulated_paths, dt = simulate_gbm_paths(
            spot, hv, 0.0, sim_days, 385, paths_to_simulate
        )
        
        net_pnls, theta_retentions, total_hedges, t0_vals = evaluate_theta_retention_engine(
            simulated_paths, strike, initial_dte, eod_dte, intraday_iv, hedge_ratio, eod_iv
        )
        
        # Calculate Display Metrics
        avg_pnl = np.mean(net_pnls)
        win_rate = np.mean(net_pnls > 0) * 100
        avg_tr = np.mean(theta_retentions) * 100
        
        st.success(f"Simulation complete for {strike} Strike.")
        
        # Display large metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Net PnL", f"{avg_pnl:.2f} Pts")
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Theta Retention", f"{avg_tr:.1f}%")
        
        # Optional: Add the graph back in Streamlit
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(simulated_paths[:, :10], linewidth=1, alpha=0.8)
        ax.axhline(spot, color='black', linestyle='--')
        ax.set_title("First 10 Paths")
        st.pyplot(fig)
else:
    st.warning("Market is closed. Simulation stopped.")

# --- 4. Auto-Refresh Trigger ---
if auto_refresh and minutes_left > 0:
    time.sleep(60)
    st.rerun()
