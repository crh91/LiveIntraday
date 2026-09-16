import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit

# --- Import your existing Black-Scholes engine ---
# Ensure your quant_utils.py has SECONDS_PER_DAY = 23100.0
from quant_utils import bs_greeks, bs_price_fast 

def simulate_gbm_paths(S0, vol, drift, sim_days, minutes_per_day, num_paths):
    """
    Generates minute-by-minute Geometric Brownian Motion paths 
    using Natural (Unforced) Variance to model real-world statistical noise.
    """
    # dt remains scaled to a full trading day for accurate intraday volatility math
    dt = 1.0 / (365.0 * minutes_per_day)
    
    # Total steps is based only on the remaining simulation days
    total_steps = int(sim_days * minutes_per_day)
    
    # Standard natural variance (no empirical forcing)
    Z = np.random.standard_normal((total_steps, num_paths))
    
    step_multipliers = np.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z)
    
    prices = np.zeros((total_steps + 1, num_paths))
    prices[0] = S0
    prices[1:] = S0 * np.cumprod(step_multipliers, axis=0)
    
    return prices, dt

@njit
def evaluate_theta_retention_engine(paths, K, initial_dte, eod_dte, intraday_iv, hedge_ratio, eod_iv):
    """
    Numba-optimized Monte Carlo evaluator.
    Linearly drops DTE intraday and uses exact portfolio delta for hedging.
    """
    total_steps = paths.shape[0]
    num_paths = paths.shape[1]
    
    net_pnls = np.zeros(num_paths)
    theta_retentions = np.zeros(num_paths)
    hedge_counts = np.zeros(num_paths, dtype=np.int32)
    initial_thetas = np.zeros(num_paths)
    
    # Pre-calculate how much DTE drops per minute step
    sim_days = initial_dte - eod_dte
    dte_drop_per_step = sim_days / (total_steps - 1) if total_steps > 1 else 0
    
    for p in range(num_paths):
        fut_carried = 0.0
        cum_cost = 0.0
        
        # --- 1. Initial Entry (Using initial_dte and intraday_iv) ---
        S_init = paths[0, p]
        T_init = initial_dte / 365.0
        
        c_price_init = bs_price_fast(S_init, K, T_init, intraday_iv, True)
        p_price_init = bs_price_fast(S_init, K, T_init, intraday_iv, False)
        anchor_straddle = c_price_init + p_price_init
        
        _, c_g_init, c_t_init = bs_greeks(S_init, K, T_init, intraday_iv, True)
        _, p_g_init, p_t_init = bs_greeks(S_init, K, T_init, intraday_iv, False)
        
        # Initial Daily Theta of the Short Straddle (Baseline for TR)
        t0 = -1 * (c_t_init + p_t_init)
        initial_thetas[p] = t0
        
        # --- 2. Intraday Minute-by-Minute Hedging ---
        for t in range(total_steps - 1):
            S_current = paths[t, p]
            
            # Linearly drop DTE based on exact step progress
            current_dte = initial_dte - (t * dte_drop_per_step)
            T_current = max(current_dte / 365.0, 1e-6)
            
            c_d, c_g, c_t = bs_greeks(S_current, K, T_current, intraday_iv, True)
            p_d, p_g, p_t = bs_greeks(S_current, K, T_current, intraday_iv, False)
            
            # Aligning exact signs: df['Gamma'] = -1 * (call_g + put_g)
            short_gamma = -1 * (c_g + p_g)
            short_theta = -1 * (c_t + p_t)
            
            # Track current net delta of the portfolio
            posi_delta = -1 * (c_d + p_d) + fut_carried
            
            # Trigger Logic based on unhedged delta (pts_out)
            if short_gamma != 0:
                pts_out = -1 * posi_delta / short_gamma
                inner_val = abs(short_theta / (2 * short_gamma))
                hedge_pt = np.sqrt(inner_val)
            else:
                pts_out = 0.0
                hedge_pt = 0.0
                
            # Hedge Execution: Neutralize exact risk based on hedge_ratio
            if abs(pts_out) > hedge_pt and short_gamma != 0:
                hq = -1.0 * hedge_ratio * posi_delta 
                fut_carried += hq
                cum_cost += hq * S_current
                hedge_counts[p] += 1
                
        # --- 3. End of Day Settlement (Using eod_dte and eod_iv) ---
        S_final = paths[-1, p]
        T_final = max(eod_dte / 365.0, 1e-6)
        
        # Price your position using the user-provided EOD IV
        c_price_final = bs_price_fast(S_final, K, T_final, eod_iv, True)
        p_price_final = bs_price_fast(S_final, K, T_final, eod_iv, False)
        straddle_final = c_price_final + p_price_final
        
        # Pure Cash-Flow MTM Tracking
        straddle_mtm = anchor_straddle - straddle_final
        hedge_mtm = (fut_carried * S_final) - cum_cost
        posi_mtm = straddle_mtm + hedge_mtm
        
        # Store final results
        net_pnls[p] = posi_mtm
        if t0 != 0:
            theta_retentions[p] = posi_mtm / t0
        else:
            theta_retentions[p] = 0.0
            
    return net_pnls, theta_retentions, hedge_counts, initial_thetas

if __name__ == "__main__":
    # --- Input Parameters ---
    S0 = 74250
    K = 74200
    
    # 1. Separated Volatilities
    realized_vol = 0.146          # ACTUAL movement of the stock (e.g., 12% RV) for path generation
    intraday_iv = 0.17           # PRICING of options intraday (17% IV) for Greeks/hedging triggers
    eod_iv = 0.184                # PRICING of options at 3:40 PM (15% IV) for final settlement
    
    # 2. Time Parameters
    initial_dte = 1.68
    eod_dte = 1.0
    sim_days = initial_dte - eod_dte  # The duration of today's simulation (0.68 days)
    
    minutes_per_day = 385        # Standard trading minutes (e.g., 9:15 AM to 3:40 PM)
    paths_to_simulate = 2000     # Increased to 2,000 for better statistical stability with unforced variance
    hedge_ratio = 1.0            # 1.0 = Hedge back to 0 Delta. 0.5 = Partial hedge

    print(f"Generating {paths_to_simulate} Paths with {realized_vol*100:.1f}% Realized Volatility...")
    
    # Feed REALIZED vol into the path generator
    simulated_paths, dt = simulate_gbm_paths(
        S0, realized_vol, 0.0, sim_days, minutes_per_day, paths_to_simulate
    )

    print(f"Running Engine (Intraday IV: {intraday_iv*100:.0f}%, EOD IV: {eod_iv*100:.0f}%)...")
    
    # Feed INTRADAY IV and EOD IV into the options engine
    net_pnls, theta_retentions, total_hedges, t0_vals = evaluate_theta_retention_engine(
        simulated_paths, K, initial_dte, eod_dte, intraday_iv, hedge_ratio, eod_iv
    )

    # --- Aggregation Metrics ---
    avg_pnl = np.mean(net_pnls)
    avg_tr = np.mean(theta_retentions) * 100
    avg_hedges = np.mean(total_hedges)
    avg_t0 = np.mean(t0_vals)
    
    med_pnl = np.median(net_pnls)
    med_tr = np.median(theta_retentions) * 100 
    med_hedges = np.median(total_hedges)
    med_t0 = np.median(t0_vals)
    
    win_rate = np.mean(net_pnls > 0) * 100
    win_rateMed = np.median(net_pnls > 0) * 100

    print("\n" + "="*45)
    print(f"{f'MONTE CARLO RESULTS ({paths_to_simulate} PATHS)':^45}")
    print("="*45)
    print(f"{'Metric':<20} | {'Mean':<10} | {'Median':<10}")
    print("-" * 45)
    print(f"{'Initial Theta (t0)':<20} | +{avg_t0:<9.2f} | +{med_t0:<9.2f}")
    print(f"{'Net PnL (Points)':<20} |  {avg_pnl:<9.2f} |  {med_pnl:<9.2f}")
    print(f"{'Theta Retention (TR)':<20} |  {avg_tr:<8.1f}% |  {med_tr:<8.1f}%")
    print(f"{'Hedges Executed':<20} |  {avg_hedges:<9.1f} |  {med_hedges:<9.1f}")
    print(f"{'Win Rate':<20} |  {win_rate:.1f}% |  {win_rateMed:.1f}%")
    print("="*45)
    # --- Visualization (First 10 Paths) ---
    paths_to_plot = min(10, paths_to_simulate)
    
    plt.figure(figsize=(12, 6))
    plt.plot(simulated_paths[:, :paths_to_plot], linewidth=1.2, alpha=0.8)
    plt.title(f"Simulation Engine: First {paths_to_plot} Paths (1-Min Intervals)")
    plt.axhline(S0, color='black', linestyle='--', linewidth=2, label=f'Initial Spot ({S0})')
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Spot Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()