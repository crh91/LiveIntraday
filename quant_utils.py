# quant_utils.py
import datetime as dt
from typing import List
import numpy as np
import math
from numba import njit

# --- 1. TIME TO EXPIRY ENGINE ---

def calculate_time_to_expiry(
    current_dt: dt.datetime, 
    expiry_dt: dt.datetime, 
    holidays_list: List[dt.date], 
    day_passing: float = 1.0
) -> float:
    """Calculates annualized time to expiry considering market hours and holidays."""
    OPEN_TIME = dt.time(9, 15, 0)
    CLOSE_TIME = dt.time(15, 40, 0)
    SECONDS_PER_DAY = 23100.0  
    
    current_time = current_dt.time()
    current_date = current_dt.date()
    expiry_date = expiry_dt.date()

    if current_time < OPEN_TIME:
        seconds_left = SECONDS_PER_DAY 
    elif current_time > CLOSE_TIME:
        seconds_left = 0.0 
    else:
        seconds_left = (dt.datetime.combine(current_date, CLOSE_TIME) - current_dt).total_seconds()

    seconds_left *= day_passing
    
    day_count = 0
    next_date = current_date + dt.timedelta(days=1)
    
    while next_date <= expiry_date:
        if next_date.weekday() < 5 and next_date not in holidays_list: 
            day_count += 1
        next_date += dt.timedelta(days=1)

    return max((seconds_left + (day_count * SECONDS_PER_DAY)) / SECONDS_PER_DAY / 365.0, 1e-6)

# --- 2. BLACK-SCHOLES & GREEKS ---

@njit
def bs_price_fast(S: float, K: float, T: float, sigma: float, is_call: bool) -> float:
    """Calculates theoretical option price."""
    d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    cdf_d1 = 0.5 * (1.0 + math.erf(d1 / np.sqrt(2.0)))
    cdf_d2 = 0.5 * (1.0 + math.erf(d2 / np.sqrt(2.0)))
    
    if is_call:
        return S * cdf_d1 - K * cdf_d2
    else:
        cdf_md1 = 0.5 * (1.0 + math.erf(-d1 / np.sqrt(2.0)))
        cdf_md2 = 0.5 * (1.0 + math.erf(-d2 / np.sqrt(2.0)))
        return K * cdf_md2 - S * cdf_md1

@njit
def bs_greeks(S: float, K: float, T: float, sigma: float, is_call: bool):
    """Returns Delta, Gamma, and Theta."""
    d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    cdf_d1 = 0.5 * (1.0 + math.erf(d1 / np.sqrt(2.0)))
    
    if is_call: 
        delta = cdf_d1
    else: 
        delta = cdf_d1 - 1.0
        
    n_prime_d1 = np.exp(-d1 * d1 / 2.0) / np.sqrt(2.0 * np.pi)
    gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
    
    # Theta per day
    theta = -(S * n_prime_d1 * sigma) / (2.0 * np.sqrt(T)) / 365.0
    return delta, gamma, theta

# --- 3. IMPLIED VOLATILITY SOLVERS ---

@njit
def find_iv_numba(S: float, K: float, T: float, mprice: float, is_call: bool) -> float:
    """Bisection method to find Implied Volatility."""
    low, high = 1e-6, 5.0
    for _ in range(60):
        mid = (low + high) / 2
        if bs_price_fast(S, K, T, mid, is_call) - mprice > 0: 
            high = mid
        else: 
            low = mid
    return mid

@njit
def compute_iv_array(S: np.ndarray, K: np.ndarray, T: np.ndarray, P: np.ndarray, CallMask: np.ndarray) -> np.ndarray:
    """Computes IV across an entire array/dataframe of options data efficiently."""
    n = len(S)
    out = np.empty(n)
    for i in range(n):
        if S[i] > 0 and K[i] > 0 and T[i] > 0 and P[i] > 0:
            out[i] = find_iv_numba(S[i], K[i], T[i], P[i], CallMask[i])
        else:
            out[i] = np.nan
    return out

@njit
def compute_delta_array(S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, CallMask: np.ndarray) -> np.ndarray:
    """Computes Delta across an entire array of options data efficiently."""
    n = len(S)
    out_delta = np.empty(n)
    for i in range(n):
        if S[i] > 0 and K[i] > 0 and T[i] > 0 and sigma[i] > 0 and not np.isnan(sigma[i]):
            d, g, th = bs_greeks(S[i], K[i], T[i], sigma[i], CallMask[i])
            out_delta[i] = d
        else:
            out_delta[i] = np.nan
    return out_delta