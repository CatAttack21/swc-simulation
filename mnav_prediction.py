import numpy as np
import pandas as pd
from datetime import datetime

def calculate_mnav_with_volatility(btc_value, days_from_start, base_volatility=0.12, end_date=None, current_date=None):
    """
    Calculates mNAV with enhanced volatility and mean reversion
    Returns: Float with calculated mNAV value including volatility
    """
    # Calculate power law baseline (theoretical fair value)
    theoretical_mcap = 1.0 + 35.1221 * (btc_value ** 0.895)
    power_law_mnav = theoretical_mcap / btc_value

    # Calculate volatility decay with safeguards
    volatility = base_volatility
    if end_date is not None and current_date is not None:
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, "%Y-%m-%d")
            
        total_days = (end_date - current_date.replace(hour=0, minute=0, second=0, microsecond=0)).days
        if total_days > 0:  # Prevent division by zero
            decay_factor = np.exp(-3.0 * days_from_start / total_days)
            volatility = base_volatility * (0.2 + 0.8 * decay_factor)

    # Add random overshooting for mean reversion targets
    if np.random.random() < 0.15:  # 15% chance of setting new overshoot target
        # Generate asymmetric overshoots
        if np.random.random() < 0.5:  # Upside overshoot
            overshoot = np.random.uniform(1.33, 2.0)
        else:  # Downside overshoot
            overshoot = np.random.uniform(0.5, 0.8)
        target_mnav = power_law_mnav * overshoot
    else:
        target_mnav = power_law_mnav

    # Calculate volatility and noise with decay
    volatility *= (1 + 0.2 * np.sin(days_from_start / 30))
    noise = np.random.normal(0, volatility)
    
    # Get previous mNAV (or use power law if first calculation)
    current_mnav = getattr(calculate_mnav_with_volatility, 'last_mnav', power_law_mnav)
    
    # Mean reversion strength varies randomly
    reversion_speed = np.random.uniform(0.05, 0.15)
    
    # Calculate new mNAV with mean reversion and noise
    new_mnav = current_mnav + (target_mnav - current_mnav) * reversion_speed + noise
    
    # Apply minimum mNAV floor
    min_mnav = power_law_mnav * 0.7
    new_mnav = max(min_mnav, new_mnav)
    
    # Store for next calculation
    calculate_mnav_with_volatility.last_mnav = new_mnav
    
    return new_mnav
