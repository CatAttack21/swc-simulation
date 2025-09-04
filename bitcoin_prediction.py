import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from volatility_analysis import (
    generate_realistic_noise
)

def btc_power_law_formula(index):
    """Bitcoin Power Law model"""
    genesis = pd.Timestamp('2009-01-03')
    days_since_genesis = (index - genesis).days.values.astype(float)
    days_since_genesis[days_since_genesis < 1] = 1
    
    # Calculate raw power law with stronger growth
    base = 10**-17  # Increased base coefficient
    exponent = 5.8  # Increased exponent
    price = base * (days_since_genesis ** exponent)
    
    # Apply exponential growth factor for far future dates
    future_boost = np.exp(days_since_genesis / 50000)  # Gradual exponential boost
    price = price * future_boost
    
    support = 0.5 * price  # Tighter support
    resistance = 3.0 * price  # Tighter resistance
    return support, price, resistance

def predict_bitcoin_prices(start_date, end_date, last_price, historical_data=None):
    """
    Predict Bitcoin prices using power law as baseline with oscillating swings
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize predictions DataFrame
    predictions = pd.DataFrame(index=dates)
    predictions['Price'] = 0.0
    
    # Get power law model values
    support, baseline, resistance = btc_power_law_formula(dates)
    
    # Scale to match last known price while maintaining growth rate
    initial_price = float(last_price.iloc[0] if isinstance(last_price, pd.Series) else last_price)
    scale_factor = initial_price / baseline[0]
    
    # Apply scale factor to all curves and ensure it maintains through projection
    baseline = baseline * scale_factor 
    support = support * scale_factor
    resistance = resistance * scale_factor
    
    # Set initial price
    predictions.at[dates[0], 'Price'] = initial_price
    
    # Generate combined oscillation pattern
    time_index = np.arange(len(dates))
    combined_oscillation = np.zeros(len(dates))
    
    # Multiple frequency components with reduced amplitudes
    periods = [1, 5, 10, 30, 90, 180, 360, 1460]  # Short, medium, and long cycles
    amplitudes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.1]  # Much smaller amplitudes
    
    for period, amplitude in zip(periods, amplitudes):
        cycle = 2 * np.pi * time_index / period
        combined_oscillation += amplitude * np.sin(cycle)
    
    # Reduced noise component
    noise = generate_realistic_noise(len(dates), 0.1)  # Volatility
    
    # Calculate prices with power law dominance
    for i in range(1, len(dates)):
        # Minimal oscillation impact
        oscillation = 1.0 + (combined_oscillation[i] + noise[i])  # Further reduced impact
        
        # Calculate price primarily from power law
        new_price = baseline[i] * oscillation
        
        # Ensure price stays within support/resistance
        new_price = np.clip(new_price, support[i], resistance[i])
        
        # More conservative daily moves
        prev_price = predictions.at[dates[i-1], 'Price']
        max_daily_change = 0.12  # Reduced from 0.15
        max_up = prev_price * (1 + max_daily_change)
        max_down = prev_price * (1 - max_daily_change)
        new_price = np.clip(new_price, max_down, max_up)
        
        predictions.at[dates[i], 'Price'] = new_price

    return predictions['Price']
