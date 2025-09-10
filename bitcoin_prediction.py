import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from volatility_analysis import (
    generate_realistic_noise
)

try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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
    future_boost = np.exp(days_since_genesis / 10000)  # Gradual exponential boost
    price = price * future_boost
    
    support = 0.63 * price  # Tighter support
    resistance = 2.5 * price  # Tighter resistance
    return support, price, resistance

def predict_bitcoin_prices(start_date, end_date, last_price, historical_data=None):
    """
    Predict Bitcoin prices using 4-year bull/bear cycle pattern
    - Bull run to ~150k by December 2025
    - Bear market crash to ~75k through 2026-2027
    - New bull run to ~300k by 2028
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize predictions DataFrame
    predictions = pd.DataFrame(index=dates)
    predictions['Price'] = 0.0
    
    # Set initial price
    initial_price = float(last_price.iloc[0] if isinstance(last_price, pd.Series) else last_price)
    predictions.at[dates[0], 'Price'] = initial_price
    
    # Define key cycle dates and prices
    cycle_points = {
        pd.Timestamp('2025-05-24'): initial_price,  # Starting point (current)
        pd.Timestamp('2025-12-31'): 150000,         # Bull market peak
        pd.Timestamp('2026-06-30'): 100000,         # First leg down
        pd.Timestamp('2026-12-31'): 85000,          # Continued decline
        pd.Timestamp('2027-06-30'): 75000,          # Bear market bottom
        pd.Timestamp('2027-12-31'): 80000,          # Slight recovery
        pd.Timestamp('2028-06-30'): 180000,         # New bull run begins
        pd.Timestamp('2028-12-31'): 300000,         # New cycle peak
        pd.Timestamp('2029-12-31'): 250000,         # Some pullback
        pd.Timestamp('2030-12-31'): 400000,         # Continued growth
    }
    
    # Create interpolated price path between key points
    cycle_dates = list(cycle_points.keys())
    cycle_prices = list(cycle_points.values())
    
    # Interpolate between cycle points for smooth transitions
    price_series = pd.Series(cycle_prices, index=cycle_dates)
    price_series = price_series.reindex(dates, method='nearest')
    
    # Apply cubic spline interpolation for smoother curves if scipy is available
    if SCIPY_AVAILABLE:
        # Convert dates to numeric for interpolation
        date_nums = np.array([(d - dates[0]).days for d in cycle_dates])
        all_date_nums = np.array([(d - dates[0]).days for d in dates])
        
        # Create cubic spline
        spline = interpolate.CubicSpline(date_nums, cycle_prices, bc_type='natural')
        smooth_prices = spline(all_date_nums)
    else:
        # Fallback to linear interpolation
        price_series = pd.Series(cycle_prices, index=cycle_dates)
        price_series = price_series.reindex(dates).interpolate(method='linear')
        smooth_prices = price_series.values
    
    # Add realistic volatility and noise
    for i in range(1, len(dates)):
        base_price = smooth_prices[i]
        
        # Add daily volatility (higher during bear markets, lower during bull runs)
        current_date = dates[i]
        
        # Determine market phase for volatility scaling
        if current_date < pd.Timestamp('2026-01-01'):
            # Bull market phase - lower volatility
            daily_vol = 0.03
        elif current_date < pd.Timestamp('2027-07-01'):
            # Bear market phase - higher volatility
            daily_vol = 0.05
        else:
            # Recovery/new bull - moderate volatility
            daily_vol = 0.04
        
        # Generate realistic price movements
        prev_price = predictions.at[dates[i-1], 'Price']
        
        # Random walk component
        random_factor = np.random.normal(1.0, daily_vol)
        
        # Trend following toward target price
        trend_factor = 0.02  # How fast to converge to target
        trend_adjustment = 1 + (base_price - prev_price) / prev_price * trend_factor
        
        # Combine factors
        new_price = prev_price * random_factor * trend_adjustment
        
        # Limit extreme daily moves
        max_daily_change = 0.15  # 15% max daily change
        max_up = prev_price * (1 + max_daily_change)
        max_down = prev_price * (1 - max_daily_change)
        new_price = np.clip(new_price, max_down, max_up)
        
        # Ensure we don't drift too far from target path
        target_price = base_price
        max_deviation = 0.3  # 30% max deviation from target
        min_bound = target_price * (1 - max_deviation)
        max_bound = target_price * (1 + max_deviation)
        new_price = np.clip(new_price, min_bound, max_bound)
        
        predictions.at[dates[i], 'Price'] = new_price

    return predictions['Price']
