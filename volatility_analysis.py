import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

def calculate_historical_volatility(prices, window=30):
    """Calculate rolling volatility from historical prices"""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)

def calculate_implied_volatility(prices, window=30):
    """Calculate implied volatility from stock prices"""
    # Calculate daily returns
    returns = prices.pct_change()
    # Calculate rolling standard deviation and annualize
    implied_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return implied_vol

def fit_power_law_params(prices):
    """Fit power law parameters to price data"""
    # Handle DataFrame input
    if isinstance(prices, pd.DataFrame):
        price_data = prices['Close'].values if 'Close' in prices.columns else prices.iloc[:, 0].values
    else:
        price_data = np.array(prices)
    
    time_index = np.arange(len(price_data))
    log_prices = np.log(price_data)
    
    slope, intercept, r_value, _, _ = stats.linregress(
        np.log(time_index + 1),
        log_prices
    )
    return np.exp(intercept), slope

def calculate_support_resistance(prices, window=20):
    """Calculate dynamic support and resistance levels"""
    rolling_low = prices.rolling(window=window).min()
    rolling_high = prices.rolling(window=window).max()
    support_slope = LinearRegression().fit(
        np.arange(len(rolling_low)).reshape(-1, 1),
        rolling_low.fillna(method='ffill')
    ).coef_[0]
    resistance_slope = LinearRegression().fit(
        np.arange(len(rolling_high)).reshape(-1, 1),
        rolling_high.fillna(method='ffill')
    ).coef_[0]
    return support_slope, resistance_slope

def generate_realistic_noise(size, hist_volatility):
    """Generate realistic price noise based on historical patterns"""
    base_noise = np.random.normal(0, 1, size)
    # Add fat tails characteristic of crypto markets
    fat_tail_noise = np.random.standard_t(df=3, size=size)
    combined_noise = 0.7 * base_noise + 0.3 * fat_tail_noise
    return combined_noise * hist_volatility
