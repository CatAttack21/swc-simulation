import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data as pdr

def get_swc_data():
    """
    Pulls price and volume data for swc since May 2025
    Returns: DataFrame with date, price, and volume in USD
    """
    start_date = "2025-06-01"
    ticker = "SWC.AQ"

    # Get stock data in JPY
    data = yf.download(ticker, start=start_date)
    
    dates_as_int = (data.index - data.index[0]).days.values.reshape(-1, 1)
    volume_regression = np.polyfit(dates_as_int.flatten(), np.log(data['Volume'].values), 1)
    
    # Calculate trend for each date
    data_usd = pd.DataFrame(index=data.index)
    data_usd['Volume'] = data['Volume']
    data_usd['Volume_Trend'] = np.exp(volume_regression[1] + volume_regression[0] * dates_as_int.flatten())
    data_usd['Close'] = data['Close'].div(150)
    # Store growth rate as column with same length as data
    data_usd['Volume_Growth'] = pd.Series([volume_regression[0]] * len(data), index=data.index)
    
    return data_usd

def get_bitcoin_historical_data():
    """Pulls Bitcoin price and volume data since 2012"""
    ticker = "BTC-USD"
    start_date = "2025-05-23"  # One day before simulation start
    data = yf.download(ticker, start=start_date)
    return data[['Close', 'Volume']]

def get_previous_day_btc():
    """Gets previous day's BTC closing price"""
    ticker = "BTC-USD"
    end_date = "2025-05-24"  # Simulation start date
    start_date = "2025-05-23"  # One day before
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].iloc[-1]

def get_historical_bitcoin_data_for_cagr():
    """Pulls Bitcoin price data since 2012 for CAGR calculation"""
    ticker = "BTC-USD"
    start_date = "2012-01-01"  # Start from 2012 for historical context
    try:
        data = yf.download(ticker, start=start_date, end="2025-05-23")  # End before simulation start
        return data['Close']
    except Exception as e:
        print(f"Warning: Could not fetch historical Bitcoin data: {e}")
        return pd.Series()
