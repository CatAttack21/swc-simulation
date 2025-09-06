import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data as pdr

def get_swc_data():
    """
    Pulls price and volume data for SWC from all 3 tickers and aggregates volume
    Tickers: SWC.AQ (primary), TSWCF (US OTC), 3M8.F (Frankfurt)
    Returns: DataFrame with date, price (from primary ticker), and aggregated volume
    """
    start_date = "2025-06-01"
    tickers = ["SWC.AQ", "TSWCF", "3M8.F"]
    
    # Get data from all tickers
    all_data = {}
    primary_ticker = "SWC.AQ"
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date)
            if not data.empty:
                all_data[ticker] = data
                print(f"Successfully downloaded {ticker} data: {len(data)} days")
            else:
                print(f"Warning: No data found for {ticker}")
        except Exception as e:
            print(f"Warning: Could not fetch data for {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No SWC data could be retrieved from any ticker")
    
    # Use primary ticker for price data
    if primary_ticker in all_data:
        primary_data = all_data[primary_ticker]
    else:
        # Fallback to first available ticker if primary not available
        primary_ticker = list(all_data.keys())[0]
        primary_data = all_data[primary_ticker]
        print(f"Using {primary_ticker} as primary ticker for price data")
    
    # Create unified date index from primary ticker
    date_index = primary_data.index
    
    # Aggregate volume across all available tickers
    aggregated_volume = pd.Series(0.0, index=date_index, dtype=float)
    
    for ticker, data in all_data.items():
        # Extract volume as Series and align to primary ticker's date index
        volume_series = data['Volume']
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0]  # Take first column if DataFrame
        
        aligned_volume = volume_series.reindex(date_index, fill_value=0.0)
        # Use simple addition with proper Series objects
        aggregated_volume = aggregated_volume + aligned_volume
        print(f"Added {ticker} volume to aggregation")
    
    # Calculate volume trend using aggregated volume
    dates_as_int = (date_index - date_index[0]).days.values.reshape(-1, 1)
    
    # Handle case where volume might be zero (take log of non-zero values)
    non_zero_volume = aggregated_volume[aggregated_volume > 0]
    if len(non_zero_volume) > 1:
        non_zero_dates = dates_as_int[aggregated_volume > 0]
        volume_regression = np.polyfit(non_zero_dates.flatten(), np.log(non_zero_volume.values), 1)
    else:
        # Fallback if insufficient data
        volume_regression = [0, np.log(aggregated_volume.mean()) if aggregated_volume.mean() > 0 else 0]
    
    # Create output DataFrame
    data_usd = pd.DataFrame(index=date_index)
    data_usd['Volume'] = aggregated_volume
    data_usd['Volume_Trend'] = np.exp(volume_regression[1] + volume_regression[0] * dates_as_int.flatten())
    data_usd['Close'] = primary_data['Close'].div(150)  # Convert JPY to USD (approximate)
    data_usd['Volume_Growth'] = pd.Series([volume_regression[0]] * len(data_usd), index=date_index)
    
    print(f"Aggregated volume statistics:")
    print(f"  Total days: {len(data_usd)}")
    print(f"  Average daily volume: {aggregated_volume.mean():.0f}")
    print(f"  Max daily volume: {aggregated_volume.max():.0f}")
    print(f"  Tickers contributing: {list(all_data.keys())}")
    
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
