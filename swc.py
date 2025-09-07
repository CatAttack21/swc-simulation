import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
from bitcoin_prediction import predict_bitcoin_prices
from mnav_prediction import calculate_mnav_with_volatility
from data_sources import (
    get_swc_data,
    get_bitcoin_historical_data,
    get_previous_day_btc,
    get_historical_bitcoin_data_for_cagr
)
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import CustomBusinessDay
from volatility_analysis import (
    calculate_historical_volatility,
    calculate_implied_volatility,
    fit_power_law_params,
    calculate_support_resistance,
    generate_realistic_noise
)

def get_bitcoin_holdings():
    """
    Gets swc's Bitcoin holdings since beginning
    Returns: Tuple of (holdings Series, purchase DataFrame)
    """
    # Read the CSV file
    df = pd.read_csv('swc_btc.csv')

    # Clean up BTC Holding column and convert to float
    df['BTC Holding'] = df['BTC Holding'].astype(float)
    df['Reported'] = pd.to_datetime(df['Reported'])
    df = df.sort_values('Reported')
    
    # Calculate purchases with proper indexing
    df['BTC_Purchased'] = df['BTC Holding'].diff()
    purchases = pd.DataFrame(index=df['Reported'])
    purchases['BTC_Purchased'] = df['BTC_Purchased']
    purchases = purchases[purchases['BTC_Purchased'] > 0]  # Filter after DataFrame is created
    
    # Create holdings series
    date_range = pd.date_range(start=df['Reported'].min(), end=datetime.now().date(), freq='D')
    btc_holdings = pd.Series(index=date_range, dtype=float)
    btc_holdings[df['Reported']] = df['BTC Holding']
    btc_holdings = btc_holdings.ffill()  # Use ffill() instead of fillna(method='ffill')
    
    return btc_holdings, purchases

def calculate_mnav(market_cap, btc_holdings, btc_price):
    """
    Calculates mNAV based on market cap and BTC value
    Returns: Series with mNAV values
    """
    btc_value = btc_holdings * btc_price
    return market_cap / btc_value

def calculate_mnav_correlated_volume(current_mnav, base_volume_pct=0.025, min_volume_pct=0.01, max_volume_pct=0.05):
    """
    Calculates volume percentage correlated to mNAV cyclical model
    Higher mNAV = Higher volume (more interest/activity)
    Range: 1% to 5% of shares outstanding
    Args:
        current_mnav: Current mNAV value
        base_volume_pct: Base volume percentage at mNAV = 1.5 (2.5%)
        min_volume_pct: Minimum volume percentage (1%)
        max_volume_pct: Maximum volume percentage (5%)
    Returns: Volume percentage correlated to mNAV
    """
    # Normalize mNAV to a correlation factor
    # mNAV 0.8-1.3: Low volume (discount/fair value)
    # mNAV 1.3-2.5: Moderate volume (moderate premium)  
    # mNAV 2.5+: High volume (high premium/speculation)
    
    if current_mnav <= 1.0:
        # Below fair value: low volume (1.0% to 2.0%)
        volume_factor = 0.4 + 0.4 * current_mnav  # 0.4 to 0.8 multiplier
    elif current_mnav <= 2.0:
        # Fair to moderate premium: normal volume (2.0% to 3.5%)
        volume_factor = 0.8 + 0.6 * (current_mnav - 1.0)  # 0.8 to 1.4 multiplier
    else:
        # High premium: high volume with diminishing returns (3.5% to 5.0%)
        excess_mnav = current_mnav - 2.0
        volume_factor = 1.4 + 0.6 * np.log(1 + excess_mnav) / np.log(4)  # Scaled log growth
    
    # Apply volume factor to base percentage
    correlated_volume_pct = base_volume_pct * volume_factor
    
    # Ensure bounds (1% to 5%)
    return max(min_volume_pct, min(max_volume_pct, correlated_volume_pct))

def calculate_volume_dampening_factor(days_from_start, initial_shares, current_shares):
    """
    Calculates a dampening factor to slow down volume growth as share count increases
    Returns: Float between 0.1 and 1.0 to reduce effective volume percentage
    """
    # Calculate share count growth ratio
    share_growth_ratio = current_shares / initial_shares
    
    # Time-based dampening: reduces volume growth over time
    time_dampening = np.exp(-0.5 * days_from_start / 365.25)  # Decay over years
    
    # Share-count-based dampening: reduces as shares grow
    # More shares = lower volume percentage to prevent infinite growth
    share_dampening = 1.0 / np.sqrt(share_growth_ratio)  # Square root dampening
    
    # Combine both factors with minimum floor of 0.1
    combined_dampening = time_dampening * share_dampening
    return max(0.1, min(1.0, combined_dampening))

def predict_future_mnav(days_from_start, btc_value):
    """
    Predicts future mNAV using the Weierstrass volatility model
    Returns: Float with predicted mNAV value
    """
    return calculate_mnav_with_volatility(btc_value, days_from_start)

def calculate_btc_per_share(btc_holdings, outstanding_shares):
    """
    Calculates BTC per share
    Returns: Series with BTC per share values
    """
    return btc_holdings / outstanding_shares

def predict_future_mnav(historical_mnav, btc_price_prediction):
    """
    Predicts future mNAV based on BTC correlation
    Returns: Series with predicted mNAV values
    """
    correlation = historical_mnav.corr(btc_price_prediction)
    return historical_mnav * correlation * (btc_price_prediction / btc_price_prediction.iloc[0])

def predict_mnav_with_swings(historical_mnav, btc_price):
    """
    Predicts mNAV considering historical swings
    Returns: Series with predicted mNAV values
    """
    volatility = historical_mnav.std()
    base_prediction = predict_future_mnav(historical_mnav, btc_price)
    return base_prediction * (1 + np.random.normal(0, volatility, len(base_prediction)))

def calculate_stock_price(predicted_mnav, btc_holdings, current_shares, btc_price):
    """
    Calculates stock price based on predicted mNAV, BTC holdings and price
    Returns: Float with predicted stock price
    """
    return (btc_holdings * btc_price * predicted_mnav) / current_shares

def calculate_daily_dilution(price_data, volume_data, days_since_last=0, dilution_rate=0.20, current_mnav=1.0):
    """
    Calculates daily dilution and funds raised with simplified logic:
    - Dilute 20% of daily volume when price increases from previous day
    - Only dilute if mNAV is above 1.1
    Volume should be daily trading volume, not outstanding shares
    Returns: DataFrame with dilution amount and funds raised
    """
    # For single day calculations, compare with previous day's price
    if len(price_data) == 1:
        # Default to no dilution if no previous price
        price_increase = 0.0
        if hasattr(price_data.index[0], 'strftime'):
            prev_day = price_data.index[0] - pd.Timedelta(days=1)
            if prev_day in price_data.index:
                price_increase = (price_data.iloc[0] / price_data.loc[prev_day]) - 1
    else:
        price_increase = price_data.pct_change().iloc[-1]
    
    dilution = pd.DataFrame(index=price_data.index)
    dilution['dilution_shares'] = 0.0
    dilution['funds_raised'] = 0.0
    
    date = price_data.index[-1]
    # Simple rule: dilute 20% of daily volume when price increases from previous day AND mNAV > 1.1
    if price_increase > 0.03 and current_mnav > 1.1:
        # Calculate dilution as 5% of daily trading volume
        dilution.loc[date, 'dilution_shares'] = volume_data.loc[date] * 0.10
        dilution.loc[date, 'funds_raised'] = dilution.loc[date, 'dilution_shares'] * price_data.loc[date]
    
    return dilution

def calculate_bitcoin_purchases(funds_raised, btc_price):
    """
    Calculates Bitcoin purchases from dilution proceeds
    Returns: Series with BTC amounts purchased
    """
    return funds_raised / btc_price

def predict_bitcoin_price(historical_data, end_date="2030-12-31"):
    """
    Predicts Bitcoin price using combined models
    Args:
        historical_data: Series or DataFrame with bitcoin prices
        end_date: String date format YYYY-MM-DD
    Returns: Series of predicted prices
    """
    if historical_data.empty:
        last_price = 65000  # Default price if no historical data
    else:
        if isinstance(historical_data, pd.DataFrame):
            last_price = historical_data['Close'].iloc[-1]
        else:
            last_price = historical_data.iloc[-1]
    
    start_date = datetime.now().date() if historical_data.empty else historical_data.index[-1] + timedelta(days=1)
    
    future_prices = predict_bitcoin_prices(
        start_date=start_date,
        end_date=end_date,
        last_price=last_price,
        historical_data=historical_data
    )
    
    # Ensure future_prices is a float Series before assignment
    future_prices = future_prices.astype(float)
    return future_prices

def is_tse_trading_day(date):
    """Check if date is a Tokyo Stock Exchange trading day"""
    # TSE is closed on weekends
    if date.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
        return False
        
    # UK holidays (simplified list - add more as needed)
    holidays = [
        "2024-01-01", "2024-04-07", "2024-04-08",  # New Year, Easter Sunday, Easter Monday
        "2024-05-06", "2024-05-27", "2024-08-26",  # Early May Bank Holiday, Spring Bank Holiday, Summer Bank Holiday
        "2024-12-25", "2024-12-26"                 # Christmas Day, Boxing Day
    ]
    return str(date.date()) not in holidays

def calculate_preferred_shares_revenue(market_cap, current_date=None):
    """
    Calculates revenue from preferred shares based on percentage of dilution proceeds.
    All proceeds go directly to Bitcoin purchases.
    Returns: Tuple of (revenue amount for BTC, dividend reserve amount) in USD
    """
    if not current_date:
        return 0.0, 0.0

    # Get current and previous share counts
    current_shares = get_preferred_shares_count(current_date)
    prev_date = current_date - pd.Timedelta(days=1)
    prev_shares = get_preferred_shares_count(prev_date)
    
    # Calculate new shares issued
    new_shares = max(0, current_shares - prev_shares)
    
    # Calculate revenue from new share sales
    face_value = 100  # $100 per preferred share
    revenue = new_shares * face_value
    
    # Calculate dividend reserve requirement
    dividend_reserve = calculate_preferred_dividend_reserve(current_date)
    
    return revenue, dividend_reserve

def calculate_preferred_dividend_reserve(current_date):
    """
    Calculates how much money needs to be reserved for preferred share dividends
    Returns: Float amount needed for next quarter's dividend in USD
    """
    face_value = 100  # $100 per preferred share
    annual_yield = 0.08  # 8% annual yield
    quarterly_yield = annual_yield / 4
    
    # Get number of preferred shares (mock data - replace with actual tracking)
    preferred_shares = get_preferred_shares_count(current_date)
    
    # Calculate quarterly dividend requirement
    quarterly_dividend = face_value * quarterly_yield * preferred_shares
    
    # Reserve requirement (one quarter's worth)
    return quarterly_dividend

def get_preferred_shares_count(current_date, end_date=None):
    """
    Gets the number of preferred shares outstanding using a single logistic growth S-curve.
    Uses sigmoid function to model natural adoption curve from 0 to 100M shares.
    Args:
        current_date: Current date to calculate shares for
        end_date: End date of simulation for S-curve calculation (defaults to 2040-12-31)
    Returns: Integer number of shares
    """
    start_date = pd.Timestamp('2026-06-01')
    if end_date is None:
        end_date = pd.Timestamp('2040-12-31')
    else:
        end_date = pd.Timestamp(end_date)
    max_shares = 10_000_000  # Maximum 1M shares

    if current_date < start_date:
        return 0
        
    days_from_start = (current_date - start_date).days
    total_days = (end_date - start_date).days
    
    # Normalized time from -4 to 4 for a more gradual sigmoid curve
    # Lower multiplier = less steep S-curve = slower initial growth
    normalized_time = 5 * (days_from_start / total_days - 0.5)
    
    # Sigmoid function: 1 / (1 + e^-x)
    # This creates the S-curve shape
    progress = 1 / (1 + np.exp(-normalized_time))
    
    # Calculate shares based on progress 
    shares = max_shares * progress
    
    return int(max(0, min(max_shares, shares)))

def calculate_weekly_revenue(market_cap, current_date=None):
    """
    Calculates weekly revenue from all income streams
    Returns: Float revenue amount in USD
    """
    annual_rate = 0.005  # 0.5% annually
    weekly_rate = annual_rate / 52
    secondary_revenue = market_cap * weekly_rate
    
    if current_date and current_date >= pd.Timestamp('2026-01-01'):
        preferred_revenue, dividend_reserve = calculate_preferred_shares_revenue(market_cap, current_date)
        return secondary_revenue + preferred_revenue
    
    return secondary_revenue

def calculate_rolling_cagr(prices, window=365):
    """Calculate rolling 1-year Compound Annual Growth Rate for given price series"""
    # Calculate compound annual growth rate: (end_value/start_value)^(1/years) - 1
    # For 1-year window, time period = 1, so formula simplifies to:
    start_prices = prices.shift(window)
    rolling_cagr = np.power(prices / start_prices, 1.0/1.0) - 1
    # Convert to percentage
    rolling_cagr = rolling_cagr * 100
    return rolling_cagr

def simulate_through_2040(btc_data, swc_data, initial_shares, btc_holdings, start_date=None, end_date="2040-12-31", enable_preferred_shares=True):
    """Simulates swc metrics through 2040"""
    # Use current date if no start date provided
    if start_date is None:
        start_date = "2025-05-24"
    
    sim_start = pd.Timestamp(start_date)
    sim_end = pd.Timestamp(end_date)
    
    # Load historical BTC holdings and get last known BTC price
    df = pd.read_csv('swc_btc.csv')
    df['Reported'] = pd.to_datetime(df['Reported'])
    df = df.sort_values('Reported')
    initial_btc = float(df['BTC Holding'].iloc[-1])
    
    # Get previous day's closing price
    last_btc_price = get_previous_day_btc()
    
    # Calculate initial BTC NAV and market cap
    initial_btc_nav = initial_btc * last_btc_price
    market_cap = initial_btc_nav  # Start at exact BTC NAV
    initial_mnav = 1.0  # Start at exact BTC NAV ratio
    
    # Initialize simulation DataFrame
    future_dates = pd.date_range(start=sim_start, end=sim_end, freq='D')
    simulation = pd.DataFrame(index=future_dates)
    simulation['is_trading_day'] = simulation.index.map(is_tse_trading_day)
    simulation['btc_holdings'] = initial_btc
    simulation['btc_price'] = last_btc_price  # Initialize with last known price
    
    # Initialize starting values with actual market data
    prev_stock_price = market_cap / initial_shares
    prev_mnav = initial_mnav

    # Track cumulative values
    current_btc = initial_btc  # Start with historical final value
    current_shares = float(initial_shares)
    cumulative_btc_purchased = 0.0  # Track only new purchases

    # Calculate total simulation days (still needed for other calculations)
    total_days = (sim_end - sim_start).days

    # Simplified dilution: 20% of daily volume when price increases
    # No complex parameters needed anymore
    
    # Removed complex get_dilution_rate function - now using simple 20% rule

    # Add historical BTC prices where available
    historical_data = btc_data.reindex(future_dates)
    simulation['btc_price'] = historical_data['Close']
    
    # Get future Bitcoin price predictions for missing dates
    missing_dates = simulation[simulation['btc_price'].isna()].index
    if len(missing_dates) > 0:
        last_known_date = simulation[simulation['btc_price'].notna()].index[-1]
        last_known_price = simulation.loc[last_known_date, 'btc_price']
        future_prices = predict_bitcoin_price(btc_data, end_date)
        # Ensure index alignment and proper dtype
        simulation.loc[missing_dates, 'btc_price'] = future_prices.reindex(missing_dates).astype(float)
    
    # Fill any remaining gaps
    simulation['btc_price'] = simulation['btc_price'].ffill()
    
    # Initialize starting values
    prev_stock_price = float(swc_data['Close'].iloc[-1] if not swc_data.empty else 5.0)
    
    # Initialize mNAV based on both models
    initial_btc_nav = current_btc * simulation['btc_price'].iloc[0]
    theoretical_mcap = 42.5221 * (initial_btc_nav ** 0.92)  # Increased power law relationship
    initial_mnav = theoretical_mcap / initial_btc_nav
    prev_mnav = initial_mnav
    
    # Get historical trading volume data
    if not swc_data.empty and 'Volume' in swc_data.columns:
        # Get last 30 days of volume data
        recent_volume = swc_data['Volume'].tail(30)
        # Calculate average daily volume as percentage of shares
        initial_volume_pct = recent_volume.median() / initial_shares
        # Keep volume stable around 3% with minimal variation
        initial_volume_pct = max(0.029, min(0.031, initial_volume_pct))
    else:
        initial_volume_pct = 0.03  # Default to exactly 3% initial volume

    # Configure exponential decay parameters
    decay_rate = -np.log(1.0) / total_days  # Decay to achieve 10% asymptote
    base_volatility = 0.5  # 30% base volatility

    # Add volume cycle counter for weekly pattern
    days_in_week = 0
    weekly_volume_factor = 1.0

    # Add dilution cycle counter
    days_since_dilution = 0
    
    # Add revenue tracking
    last_revenue_date = sim_start - timedelta(days=1)  # Start counting from day 1
    
    # Add dividend tracking
    simulation['preferred_shares'] = simulation.index.map(lambda x: get_preferred_shares_count(x, end_date) if enable_preferred_shares else 0)
    simulation['dividend_reserve'] = 0.0
    simulation['quarterly_dividend'] = 0.0
    
    # Track last dividend date
    last_dividend_date = pd.Timestamp('2026-01-01') - pd.Timedelta(days=1)
    
    # Store previous day's price for dilution calculation
    prev_prices = pd.Series(index=simulation.index)
    
    for date in simulation.index:
        btc_price = simulation.loc[date, 'btc_price']
        btc_value = current_btc * btc_price
        market_cap = btc_value * prev_mnav
        days_from_start = (date - sim_start).days
        
        # Convert end_date to datetime if it's a string
        if isinstance(end_date, str):
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_date_obj = end_date
        
        # Calculate current dilution rate for mNAV dampening
        # Get the last dilution rate from recent activity
        current_dilution_rate_pct = getattr(simulate_through_2040, 'last_dilution_rate_pct', 0.0)
            
        current_mnav = calculate_mnav_with_volatility(
            btc_value, 
            days_from_start,
            end_date=end_date_obj,
            current_date=date,
            dilution_rate_pct=current_dilution_rate_pct
        )
        
        # Calculate mNAV-correlated volume instead of fixed percentage
        mnav_correlated_volume_pct = calculate_mnav_correlated_volume(current_mnav)
        
        # Add small random noise for daily variation
        vol_noise = np.random.normal(1.0, 0.05)  # 5% daily noise
        final_volume_pct = mnav_correlated_volume_pct * max(0.9, min(1.1, vol_noise))
        
        daily_volume = current_shares * final_volume_pct

        # Only update stock price and apply dilution on trading days
        if simulation.loc[date, 'is_trading_day']:
            stock_price = calculate_stock_price(current_mnav, current_btc, current_shares, btc_price)
            prev_prices[date] = stock_price
            
            if date >= sim_start:
                # Get current dilution rate from S-curve
                # Simplified dilution logic: 20% of daily volume when price increases
                # Create price series with previous day's price for proper return calculation
                price_series = pd.Series({
                    date - pd.Timedelta(days=1): prev_prices.loc[:date].dropna().iloc[-2] if len(prev_prices.loc[:date].dropna()) > 1 else stock_price * 0.99,
                    date: stock_price
                })
                
                # Calculate dilution using simplified logic (20% of daily trading volume, only if mNAV > 1.1)
                daily_dilution = calculate_daily_dilution(
                    price_series,
                    pd.Series({date: daily_volume}),
                    days_since_dilution,
                    0.20,  # Fixed 20% rate
                    current_mnav  # Pass current mNAV to check threshold
                )
                
                if not daily_dilution.empty and daily_dilution.loc[date, 'dilution_shares'] > 0:
                    new_shares = daily_dilution.loc[date, 'dilution_shares']
                    funds_raised = daily_dilution.loc[date, 'funds_raised']
                    btc_purchased = funds_raised / btc_price
                    
                    # Calculate dilution rate as percentage of outstanding shares
                    dilution_rate_pct = (new_shares / current_shares) * 100
                    
                    # Store dilution rate for mNAV dampening (with decay)
                    previous_rate = getattr(simulate_through_2040, 'last_dilution_rate_pct', 0.0)
                    # Use exponential decay: new rate has 70% weight, previous 30%
                    simulate_through_2040.last_dilution_rate_pct = 0.7 * dilution_rate_pct + 0.3 * previous_rate
                    
                    # Apply dilution
                    current_shares += new_shares
                    current_btc += btc_purchased
                    cumulative_btc_purchased += btc_purchased
                    days_since_dilution = 0  # Reset the counter
                else:
                    days_since_dilution += 1
                    btc_purchased = 0.0  # Explicit zero when no dilution occurs
                    
                    # Decay the dilution rate when no dilution occurs
                    previous_rate = getattr(simulate_through_2040, 'last_dilution_rate_pct', 0.0)
                    simulate_through_2040.last_dilution_rate_pct = previous_rate * 0.95  # 5% decay per day
                
                # Check for quarterly dividend and preferred share revenue
                if date >= pd.Timestamp('2026-01-01') and enable_preferred_shares:
                    # Handle preferred share sales revenue
                    pref_revenue, dividend_reserve = calculate_preferred_shares_revenue(market_cap, date)
                    if pref_revenue > 0:
                        # Convert preferred share revenue directly to BTC
                        btc_from_pref = pref_revenue / btc_price
                        current_btc += btc_from_pref
                    
                    # Handle quarterly dividends
                    days_since_dividend = (date - last_dividend_date).days
                    if days_since_dividend >= 90:  # Quarterly
                        preferred_count = simulation.loc[date, 'preferred_shares']
                        quarterly_amount = (100 * 0.05 / 4) * preferred_count  # $5 annual per $100 share
                        simulation.loc[date, 'quarterly_dividend'] = quarterly_amount
                        
                        # Instead of reducing BTC purchases, use new dilution for dividends
                        dividend_shares = quarterly_amount / stock_price
                        current_shares += dividend_shares
                        simulation.loc[date, 'dividend_dilution'] = dividend_shares
                        last_dividend_date = date
                
                simulation.loc[date, 'shares_outstanding'] = current_shares
                simulation.loc[date, 'btc_purchased'] = btc_purchased
                prev_stock_price = stock_price
        else:
            stock_price = prev_stock_price if prev_stock_price is not None else 5.0
            btc_purchased = 0.0  # Explicit zero on non-trading days
        
        # Store simulation results
        simulation.loc[date, 'stock_price'] = stock_price
        simulation.loc[date, 'volume'] = daily_volume if simulation.loc[date, 'is_trading_day'] else 0
        simulation.loc[date, 'shares_outstanding'] = current_shares
        simulation.loc[date, 'btc_holdings'] = current_btc
        simulation.loc[date, 'mnav'] = current_mnav
        simulation.loc[date, 'market_cap'] = market_cap
        simulation.loc[date, 'btc_purchased'] = btc_purchased  # Now always defined
        simulation.loc[date, 'weekly_revenue'] = calculate_weekly_revenue(market_cap) if (date - last_revenue_date).days >= 7 else 0.0

    # Forward fill any missing values in final results
    simulation = simulation.ffill()  # Use ffill() instead of fillna(method='ffill')
    
    return simulation

def generate_yearly_metrics(simulation):
    """
    Generates yearly metrics summary using 3-month EMA
    Returns: DataFrame with yearly metrics
    """
    # Get all years in the simulation
    years = sorted(set(simulation.index.year))
    
    # Create yearly metrics DataFrame
    yearly_metrics = pd.DataFrame()
    
    for year in years:
        # Get data for the year
        year_data = simulation[simulation.index.year == year]
        if not year_data.empty:
            # Calculate 3-month (90-day) EMA for all metrics at year end
            ema_span = 90
            last_date = year_data.index[-1]
            
            # Get last 90 days of data using loc
            last_90_days_start = last_date - pd.Timedelta(days=90)
            last_90_days = year_data.loc[last_90_days_start:last_date]
            
            yearly_metrics.loc[year, 'BTC Price'] = last_90_days['btc_price'].ewm(span=ema_span).mean().iloc[-1]
            yearly_metrics.loc[year, 'Stock Price'] = last_90_days['stock_price'].ewm(span=ema_span).mean().iloc[-1]
            yearly_metrics.loc[year, 'BTC Holdings'] = last_90_days['btc_holdings'].ewm(span=ema_span).mean().iloc[-1]
            yearly_metrics.loc[year, 'Shares Outstanding'] = last_90_days['shares_outstanding'].ewm(span=ema_span).mean().iloc[-1]
            yearly_metrics.loc[year, 'mNAV'] = last_90_days['mnav'].ewm(span=ema_span).mean().iloc[-1]
            yearly_metrics.loc[year, 'Market Cap (USD)'] = last_90_days['market_cap'].ewm(span=ema_span).mean().iloc[-1]
            yearly_metrics.loc[year, 'BTC per 1000 Shares'] = (
                yearly_metrics.loc[year, 'BTC Holdings'] / 
                yearly_metrics.loc[year, 'Shares Outstanding']
            ) * 1000
            
            # Calculate cumulative dividends for the year
            year_dividends = year_data['quarterly_dividend'].sum()
            if year_dividends > 0:  # Only calculate ratio if dividends were paid
                yearly_metrics.loc[year, 'Market Cap to Dividends Ratio'] = (
                    yearly_metrics.loc[year, 'Market Cap (USD)'] / year_dividends
                )
    
    return yearly_metrics

def plot_simulation_results(simulation, enable_preferred_shares=True):
    """Plot simulation results with aligned axes"""
    # Read historical BTC holdings
    historical_df = pd.read_csv('swc_btc.csv')
    historical_df['Reported'] = pd.to_datetime(historical_df['Reported'])
    historical_df = historical_df.sort_values('Reported')
    
    # Set common x-axis limits
    start_date = simulation.index[0]  # Use simulation start date instead of today
    end_date = simulation.index[-1]
    date_formatter = plt.matplotlib.dates.DateFormatter('%Y-%m')

    # Create complete BTC holdings series
    complete_holdings = pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='D'))
    
    # Only include historical dates that fall within our simulation period
    valid_historical_dates = historical_df[
        (historical_df['Reported'] >= start_date) & 
        (historical_df['Reported'] <= end_date)
    ]
    
    if not valid_historical_dates.empty:
        complete_holdings[valid_historical_dates['Reported']] = valid_historical_dates['BTC Holding']
        complete_holdings = complete_holdings.ffill()
    
    # Add simulated data
    complete_holdings[simulation.index] = simulation['btc_holdings']

    # Update simulation's BTC holdings to match complete series
    simulation['btc_holdings'] = complete_holdings[simulation.index]
    
    # Create figure with standard subplot grid
    fig = plt.figure(figsize=(15, 36))  # Increased height for new plot
    gs = GridSpec(10, 2, figure=fig)  # Changed from 9 to 10 rows

    def format_millions(x, pos):
        """Format large numbers in millions"""
        return f'{x/1e6:.1f}M'

    def format_thousands(x, pos):
        """Format large numbers in thousands"""
        return f'{x/1e3:.1f}K'

    millions_formatter = plt.FuncFormatter(format_millions)
    thousands_formatter = plt.FuncFormatter(format_thousands)
    
    # Left Column (Bitcoin metrics)
    ax1 = fig.add_subplot(gs[0, 0])  # Bitcoin Price
    ax1.plot(simulation.index, simulation['btc_price'], 'orange', label='BTC Price', linewidth=2)
    ax1.set_ylabel('BTC Price (USD)')
    ax1.set_title('Bitcoin Price')
    ax1.set_ylim(simulation['btc_price'].min() * 0.95, simulation['btc_price'].max() * 1.05)
    if simulation['btc_price'].max() > 1e6:
        ax1.yaxis.set_major_formatter(millions_formatter)
    
    ax2 = fig.add_subplot(gs[1, 0])  # Bitcoin CAGR
    historical_btc = get_historical_bitcoin_data_for_cagr()
    all_prices = pd.Series()
    if not historical_btc.empty:
        all_prices = pd.concat([historical_btc, simulation['btc_price']])
        all_prices = all_prices[~all_prices.index.duplicated(keep='last')]
        all_prices = all_prices.sort_index()
    else:
        all_prices = simulation['btc_price']
    rolling_cagr = calculate_rolling_cagr(all_prices)
    ax2.plot(rolling_cagr.index, rolling_cagr, 'orange', label='1-Year CAGR', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('CAGR (%)')
    ax2.set_title('Bitcoin 1-Year Rolling CAGR')
    ax2.set_ylim(-20, 150)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

    ax3 = fig.add_subplot(gs[2, 0])  # Bitcoin Holdings
    ax3.plot(complete_holdings.index, complete_holdings.values, 'b-', label='BTC Holdings', linewidth=2)
    ax3.set_ylabel('BTC Holdings (Thousands)')
    ax3.set_title('Bitcoin Holdings')
    ax3.set_ylim(0, complete_holdings.max() * 1.05)
    ax3.yaxis.set_major_formatter(thousands_formatter)  # Changed to thousands formatter
    
    ax4 = fig.add_subplot(gs[3, 0])  # BTC per 1000 shares
    btc_per_1000 = (simulation['btc_holdings'] / simulation['shares_outstanding']) * 1000
    ax4.plot(simulation.index, btc_per_1000, 'b-', label='BTC per 1000 Shares', linewidth=2)
    ax4.set_ylabel('BTC Amount')
    ax4.set_title('Bitcoin per 1000 Shares')
    ax4.set_ylim(btc_per_1000.min() * 0.95, btc_per_1000.max() * 1.05)
    
    ax5 = fig.add_subplot(gs[4, 1])  # Daily Volume (swapped to right column)
    ax5.plot(simulation.index, simulation['volume'], 'r-', label='Daily Volume', linewidth=2)
    ax5.set_ylabel('Number of Shares')
    ax5.set_title('Daily Trading Volume')
    ax5.set_ylim(0, simulation['volume'].max() * 1.05)
    ax5.yaxis.set_major_formatter(millions_formatter)

    # Right Column (Stock metrics)
    ax6 = fig.add_subplot(gs[0, 1])  # Stock Price
    ax6.plot(simulation.index, simulation['stock_price'], 'g-', label='Stock Price', linewidth=2)
    ax6.set_ylabel('Stock Price (USD)')
    ax6.set_title('swc Stock Price')
    ax6.set_ylim(simulation['stock_price'].min() * 0.95, simulation['stock_price'].max() * 1.05)
    
    ax7 = fig.add_subplot(gs[1, 1])  # mNAV
    ax7.plot(simulation.index, simulation['mnav'], 'g-', label='mNAV', linewidth=2)
    ax7.set_ylabel('mNAV')
    ax7.set_title('mNAV')
    ax7.set_ylim(simulation['mnav'].min() * 0.95, simulation['mnav'].max() * 1.05)
    
    ax8 = fig.add_subplot(gs[2, 1])  # Implied Volatility
    implied_vol = calculate_implied_volatility(simulation['stock_price'])
    ax8.plot(simulation.index, implied_vol, 'g-', label='30-Day Implied Vol', linewidth=2)
    ax8.set_ylabel('Volatility (%)')
    ax8.set_title('Stock Implied Volatility')
    ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax8.set_ylim(75, 300)

    ax9 = fig.add_subplot(gs[3, 1])  # Shares Outstanding
    ax9.plot(simulation.index, simulation['shares_outstanding'], 'r-', 
            label='Shares Outstanding', linewidth=2)
    ax9.set_ylabel('Shares Outstanding (Millions)')
    ax9.set_title('Shares Outstanding')
    ax9.set_ylim(simulation['shares_outstanding'].min() * 0.95, 
                simulation['shares_outstanding'].max() * 1.05)
    ax9.yaxis.set_major_formatter(millions_formatter)

    # Daily BTC Purchased plot
    ax10 = fig.add_subplot(gs[4, 0])  # Swapped to left column
    ax10.plot(simulation.index, simulation['btc_purchased'], 'r-', 
            label='Daily BTC Purchased', linewidth=2)
    ax10.set_ylabel('BTC Amount')
    ax10.set_title('Daily BTC Purchased from Dilution')
    ax10.set_ylim(0, simulation['btc_purchased'].max() * 1.05)

    # Add new Daily Share Dilution plot
    ax10b = fig.add_subplot(gs[5, 0])  # Place in left column
    daily_dilution = simulation['shares_outstanding'].diff().fillna(0)
    ax10b.plot(simulation.index, daily_dilution / 1e6, 'r-',  # Convert to millions
            label='Daily Share Dilution', linewidth=2)
    ax10b.set_ylabel('Shares (Millions)')
    ax10b.set_title('Daily Share Dilution')
    ax10b.set_ylim(0, (daily_dilution / 1e6).max() * 1.05)
    
    # Shift remaining plots down one position
    if enable_preferred_shares:
        ax11 = fig.add_subplot(gs[6, 0])  # Preferred Shares (swapped with cumulative dividends)
        ax11.plot(simulation.index, simulation['preferred_shares'], 'purple', 
                label='Preferred Shares', linewidth=2)
        ax11.set_ylabel('Number of Shares')
        ax11.set_title('Preferred Shares Outstanding')
        ax11.yaxis.set_major_formatter(millions_formatter)
        
        ax12 = fig.add_subplot(gs[7, 0])  # Cumulative Dividends (swapped with preferred shares)
        cumulative_dividends = simulation['quarterly_dividend'].cumsum()
        ax12.plot(simulation.index, cumulative_dividends, 'purple', 
                label='Cumulative Dividends', linewidth=2)
        ax12.set_ylabel('USD')
        ax12.set_title('Cumulative Preferred Share Dividends')
        if cumulative_dividends.max() > 1e6:
            ax12.yaxis.set_major_formatter(millions_formatter)

    # Market Cap Ratio plot moved up to position 6
    if enable_preferred_shares:
        ax13 = fig.add_subplot(gs[6, 1])  # Changed from gs[7, 1] to gs[6, 1]
        
        # Calculate cumulative dividends and ratio, scale to hundreds
        cumulative_dividends = simulation['quarterly_dividend'].fillna(0).cumsum()
        valid_dates = cumulative_dividends > 0
        ratio = simulation.loc[valid_dates, 'market_cap'] / cumulative_dividends[valid_dates] / 100  # Scale to hundreds
        
        ax13.plot(simulation.index[valid_dates], ratio, 'purple', 
                label='Market Cap / Dividends (100x)', linewidth=2)
        ax13.set_ylabel('Ratio (Hundreds)') 
        ax13.set_title('Market Cap to Cumulative Dividends Ratio')
        ax13.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Set y-axis limits based on data with 5% padding 
        max_ratio = ratio.max()
        ax13.set_ylim(0, max_ratio * 1.05)

    # Add new Volume Percentage plot
    ax14 = fig.add_subplot(gs[5, 1])  # Place in preferred shares position
    volume_pct = (simulation['volume'] / simulation['shares_outstanding']) * 100
    ax14.plot(simulation.index, volume_pct, 'r-', 
            label='Volume % of Shares', linewidth=2)
    ax14.set_ylabel('Volume %')
    ax14.set_title('Daily Volume as % of Shares Outstanding')
    ax14.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax14.set_ylim(0, volume_pct.max() * 1.05)

    # Add Quarterly Dividend Payments plot
    if enable_preferred_shares:
        ax15 = fig.add_subplot(gs[7, 1])  # Place in right column
        quarterly_dividends = simulation['quarterly_dividend']
        ax15.bar(simulation.index[quarterly_dividends > 0], 
                 quarterly_dividends[quarterly_dividends > 0], 
                 width=30, color='purple',
                 label='Quarterly Dividend')
        ax15.set_ylabel('USD')
        ax15.set_title('Quarterly Dividend Payments')
        if quarterly_dividends.max() > 1e6:
            ax15.yaxis.set_major_formatter(millions_formatter)

    # Common settings for all plots
    base_plots = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax10b, ax14]
    if enable_preferred_shares:
        all_plots = base_plots + [ax11, ax12, ax13, ax15]
    else:
        all_plots = base_plots
    
    for ax in all_plots:
        ax.grid(True)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xlim(start_date, end_date)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_three_year_results(simulation, enable_preferred_shares=True):
    """Plot simulation results limited to first 3 years"""
    # Create subset of data ending in 2028
    end_2028 = pd.Timestamp('2028-12-31')
    early_data = simulation[simulation.index <= end_2028].copy()
    
    # Use existing plot_simulation_results logic but with early_data
    fig = plot_simulation_results(early_data, enable_preferred_shares)
    fig.suptitle('SWC 3-Year Estimate (2025-2028)', y=1.02, fontsize=16)
    return fig

def run_complete_simulation(start_date="2025-09-04", end_date="2026-12-31", initial_shares=274683205, initial_btc=2440, enable_preferred_shares=True):
    """
    Runs complete simulation and generates visualizations
    Default values:
    - start_date: Current date (Sept 4th 2025) 
    - end_date: December 31st 2027
    - 593.21M shares (current shares outstanding)
    - 7800 BTC (approximate current holdings)
    - enable_preferred_shares: True to include preferred shares simulation
    """
    if start_date is None:
        start_date = "2025-09-04"
    
    print("Starting simulation...")
    
    # Get historical data
    btc_data = get_bitcoin_historical_data()
    swc_data = get_swc_data()
    btc_holdings, _ = get_bitcoin_holdings()
    
    # Run simulation and get merged data
    simulation = simulate_through_2040(
        btc_data, swc_data, initial_shares, btc_holdings, 
        start_date=start_date, end_date=end_date, enable_preferred_shares=enable_preferred_shares
    )
    
    # Create and save both plot sets
    print("Generating plots...")
    
    # Full simulation plots
    fig_full = plot_simulation_results(simulation, enable_preferred_shares)
    fig_full.suptitle('swc Full Simulation (2025-2027)', y=1.02, fontsize=16)
    fig_full.savefig('swc_simulation_2027.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Three year estimate plots
    fig_early = plot_three_year_results(simulation, enable_preferred_shares)
    fig_early.savefig('swc_simulation_2027.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    simulation.to_csv('swc_simulation_2027.csv')
    
    # Generate and save yearly metrics
    yearly_metrics = generate_yearly_metrics(simulation)
    yearly_metrics.to_csv('swc_yearly_metrics_2027.csv')

    # Create formatted text output
    with open('swc_yearly_summary_2027.txt', 'w') as f:
        f.write("SWC Yearly Key Metrics\n")
        f.write("============================\n\n")
        for year in yearly_metrics.index:
            f.write(f"Year: {year}\n")
            f.write(f"BTC Price: ${yearly_metrics.loc[year, 'BTC Price']:,.2f}\n")
            f.write(f"Stock Price: ${yearly_metrics.loc[year, 'Stock Price']:,.2f}\n")
            f.write(f"BTC Holdings: {yearly_metrics.loc[year, 'BTC Holdings']:,.2f}\n")
            f.write(f"Shares Outstanding: {yearly_metrics.loc[year, 'Shares Outstanding']:,.0f}\n")
            f.write(f"mNAV: {yearly_metrics.loc[year, 'mNAV']:,.4f}\n")
            f.write(f"Market Cap: ${yearly_metrics.loc[year, 'Market Cap (USD)']:,.2f}\n")
            f.write(f"BTC per 1000 Shares: {yearly_metrics.loc[year, 'BTC per 1000 Shares']:,.4f}\n")
            if 'Market Cap to Dividends Ratio' in yearly_metrics.columns and not pd.isna(yearly_metrics.loc[year, 'Market Cap to Dividends Ratio']):
                f.write(f"Market Cap to Dividends Ratio: {yearly_metrics.loc[year, 'Market Cap to Dividends Ratio']:,.1f}x\n")
            f.write("----------------------------\n\n")
    
    # Print summary statistics
    print("\nSimulation Results for 2027:")
    print(f"Final Bitcoin Price: ${simulation['btc_price'].iloc[-1]:,.2f}")
    print(f"Final Stock Price: ${simulation['stock_price'].iloc[-1]:,.2f}")
    print(f"Final BTC Holdings: {simulation['btc_holdings'].iloc[-1]:,.2f} BTC")
    print(f"Final Shares Outstanding: {simulation['shares_outstanding'].iloc[-1]:,.0f}")
    print(f"Final mNAV: {simulation['mnav'].iloc[-1]:,.4f}")
    
    return simulation

def main(start_date="2025-09-04", end_date="2028-12-31", initial_shares=274683205, initial_btc=2440, enable_preferred_shares=True):
    """
    Main function to run the complete swc analysis and simulation
    Args:
        start_date (str): Simulation start date (YYYY-MM-DD)
        end_date (str): Simulation end date (YYYY-MM-DD)
        initial_shares (int): Initial number of outstanding shares
        initial_btc (float): Initial BTC holdings
        enable_preferred_shares (bool): Whether to include preferred shares simulation
    """
    print("Starting swc analysis...")
    
    # Run simulation
    print("Running simulation through 2027...")
    simulation = run_complete_simulation(
        start_date=start_date, end_date=end_date, 
        initial_shares=initial_shares, initial_btc=initial_btc,
        enable_preferred_shares=enable_preferred_shares
    )
    
    return simulation

if __name__ == "__main__":
    # Example usage: provide your own start_date, initial_shares, and initial_btc
    # To disable preferred shares simulation:
    simulation_results = run_complete_simulation("2025-09-04", "2027-12-31", 274683205, 2440, enable_preferred_shares=True)
    # 
    # To enable preferred shares simulation (default):
    # simulation_results = run_complete_simulation("2025-09-04", "2027-12-31", 274683205, 2440, enable_preferred_shares=True)
    
    # simulation_results = run_complete_simulation()

