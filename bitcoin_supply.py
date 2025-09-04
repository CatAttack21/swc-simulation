import numpy as np
import pandas as pd
from datetime import datetime

MAX_BITCOIN = 21_000_000
LOST_BITCOIN = 4_000_000
TRADEABLE_SUPPLY = MAX_BITCOIN - LOST_BITCOIN

# Target treasury holdings (% of total supply)
TREASURY_TARGETS = {
    '2028': 0.25,  # 25% by 2028
    '2030': 0.50   # 50% by 2030
}

INITIAL_COMPETITOR_HOLDINGS = {
    'MicroStrategy': 193000,
    'Grayscale': 350000,
    'Tesla': 25000,
    'Other_Corps': 150000,
    'Nation_States': 200000
}

def get_target_treasury_holdings(date):
    """Calculate target treasury holdings for given date"""
    date = pd.Timestamp(date)
    
    if date.year >= 2030:
        return TREASURY_TARGETS['2030'] * MAX_BITCOIN
    elif date.year >= 2028:
        return TREASURY_TARGETS['2028'] * MAX_BITCOIN
    else:
        # Linear interpolation between now and 2028
        total_days = (pd.Timestamp('2028-01-01') - pd.Timestamp('2025-05-24')).days
        days_passed = (date - pd.Timestamp('2025-05-24')).days
        progress = days_passed / total_days
        return (TREASURY_TARGETS['2028'] * MAX_BITCOIN * progress) + sum(INITIAL_COMPETITOR_HOLDINGS.values())

def calculate_remaining_bitcoin(current_date):
    """Calculate remaining mineable Bitcoin at given date"""
    halving_interval = 210000  # blocks
    blocks_per_day = 144
    days_since_genesis = (current_date - pd.Timestamp('2009-01-03')).days
    total_blocks = days_since_genesis * blocks_per_day
    halvings = total_blocks // halving_interval
    current_block_reward = 50 / (2 ** halvings)
    
    total_mined = 21_000_000 - (current_block_reward * blocks_per_day * 365.25 / 4)
    return max(0, MAX_BITCOIN - total_mined)

def simulate_competitor_purchases(date, available_btc, current_price):
    """Simulate competitive buying behavior"""
    # Base purchase probability increases as price increases (momentum buying)
    price_momentum = np.clip(current_price / 100000, 0.1, 1.0)
    
    # Competition intensity increases as available supply decreases
    supply_pressure = 1 - (available_btc / MAX_BITCOIN)
    
    # Calculate purchase probabilities for each competitor
    purchase_probs = {
        'MicroStrategy': 0.3 * price_momentum * supply_pressure,
        'Grayscale': 0.25 * price_momentum * supply_pressure,
        'Tesla': 0.1 * price_momentum * supply_pressure,
        'Other_Corps': 0.2 * price_momentum * supply_pressure,
        'Nation_States': 0.15 * price_momentum * supply_pressure
    }
    
    total_purchased = 0
    purchases = {}
    
    for competitor, prob in purchase_probs.items():
        if np.random.random() < prob:
            # Purchase size increases with price and scarcity
            max_purchase = min(1000, available_btc * 0.1)
            purchase_amount = np.random.exponential(max_purchase/3)
            purchase_amount = min(purchase_amount, available_btc - total_purchased)
            
            if purchase_amount > 0:
                purchases[competitor] = purchase_amount
                total_purchased += purchase_amount
                
            if total_purchased >= available_btc:
                break
    
    return purchases, total_purchased

def calculate_supply_shock(total_btc, recent_purchases, price):
    """Calculate price impact of supply shock"""
    # Calculate effective tradeable supply
    effective_supply = TRADEABLE_SUPPLY - total_btc
    
    # Impact increases with % of tradeable supply purchased
    recent_purchase_ratio = recent_purchases / effective_supply if effective_supply > 0 else 1
    price_level_factor = np.log10(max(price, 1000)) / 5
    
    # Enhanced supply shock as tradeable supply decreases
    supply_scarcity = 1 + ((TRADEABLE_SUPPLY - effective_supply) / TRADEABLE_SUPPLY) * 2
    
    # Calculate combined multiplier
    shock_multiplier = 1 + (recent_purchase_ratio * price_level_factor * supply_scarcity)
    
    # Additional premium as tradeable supply approaches zero
    scarcity_premium = np.exp((total_btc / TRADEABLE_SUPPLY - 0.8) * 5) 
    
    return shock_multiplier * scarcity_premium

def get_effective_float():
    """Returns current tradeable Bitcoin supply"""
    return TRADEABLE_SUPPLY
