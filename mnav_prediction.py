import numpy as np
import pandas as pd
from datetime import datetime

def calculate_mnav_with_volatility(btc_value, days_from_start, base_volatility=0.12, end_date=None, current_date=None, dilution_rate_pct=0.0):
    """
    Calculates mNAV tied to Bitcoin price with specific distribution:
    - 10% of time: minimum mNAV of 0.5
    - 40% of time: intermediate mNAV of 0.5 to 0.9
    - 50% of time: higher mNAV of 0.9 to 2.0
    
    The mNAV levels correlate with Bitcoin price momentum and cycles.
    Returns: Float with calculated mNAV value
    """
    
    # Get Bitcoin price momentum by looking at price relative to a moving average concept
    # Use days_from_start as a proxy for price cycle position
    
    # Create a normalized cycle that correlates with Bitcoin price movements
    # This represents Bitcoin's position in its bull/bear cycle
    btc_cycle_position = np.sin(days_from_start * 2 * np.pi / 1460)  # 4-year cycle
    
    # Add shorter-term volatility cycles
    short_cycle = np.sin(days_from_start * 2 * np.pi / 90)   # 3-month cycle
    medium_cycle = np.sin(days_from_start * 2 * np.pi / 180) # 6-month cycle
    
    # Combine cycles with weights
    combined_momentum = (0.6 * btc_cycle_position + 0.3 * medium_cycle + 0.1 * short_cycle)
    
    # Normalize to [0, 1] range
    normalized_momentum = (combined_momentum + 1) / 2
    
    # Add some randomness to prevent perfect correlation
    random_factor = np.random.normal(0, 0.15)  # 15% random variation
    adjusted_momentum = np.clip(normalized_momentum + random_factor, 0, 1)
    
    # Map the momentum to the three mNAV zones based on desired distribution
    random_selector = np.random.random()
    
    if random_selector < 0.10:  # 10% of time: minimum mNAV of 0.5
        base_mnav = 0.5
        
    elif random_selector < 0.50:  # 40% of time: intermediate mNAV (0.5 to 0.9)
        # Use momentum to determine position within range
        base_mnav = 0.5 + (adjusted_momentum * 0.4)  # Maps to 0.5-0.9
        
    else:  # 50% of time: higher mNAV (0.9 to 2.0)
        # Use momentum to determine position within range, with bias toward lower end
        # Apply power function to bias toward 0.9 end
        biased_momentum = adjusted_momentum ** 0.7  # Bias toward lower values
        base_mnav = 0.9 + (biased_momentum * 1.1)  # Maps to 0.9-2.0
    
    # Apply smoothing to prevent sharp jumps between days
    if hasattr(calculate_mnav_with_volatility, 'last_mnav'):
        prev_mnav = calculate_mnav_with_volatility.last_mnav
        smoothing_factor = 0.05  # 5% move toward target each day
        base_mnav = prev_mnav + (base_mnav - prev_mnav) * smoothing_factor
    
    # Add small daily volatility
    daily_noise = np.random.normal(0, base_volatility * 0.2)
    new_mnav = base_mnav + daily_noise
    
    # Apply dilution dampening: For each 1% dilution, dampen mNAV by 1%
    dilution_dampening_factor = 1.0 - (dilution_rate_pct / 100.0)
    dilution_dampening_factor = max(0.1, dilution_dampening_factor)  # Minimum 10% of original
    new_mnav *= dilution_dampening_factor
    
    # Ensure final bounds
    new_mnav = max(0.1, min(3.0, new_mnav))  # Absolute bounds 0.1 to 3.0
    
    # Store for next calculation
    calculate_mnav_with_volatility.last_mnav = new_mnav
    
    return new_mnav
