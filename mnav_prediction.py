import numpy as np
import pandas as pd
from datetime import datetime

def calculate_mnav_with_volatility(btc_value, days_from_start, base_volatility=0.12, end_date=None, current_date=None, dilution_rate_pct=0.0):
    """
    Calculates mNAV with oscillating cycles and decaying power law peaks
    mNAV is dampened by dilution rate: 1% dilution = 1% mNAV reduction
    Returns: Float with calculated mNAV value including volatility
    """
    # Create oscillating cycles with varying periods (30-90 days)
    # Use multiple sine waves with different frequencies and phases for complexity
    
    # Primary cycle: 180-day period (6 months)
    primary_cycle = np.sin(days_from_start * 2 * np.pi / 180)

    # Secondary cycle: 45-day period (1.5 months)
    secondary_cycle = np.sin(days_from_start * 2 * np.pi / 45 + np.pi/3)
    
    # Tertiary cycle: 360-day period (12 months)
    tertiary_cycle = np.sin(days_from_start * 2 * np.pi / 360 + np.pi/6)

    # Combine cycles with different weights
    combined_cycle = (0.5 * primary_cycle + 0.3 * secondary_cycle + 0.2 * tertiary_cycle)
    
    # Apply decaying power law to the maximum mNAV peaks
    # Power law decay: max_mnav = initial_max * (days_from_start + 1)^(-decay_exponent)
    initial_max_mnav = 7.0  # Starting maximum mNAV
    decay_exponent = 0.5   # Controls how fast peaks decay (higher = faster decay)
    time_factor = (days_from_start / 365.25) + 1  # Convert to years, add 1 to avoid zero
    
    # Calculate current maximum mNAV using power law decay
    current_max_mnav = initial_max_mnav * (time_factor ** (-decay_exponent))
    # Ensure minimum peak doesn't go below reasonable levels
    current_max_mnav = max(2.5, current_max_mnav)
    
    # Map the combined cycle (-1 to 1) to mNAV range with more downside bias
    # Normalize combined_cycle to [0, 1] range first
    normalized_cycle = (combined_cycle + 1) / 2
    
    # Apply power function to bias toward lower values (more downside)
    # Power > 1 creates more time spent at lower values
    biased_cycle = normalized_cycle ** 2.5  # Strong bias toward lower values
    
    # Map to 3-zone asymmetric range with decaying peaks
    if biased_cycle < 0.05:  # 5% of time in discount range (0.8 to 1.3)
        base_mnav = 0.8 + (biased_cycle / 0.05) * 0.5  # Maps to 0.8-1.3
    elif biased_cycle < 0.35:  # 30% of time in moderate premium range (1.3 to 2.5)
        base_mnav = 1.3 + ((biased_cycle - 0.05) / 0.30) * 1.2  # Maps to 1.3-2.5
    else:  # 65% of time in high premium range (2.5 to current_max_mnav)
        peak_range = current_max_mnav - 2.5
        base_mnav = 2.5 + ((biased_cycle - 0.35) / 0.65) * peak_range  # Maps to 2.5-current_max_mnav

    # No level shift needed - ranges already start at 1.3
    # base_mnav is already in the correct range
    
    # Add some random volatility (much smaller than the cycle)
    volatility = base_volatility * 0.3  # Reduced volatility to keep cycles clear
    noise = np.random.normal(0, volatility)
    
    # Get previous mNAV for smoothing (or use base if first calculation)
    current_mnav = getattr(calculate_mnav_with_volatility, 'last_mnav', base_mnav)
    
    # Apply slight smoothing to prevent sharp jumps
    smoothing_factor = 0.1  # How much to move toward target each day
    new_mnav = current_mnav + (base_mnav - current_mnav) * smoothing_factor + noise
    
    # Apply dilution dampening: For each 1% dilution, dampen mNAV by 5%
    dilution_dampening_factor = 1.0 - (dilution_rate_pct / 20.0)  # Divide by 20 for 5x effect
    dilution_dampening_factor = max(0.2, dilution_dampening_factor)  # Minimum 20% dampening
    new_mnav *= dilution_dampening_factor
    
    # Ensure bounds are respected - range 0.8 to current_max_mnav (decaying)
    new_mnav = max(0.8, min(current_max_mnav, new_mnav))
    
    # Store for next calculation
    calculate_mnav_with_volatility.last_mnav = new_mnav
    
    return new_mnav
