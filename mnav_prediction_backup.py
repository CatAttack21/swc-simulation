import numpy as np
import pandas as pd
from datetime import datetime

def calculate_mnav_with_volatility(btc_value, days_from_start, base_volatility=0.12, end_date=None, current_date=None, dilution_rate_pct=0.0):
    """
    Calculates mNAV tied to Bitcoin price with specific distribution:
    - mNAV correlates with Bitcoin price momentum and market phase
    - Lower mNAV during bear markets, higher during bull markets
    - 8% of time: deep discount mNAV of 0.5
    - 17% of time: low discount mNAV around 0.75 (0.65-0.85 range)
    - 30% of time: moderate discount mNAV (0.75 to 0.9)
    - 45% of time: higher mNAV (0.9 to 2.0)
    
    Returns: Float with calculated mNAV value
    """
    
    # Determine Bitcoin market phase based on current date
    if current_date is not None:
        current_ts = pd.Timestamp(current_date)
        
        # Define Bitcoin market phases based on the cycle
        if current_ts < pd.Timestamp('2025-11-01'):
            # Early bull market phase (building up)
            market_phase = "early_bull"
            phase_strength = 0.7
        elif current_ts < pd.Timestamp('2026-02-01'):
            # Blow-off top phase (peak euphoria around 150k-200k)
            market_phase = "blowoff_top"
            # Calculate position within blow-off top (Nov 2025 - Jan 2026)
            blowoff_days = (current_ts - pd.Timestamp('2025-11-01')).days
            blowoff_progress = min(1.0, blowoff_days / 92)  # 3 months
            phase_strength = 0.95 + (blowoff_progress * 0.05)  # 0.95 to 1.0
        elif current_ts < pd.Timestamp('2027-07-01'):
            # Bear market phase (falling from peak to 75k)
            market_phase = "bear"
            # Calculate how deep into bear market - correlate with Bitcoin price decline
            bear_start = pd.Timestamp('2026-02-01')
            bear_duration = (pd.Timestamp('2027-07-01') - bear_start).days
            bear_progress = (current_ts - bear_start).days / bear_duration
            # As Bitcoin falls from 150k to 75k, mNAV should fall proportionally
            phase_strength = max(0.2, 0.8 - (bear_progress * 0.6))  # 0.8 to 0.2
        elif current_ts < pd.Timestamp('2028-01-01'):
            # Recovery phase (75k to 80k)
            market_phase = "recovery"
            phase_strength = 0.4
        else:
            # New bull market (80k to 300k)
            market_phase = "bull"
            phase_strength = 0.9
    else:
        # Fallback to time-based cycle
        cycle_position = (days_from_start % 1460) / 1460
        if cycle_position < 0.1:
            market_phase = "blowoff_top"
            phase_strength = 1.0
        elif cycle_position < 0.25:
            market_phase = "early_bull"
            phase_strength = 0.8
        elif cycle_position < 0.75:
            market_phase = "bear"
            phase_strength = 0.5
        else:
            market_phase = "recovery"
            phase_strength = 0.4
    
    # Create short-term volatility cycles
    short_cycle = np.sin(days_from_start * 2 * np.pi / 90)   # 3-month cycle
    medium_cycle = np.sin(days_from_start * 2 * np.pi / 180) # 6-month cycle
    
    # Combine cycles with market phase influence
    combined_momentum = (0.4 * phase_strength + 0.4 * medium_cycle + 0.2 * short_cycle)
    
    # Normalize to [0, 1] range
    normalized_momentum = (combined_momentum + 1) / 2
    
    # Add randomness, but less during bear markets (more predictable discounts)
    if market_phase == "bear":
        random_factor = np.random.normal(0, 0.10)  # Reduced randomness in bear
    else:
        random_factor = np.random.normal(0, 0.15)  # Normal randomness
    
    adjusted_momentum = np.clip(normalized_momentum + random_factor, 0, 1)
    
    # Map momentum to mNAV zones with market phase bias
    random_selector = np.random.random()
    
    # Adjust probability thresholds based on market phase
    if market_phase == "blowoff_top":
        # Blow-off top: extreme euphoria, mNAV can reach 5.0
        deep_threshold = 0.02   # 2% deep discount (rare during euphoria)
        low_threshold = 0.05    # 3% low discount 
        mod_threshold = 0.15    # 10% moderate discount
        high_threshold = 0.60   # 45% normal high (0.9-2.0)
        # 40% extreme euphoria (2.0-5.0)
        
    elif market_phase == "early_bull":
        # Early bull: building momentum
        deep_threshold = 0.08
        low_threshold = 0.20
        mod_threshold = 0.40
        high_threshold = 0.85   # 45% normal high
        # 15% elevated euphoria (2.0-3.0)
        
    elif market_phase == "bear":
        # Bear market: spend more time in discount zones
        deep_threshold = 0.15   # 15% deep discount
        low_threshold = 0.35    # 20% low discount (0.75 area)
        mod_threshold = 0.70    # 35% moderate discount
        high_threshold = 1.0    # 30% higher mNAV (but capped lower)
        
    elif market_phase == "recovery":
        # Recovery: moderate distribution
        deep_threshold = 0.10
        low_threshold = 0.25
        mod_threshold = 0.55
        high_threshold = 0.85
        
    else:  # regular bull market
        # Bull market: spend more time in higher mNAV zones
        deep_threshold = 0.05
        low_threshold = 0.15
        mod_threshold = 0.35
        high_threshold = 0.80
    
    if random_selector < deep_threshold:
        # Deep discount
        base_mnav = 0.5
        
    elif random_selector < low_threshold:
        # Low discount around 0.75
        base_mnav = 0.65 + (adjusted_momentum * 0.2)  # Maps to 0.65-0.85
        
    elif random_selector < mod_threshold:
        # Moderate discount
        base_mnav = 0.75 + (adjusted_momentum * 0.15)  # Maps to 0.75-0.9
        
    elif random_selector < high_threshold:
        # Normal higher mNAV
        if market_phase == "bear":
            # Conservative during bear markets
            biased_momentum = adjusted_momentum ** 1.2
            base_mnav = 0.9 + (biased_momentum * 0.6)  # Maps to 0.9-1.5
        else:
            # Normal range
            biased_momentum = adjusted_momentum ** 0.7
            base_mnav = 0.9 + (biased_momentum * 1.1)  # Maps to 0.9-2.0
            
    else:
        # Extreme euphoria phase (only during blow-off top and strong bulls)
        if market_phase == "blowoff_top":
            # Peak euphoria: mNAV can reach 5.0
            euphoria_momentum = adjusted_momentum ** 0.5  # Bias toward higher values
            base_mnav = 2.0 + (euphoria_momentum * 3.0)  # Maps to 2.0-5.0
        elif market_phase == "early_bull":
            # Elevated but not extreme
            euphoria_momentum = adjusted_momentum ** 0.6
            base_mnav = 2.0 + (euphoria_momentum * 1.0)  # Maps to 2.0-3.0
        else:
            # Other phases: cap at normal high range
            biased_momentum = adjusted_momentum ** 0.7
            base_mnav = 0.9 + (biased_momentum * 1.1)  # Maps to 0.9-2.0
    
    if random_selector < 0.08:  # 8% of time: deep discount mNAV of 0.5
        base_mnav = 0.5
        
    elif random_selector < 0.25:  # 17% of time: low discount mNAV around 0.75
        # Use momentum to vary around 0.75 with some spread
        base_mnav = 0.65 + (adjusted_momentum * 0.2)  # Maps to 0.65-0.85, centered on 0.75
        
    elif random_selector < 0.55:  # 30% of time: moderate discount mNAV (0.75 to 0.9)
        # Use momentum to determine position within range
        base_mnav = 0.75 + (adjusted_momentum * 0.15)  # Maps to 0.75-0.9
        
    else:  # 45% of time: higher mNAV (0.9 to 2.0)
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
    
    # Ensure final bounds - allow higher during blow-off top
    if hasattr(calculate_mnav_with_volatility, 'last_phase'):
        last_phase = calculate_mnav_with_volatility.last_phase
    else:
        last_phase = market_phase if 'market_phase' in locals() else "bull"
    
    if last_phase == "blowoff_top":
        new_mnav = max(0.1, min(5.5, new_mnav))  # Allow up to 5.5 during blow-off top
    else:
        new_mnav = max(0.1, min(3.0, new_mnav))  # Normal bounds 0.1 to 3.0
    
    # Store phase and value for next calculation
    calculate_mnav_with_volatility.last_mnav = new_mnav
    if 'market_phase' in locals():
        calculate_mnav_with_volatility.last_phase = market_phase
    
    return new_mnav
