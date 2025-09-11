import numpy as np
import pandas as pd
from datetime import datetime

def calculate_mnav_with_volatility(btc_value, days_from_start, base_volatility=0.12, end_date=None, current_date=None, dilution_rate_pct=0.0):
    """
    Calculates mNAV highly correlated to Bitcoin price in 2 phases:
    
    Phase 1 - Bull Run: mNAV rises from 2 to 5 during Bitcoin bull market to blow-off top
    Phase 2 - Bear Cycle: mNAV falls from 5 to 1, then ranges 0.8-1.3, bottoms at 0.5
    
    Returns: Float with calculated mNAV value
    """
    
    # Determine Bitcoin market phase and position within cycle
    if current_date is not None:
        current_ts = pd.Timestamp(current_date)
        
        # Phase 1: Bull Run (mNAV 2 → 5)
        if current_ts < pd.Timestamp('2026-01-01'):
            market_phase = "bull_run"
            
            # Calculate progress through bull run
            bull_start = pd.Timestamp('2025-05-24')  # Simulation start
            bull_peak = pd.Timestamp('2025-12-31')   # Bitcoin peaks at 150k
            
            if current_ts <= bull_peak:
                # Rising phase: mNAV 2 → 5 with cyclical behavior
                bull_progress = (current_ts - bull_start).days / (bull_peak - bull_start).days
                bull_progress = max(0, min(1, bull_progress))
                
                # Base linear correlation with Bitcoin rise
                base_mnav = 1.0 + (bull_progress * 3.0)  # 2.0 → 5.0
                
                # Add cyclical oscillations around the main trend
                days_since_start = (current_ts - bull_start).days
                
                # Multiple cycles for realistic market behavior
                short_cycle = np.sin(days_since_start * 2 * np.pi / 30)    # 1-month cycle
                medium_cycle = np.sin(days_since_start * 2 * np.pi / 60)   # 2-month cycle
                long_cycle = np.sin(days_since_start * 2 * np.pi / 120)    # 4-month cycle
                
                # Combine cycles with weights (favor shorter cycles for more activity)
                combined_cycle = (0.5 * short_cycle + 0.3 * medium_cycle + 0.2 * long_cycle)
                
                # Apply cyclical variation as percentage of base mNAV
                cycle_amplitude = 0.15 * base_mnav  # ±15% cyclical variation
                cyclical_adjustment = combined_cycle * cycle_amplitude
                
                # Combine base trend with cyclical behavior
                base_mnav_with_cycles = base_mnav + cyclical_adjustment
                
                # Add small random volatility on top of cyclical behavior
                volatility_range = 0.2  # Reduced from 0.3 since cycles add variation
                daily_noise = np.random.normal(0, volatility_range * base_mnav_with_cycles * 0.08)
                target_mnav = base_mnav_with_cycles + daily_noise
                
            else:
                # Brief peak/transition period
                target_mnav = 5.0 + np.random.normal(0, 0.2)  # Stay near peak with small noise
        
        # Phase 2: Bear Cycle (mNAV 5 → 0.5 in 2-3 months, then oscillate 0.8-1.3)
        else:
            market_phase = "bear_cycle"
            
            bear_start = pd.Timestamp('2026-01-01')
            crash_bottom = pd.Timestamp('2026-03-31')  # Quick crash: 3 months to bottom
            oscillation_start = pd.Timestamp('2026-04-01')  # Start oscillating after crash
            
            if current_ts <= crash_bottom:
                # Quick crash phase: mNAV 5 → 0.5 in just 3 months
                crash_progress = (current_ts - bear_start).days / (crash_bottom - bear_start).days
                crash_progress = max(0, min(1, crash_progress))
                
                # Aggressive exponential decay for quick crash
                crash_factor = np.exp(-crash_progress * 4.0)  # More aggressive decay (was 2.5)
                target_mnav = 0.5 + (4.5 * crash_factor)  # 5.0 → 0.5 quickly
                
                # Add high volatility during crash
                volatility_range = 0.5  # ±50% volatility during rapid crash
                daily_noise = np.random.normal(0, volatility_range * target_mnav * 0.2)
                target_mnav = target_mnav + daily_noise
                
            else:
                # Oscillation phase: steady oscillation between 0.8-1.3
                days_since_bottom = (current_ts - oscillation_start).days
                
                # Create smooth oscillation between 0.8-1.3
                # Use multiple cycles for more realistic movement
                primary_cycle = np.sin(days_since_bottom * 2 * np.pi / 120)    # 4-month cycle
                secondary_cycle = np.sin(days_since_bottom * 2 * np.pi / 60)   # 2-month cycle
                tertiary_cycle = np.sin(days_since_bottom * 2 * np.pi / 30)    # 1-month cycle
                
                # Combine cycles with weights
                combined_oscillation = (0.6 * primary_cycle + 0.3 * secondary_cycle + 0.1 * tertiary_cycle)
                
                # Normalize to [0, 1] range
                normalized_osc = (combined_oscillation + 1) / 2
                
                # Map to 0.8-1.3 range with slight bias toward middle
                target_mnav = 0.8 + (normalized_osc * 0.5)  # 0.8 → 1.3
                
                # Add small volatility for realism
                daily_noise = np.random.normal(0, 0.08)  # Reduced volatility during oscillation
                target_mnav = target_mnav + daily_noise
    
    else:
        # Fallback for when current_date is not provided
        cycle_position = (days_from_start % 1095) / 1095  # 3-year cycle
        
        if cycle_position < 0.33:  # Bull phase
            bull_progress = cycle_position / 0.33
            target_mnav = 2.0 + (bull_progress * 3.0)
        else:  # Bear phase
            bear_progress = (cycle_position - 0.33) / 0.67
            if bear_progress < 0.5:
                # Crash phase
                crash_factor = np.exp(-bear_progress * 2 * 2.5)
                target_mnav = 0.5 + (4.5 * crash_factor)
            else:
                # Ranging phase
                target_mnav = 0.8 + (np.random.random() * 0.5)
    
    # Apply smoothing to prevent sharp daily jumps
    if hasattr(calculate_mnav_with_volatility, 'last_mnav'):
        prev_mnav = calculate_mnav_with_volatility.last_mnav
        smoothing_factor = 0.15  # 15% move toward target each day
        new_mnav = prev_mnav + (target_mnav - prev_mnav) * smoothing_factor
    else:
        new_mnav = target_mnav
    
    # Apply dilution dampening: For each 1% dilution, dampen mNAV by 1%
    dilution_dampening_factor = 1.0 - (dilution_rate_pct / 100.0)
    dilution_dampening_factor = max(0.1, dilution_dampening_factor)
    new_mnav *= dilution_dampening_factor
    
    # Ensure reasonable bounds
    new_mnav = max(0.1, min(6.0, new_mnav))
    
    # Store for next calculation
    calculate_mnav_with_volatility.last_mnav = new_mnav
    
    return new_mnav
