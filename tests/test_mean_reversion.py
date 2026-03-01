"""
Tests for MeanReversionStrategy
"""
import pytest
import pandas as pd
import numpy as np
from strategies.statistical.mean_reversion import MeanReversionStrategy


def test_calculate_zscore_basic():
    """Test Z-Score calculation with simple data"""
    # Create data where mean=100, std should be calculable
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })

    config = {'lookback_period': 5}
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    zscore = strategy.calculate_zscore(data)

    # Verify output
    assert len(zscore) == len(data)
    assert zscore.dtype == np.float64
    # First 4 values should be NaN (insufficient data for 5-period window)
    assert pd.isna(zscore.iloc[0])
    assert pd.isna(zscore.iloc[3])
    # 5th value should be valid
    assert not pd.isna(zscore.iloc[4])


def test_get_position_size_multi_layer():
    """Test multi-layer position sizing based on Z-Score"""
    config = {
        'level1_threshold': 1.5,
        'level2_threshold': 2.0,
        'level3_threshold': 2.5
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Test different Z-Score levels
    assert strategy.get_position_size(0) == 0.0  # No deviation
    assert strategy.get_position_size(1.0) == 0.0  # Below threshold
    assert strategy.get_position_size(1.5) == 0.5  # Level 1
    assert strategy.get_position_size(1.8) == 0.5  # Level 1
    assert strategy.get_position_size(2.0) == 0.75  # Level 2
    assert strategy.get_position_size(2.3) == 0.75  # Level 2
    assert strategy.get_position_size(2.5) == 1.0  # Level 3
    assert strategy.get_position_size(3.0) == 1.0  # Level 3

    # Test short positions (negative Z-Score)
    assert strategy.get_position_size(-1.5) == 0.5
    assert strategy.get_position_size(-2.0) == 0.75
    assert strategy.get_position_size(-2.5) == 1.0


def test_should_exit_reversion():
    """Test exit when Z-Score reverts to mean"""
    config = {
        'exit_threshold': 0.5,
        'stop_multiplier': 1.5,
        'max_hold_period': 50
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Entered at Z-Score = 2.0
    # Should exit when Z-Score returns to ±0.5
    assert strategy.should_exit(current_zscore=0.3, entry_zscore=2.0, bars_held=10) == True
    assert strategy.should_exit(current_zscore=-0.3, entry_zscore=-2.0, bars_held=10) == True

    # Should NOT exit when still far from mean
    assert strategy.should_exit(current_zscore=1.5, entry_zscore=2.0, bars_held=10) == False


def test_should_exit_stop_loss():
    """Test exit when stop loss is hit"""
    config = {'stop_multiplier': 1.5}
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Entered at Z-Score = 2.0, stop at 2.0 × 1.5 = 3.0
    assert strategy.should_exit(current_zscore=3.1, entry_zscore=2.0, bars_held=10) == True
    assert strategy.should_exit(current_zscore=2.9, entry_zscore=2.0, bars_held=10) == False


def test_should_exit_max_holding_period():
    """Test exit when max holding period exceeded"""
    config = {'max_hold_period': 50}
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Should exit after 50 bars regardless of Z-Score
    assert strategy.should_exit(current_zscore=1.0, entry_zscore=2.0, bars_held=51) == True
    assert strategy.should_exit(current_zscore=1.0, entry_zscore=2.0, bars_held=50) == False


def test_generate_signals_basic():
    """Test signal generation with oscillating price"""
    # Create price data that oscillates
    data = pd.DataFrame({
        'close': [100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100]
    })

    config = {
        'lookback_period': 5,
        'entry_threshold': 1.0  # Lower threshold for test data
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    signals = strategy.generate_signals(data)

    # Verify output format
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert 'target_position' in signals.columns
    assert len(signals) == len(data)

    # Verify signal values are -1, 0, or 1
    assert signals['signal'].isin([-1, 0, 1]).all()

    # Verify position sizes are 0.0, 0.5, 0.75, or 1.0
    assert signals['target_position'].isin([0.0, 0.5, 0.75, 1.0]).all()


def test_generate_signals_with_extreme_zscore():
    """Test signal generation generates signals with extreme Z-Score"""
    # Create data with extreme deviation
    data = pd.DataFrame({
        'close': [100, 100, 100, 100, 100, 100, 100, 110]  # Last bar is extreme
    })

    config = {
        'lookback_period': 5,
        'level1_threshold': 1.0  # Low threshold to trigger signal
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    signals = strategy.generate_signals(data)

    # The last bar should have a signal (extreme price deviation)
    # Z-Score will be high since 110 is far from mean of 100
    assert signals['signal'].iloc[-1] != 0 or signals['target_position'].iloc[-1] > 0


def test_volatility_adjustment():
    """Test position size adjustment for volatility"""
    config = {
        'use_volatility_adjustment': True,
        'atr_period': 14,
        'atr_lookback': 100,
        'volatility_threshold': 80
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Create data with high, low, close
    data = pd.DataFrame({
        'close': [100] * 150,
        'high': [101] * 150,
        'low': [99] * 150
    })

    # Get position with volatility adjustment
    base_position = 0.5
    adjusted_position = strategy.get_position_with_volatility(
        data, current_bar=149, base_position=base_position
    )

    # Should return a float between 0 and base_position
    assert isinstance(adjusted_position, float)
    assert 0 <= adjusted_position <= base_position
