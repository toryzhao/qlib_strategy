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
