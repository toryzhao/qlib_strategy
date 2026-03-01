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
