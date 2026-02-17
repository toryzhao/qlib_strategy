# trading/tests/test_feature_engineering.py
import pytest
import pandas as pd
import numpy as np
from utils.feature_engineering import FeatureEngineer

def test_add_technical_features():
    """Test adding technical indicators"""
    df = pd.DataFrame({
        'datetime': pd.date_range('2020-01-01', periods=100, freq='1min'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 5000, 100)
    })

    result = FeatureEngineer.add_technical_features(df)

    # Check moving averages
    assert 'MA5' in result.columns
    assert 'MA20' in result.columns
    assert 'EMA12' in result.columns

    # Check MACD
    assert 'MACD' in result.columns
    assert 'MACD_signal' in result.columns
    assert 'MACD_hist' in result.columns

    # Check Bollinger Bands
    assert 'BOLL_upper' in result.columns
    assert 'BOLL_lower' in result.columns
    assert 'BOLL_mid' in result.columns

    # Check RSI
    assert 'RSI' in result.columns

    # Check ATR
    assert 'ATR' in result.columns

def test_add_time_features():
    """Test adding time-based features"""
    df = pd.DataFrame({
        'datetime': pd.date_range('2020-01-01 09:00:00', periods=100, freq='1min')
    })

    result = FeatureEngineer.add_time_features(df)

    assert 'hour' in result.columns
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    assert 'quarter' in result.columns

    # Check values
    assert result['hour'].iloc[0] == 9
    assert result['day_of_week'].iloc[0] in range(7)  # Wednesday
