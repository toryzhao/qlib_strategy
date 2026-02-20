import pytest
import pandas as pd
import numpy as np
from strategies.risk.risk_manager import RiskManager

def test_risk_manager_initialization():
    """Test RiskManager initialization with config"""
    config = {
        'trend_ma_period': 200,
        'atr_period': 14,
        'atr_lookback': 100,
        'volatility_threshold': 80,
        'swing_period': 20
    }
    rm = RiskManager(config)
    assert rm.trend_ma_period == 200
    assert rm.atr_period == 14
    assert rm.atr_lookback == 100
    assert rm.volatility_threshold == 80
    assert rm.swing_period == 20

def test_calculate_atr():
    """Test ATR calculation"""
    config = {'atr_period': 14}
    rm = RiskManager(config)

    # Create sample data with known ATR
    data = pd.DataFrame({
        'high': [10, 11, 12, 13, 14],
        'low': [9, 10, 11, 12, 13],
        'close': [9.5, 10.5, 11.5, 12.5, 13.5]
    })

    atr = rm._calculate_atr(data)
    assert len(atr) == len(data)
    assert atr.iloc[0] > 0  # First value is positive (ewm starts from first value)
    assert not pd.isna(atr.iloc[-1])  # Last value is not NaN
    assert atr.iloc[-1] > 0  # ATR should be positive

def test_calculate_trend_filter_uptrend():
    """Test trend filter in uptrend"""
    config = {'trend_ma_period': 5}
    rm = RiskManager(config)

    # Price above MA = uptrend
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })

    trend = rm.calculate_trend_filter(data)
    assert trend.iloc[-1] == 1  # Uptrend signal

def test_calculate_trend_filter_downtrend():
    """Test trend filter in downtrend"""
    config = {'trend_ma_period': 5}
    rm = RiskManager(config)

    # Price below MA = downtrend
    data = pd.DataFrame({
        'close': [110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
    })

    trend = rm.calculate_trend_filter(data)
    assert trend.iloc[-1] == -1  # Downtrend signal

def test_calculate_trend_filter_insufficient_data():
    """Test trend filter with insufficient data"""
    config = {'trend_ma_period': 200}
    rm = RiskManager(config)

    # Only 50 bars, need 200
    data = pd.DataFrame({
        'close': [100 + i for i in range(50)]
    })

    trend = rm.calculate_trend_filter(data)
    # Should return 0 (no trend) when insufficient data
    assert trend.iloc[-1] == 0
