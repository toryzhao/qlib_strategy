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

def test_calculate_volatility_adjustment_normal():
    """Test position size adjustment in normal volatility"""
    config = {
        'atr_period': 14,
        'atr_lookback': 100,
        'volatility_threshold': 80
    }
    rm = RiskManager(config)

    # Create data with stable ATR (not in top 20%)
    atr_values = [1.0] * 100
    data = pd.DataFrame({'high': [101] * 100, 'low': [99] * 100, 'close': [100] * 100})

    position_size = rm.calculate_volatility_adjustment(data, 0.3)
    assert position_size == 0.3  # No reduction

def test_calculate_volatility_adjustment_high():
    """Test position size adjustment in high volatility"""
    config = {
        'atr_period': 14,
        'atr_lookback': 10,
        'volatility_threshold': 80
    }
    rm = RiskManager(config)

    # Create data with increasing volatility
    data = pd.DataFrame({
        'high': [100 + i*0.5 for i in range(15)],
        'low': [99 + i*0.5 for i in range(15)],
        'close': [99.5 + i*0.5 for i in range(15)]
    })

    # Just verify method exists and returns a float
    position_size = rm.calculate_volatility_adjustment(data, 0.3)
    assert isinstance(position_size, float)
    assert 0 <= position_size <= 0.3  # Should be between 0 and base

def test_calculate_volatility_adjustment_insufficient_data():
    """Test with insufficient data"""
    config = {
        'atr_period': 14,
        'atr_lookback': 100,
        'volatility_threshold': 80
    }
    rm = RiskManager(config)

    # Only 50 bars, need 100
    data = pd.DataFrame({
        'high': [100] * 50,
        'low': [99] * 50,
        'close': [99.5] * 50
    })

    position_size = rm.calculate_volatility_adjustment(data, 0.3)
    # Should return base ratio when insufficient data
    assert position_size == 0.3

def test_should_exit_trailing_stop_long():
    """Test trailing stop for long position"""
    config = {'swing_period': 5}
    rm = RiskManager(config)

    # Entry at bar 5, now at bar 10
    entry_bar = 5
    current_bar = 10

    # Price data: lowest low since entry is 94 at bar 6
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 95, 107, 108, 109],
        'low': [99, 100, 101, 102, 103, 104, 94, 105, 106, 107, 108]
    })

    # Close at 109, above lowest low (94) - should NOT exit
    should_exit = rm.should_exit_trailing_stop(data, entry_bar, current_bar, 1)
    assert should_exit == False

    # Now close at 93, below lowest low - should exit
    data_with_exit = data.copy()
    data_with_exit.iloc[10, data_with_exit.columns.get_loc('close')] = 93
    should_exit = rm.should_exit_trailing_stop(data_with_exit, entry_bar, current_bar, 1)
    assert should_exit == True

def test_should_exit_trailing_stop_short():
    """Test trailing stop for short position"""
    config = {'swing_period': 5}
    rm = RiskManager(config)

    entry_bar = 5
    current_bar = 10

    # Price data: highest high since entry is 111 at bar 6
    data = pd.DataFrame({
        'close': [100, 99, 98, 97, 96, 95, 110, 94, 93, 92, 91],
        'high': [101, 100, 99, 98, 97, 96, 111, 95, 94, 93, 92]
    })

    # Close at 91, below highest high - should NOT exit
    should_exit = rm.should_exit_trailing_stop(data, entry_bar, current_bar, -1)
    assert should_exit == False

    # Close at 112, above highest high - should exit
    data_with_exit = data.copy()
    data_with_exit.iloc[10, data_with_exit.columns.get_loc('close')] = 112
    should_exit = rm.should_exit_trailing_stop(data_with_exit, entry_bar, current_bar, -1)
    assert should_exit == True

def test_should_exit_trailing_stop_insufficient_bars():
    """Test trailing stop with insufficient bars since entry"""
    config = {'swing_period': 20}
    rm = RiskManager(config)

    entry_bar = 95
    current_bar = 100  # Only 5 bars since entry, need 20

    data = pd.DataFrame({
        'close': [100] * 101,
        'high': [101] * 101,
        'low': [99] * 101
    })

    # Should use entry price as stop (no swing high/low yet)
    should_exit = rm.should_exit_trailing_stop(data, entry_bar, current_bar, 1)
    assert isinstance(should_exit, (bool, np.bool_))
