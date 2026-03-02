"""
Tests for TrendFilter module
"""
import pytest
import pandas as pd
import numpy as np
from strategies.statistical.trend_filter import TrendFilter


def test_trend_filter_initialization():
    """Test TrendFilter initialization"""
    filter = TrendFilter(lookback=60, slope_threshold=0.005, r2_threshold=0.3)

    assert filter.lookback == 60
    assert filter.slope_threshold == 0.005
    assert filter.r2_threshold == 0.3


def test_detect_uptrend():
    """Test detection of upward trend"""
    # Create strong uptrend data
    data = pd.DataFrame({
        'close': [100 + i * 2 for i in range(100)]  # Clear upward trend
    })

    filter = TrendFilter(lookback=50, slope_threshold=0.01, r2_threshold=0.9)
    trend = filter.detect_trend(data)

    assert trend == 'uptrend'


def test_detect_downtrend():
    """Test detection of downward trend"""
    # Create strong downtrend data
    data = pd.DataFrame({
        'close': [5000 - i * 10 for i in range(100)]  # Clear downward trend
    })

    filter = TrendFilter(lookback=50, slope_threshold=0.01, r2_threshold=0.9)
    trend = filter.detect_trend(data)

    assert trend == 'downtrend'


def test_detect_sideways():
    """Test detection of sideways market"""
    # Create sideways/choppy data
    np.random.seed(42)
    base_price = 4500
    noise = np.random.randn(100) * 50  # Random noise around mean
    data = pd.DataFrame({
        'close': base_price + noise
    })

    filter = TrendFilter(lookback=50, slope_threshold=0.001, r2_threshold=0.1)
    trend = filter.detect_trend(data)

    assert trend == 'sideways'


def test_insufficient_data():
    """Test behavior with insufficient data"""
    data = pd.DataFrame({
        'close': [100, 101, 102]  # Only 3 bars
    })

    filter = TrendFilter(lookback=60)
    trend = filter.detect_trend(data)

    # Should return sideways when insufficient data
    assert trend == 'sideways'


def test_get_trend_strength():
    """Test trend strength calculation"""
    # Strong trend
    strong_trend_data = pd.DataFrame({
        'close': [100 + i * 5 for i in range(100)]
    })

    # Weak trend
    np.random.seed(42)
    weak_trend_data = pd.DataFrame({
        'close': 4500 + np.random.randn(100) * 20
    })

    filter = TrendFilter(lookback=50)

    strong_strength = filter.get_trend_strength(strong_trend_data)
    weak_strength = filter.get_trend_strength(weak_trend_data)

    # Strong trend should have higher strength
    assert strong_strength > weak_strength


def test_should_trade_mean_reversion():
    """Test mean reversion suitability check"""
    # Sideways market - should return True
    np.random.seed(42)
    sideways_data = pd.DataFrame({
        'close': 4500 + np.random.randn(100) * 30
    })

    # Strong uptrend - should return False
    uptrend_data = pd.DataFrame({
        'close': [100 + i * 5 for i in range(100)]
    })

    filter = TrendFilter(lookback=50, slope_threshold=0.01, r2_threshold=0.8)

    assert filter.should_trade_mean_reversion(sideways_data) == True
    assert filter.should_trade_mean_reversion(uptrend_data) == False


def test_get_trend_signal():
    """Test trend signal conversion"""
    uptrend_data = pd.DataFrame({
        'close': [100 + i * 5 for i in range(100)]
    })

    downtrend_data = pd.DataFrame({
        'close': [5000 - i * 10 for i in range(100)]
    })

    np.random.seed(42)
    sideways_data = pd.DataFrame({
        'close': 4500 + np.random.randn(100) * 30
    })

    filter = TrendFilter(lookback=50, slope_threshold=0.01, r2_threshold=0.8)

    assert filter.get_trend_signal(uptrend_data) == 1
    assert filter.get_trend_signal(downtrend_data) == -1
    assert filter.get_trend_signal(sideways_data) == 0


def test_trend_filter_with_2020_ta_data():
    """Test trend filter on actual 2020 TA futures data pattern"""
    # Simulate 2020 TA pattern: downtrend from 4500 to 3500
    data = pd.DataFrame({
        'close': [4500 - i * 3 for i in range(340)]  # Simulate downtrend
    })

    filter = TrendFilter(lookback=60, slope_threshold=0.005, r2_threshold=0.3)
    trend = filter.detect_trend(data)

    # Should detect downtrend
    assert trend == 'downtrend'

    # Should NOT recommend mean reversion
    assert filter.should_trade_mean_reversion(data) == False
