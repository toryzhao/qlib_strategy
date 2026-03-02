"""
Tests for MeanReversionStrategy with trend filter enabled
"""
import pytest
import pandas as pd
import numpy as np
from strategies.statistical.mean_reversion import MeanReversionStrategy


def test_trend_filter_blocks_downtrend():
    """Test that trend filter prevents trading in downtrend"""
    # Create downtrend data (similar to 2020 TA)
    data = pd.DataFrame({
        'close': [4500 - i * 3 for i in range(100)]
    })

    config = {
        'use_trend_filter': True,
        'trend_filter_lookback': 50,
        'trend_slope_threshold': 0.005,
        'trend_r2_threshold': 0.3,
        'lookback_period': 20,
        'entry_threshold': 1.5
    }

    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(data)

    # All signals should be 0 (no trading) in downtrend
    assert (signals == 0).all(), "Trend filter should block all signals in downtrend"


def test_trend_filter_allows_sideways():
    """Test that trend filter allows trading in sideways market"""
    # Create sideways data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 4500 + np.random.randn(150) * 30
    })

    config = {
        'use_trend_filter': True,
        'trend_filter_lookback': 60,
        'trend_slope_threshold': 0.01,
        'trend_r2_threshold': 0.8,
        'lookback_period': 20,
        'level1_threshold': 1.0  # Low threshold to trigger signals
    }

    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(data)

    # Should have some non-zero signals in sideways market
    assert (signals != 0).any(), "Should generate signals in sideways market"


def test_trend_filter_can_be_disabled():
    """Test that trend filter can be disabled"""
    # Create downtrend data
    data = pd.DataFrame({
        'close': [4500 - i * 3 for i in range(100)]
    })

    config = {
        'use_trend_filter': False,  # Disable trend filter
        'lookback_period': 20,
        'level1_threshold': 1.0
    }

    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(data)

    # Should generate signals even in downtrend when filter disabled
    assert (signals != 0).any(), "Should generate signals when trend filter disabled"


def test_dynamic_position_in_strong_trend():
    """Test dynamic position control in strong trend"""
    # Create strong uptrend
    data = pd.DataFrame({
        'close': [100 + i * 5 for i in range(100)]
    })

    config = {
        'use_trend_filter': False,  # Don't block trading
        'enable_dynamic_position': True,
        'trend_strength_threshold': 0.01,
        'max_position_strong_trend': 0.1,  # 10% max position
        'max_position_weak_trend': 0.3,
        'lookback_period': 20,
        'level1_threshold': 1.0
    }

    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(data)

    # Should still generate signals but with reduced position potential
    assert (signals != 0).any(), "Should generate signals in strong trend"


def test_dynamic_position_in_weak_trend():
    """Test dynamic position control in weak trend/sideways"""
    # Create sideways data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 4500 + np.random.randn(150) * 20
    })

    config = {
        'use_trend_filter': False,
        'enable_dynamic_position': True,
        'trend_strength_threshold': 0.01,
        'max_position_strong_trend': 0.1,
        'max_position_weak_trend': 0.3,  # 30% max position
        'lookback_period': 20,
        'level1_threshold': 1.0
    }

    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(data)

    # Should generate signals with normal position potential
    assert (signals != 0).any(), "Should generate signals in weak trend"


def test_combined_trend_filter_and_dynamic_position():
    """Test trend filter and dynamic position working together"""
    # Strong downtrend - should be blocked by trend filter
    downtrend_data = pd.DataFrame({
        'close': [4500 - i * 5 for i in range(100)]
    })

    config = {
        'use_trend_filter': True,
        'enable_dynamic_position': True,
        'trend_filter_lookback': 50,
        'trend_slope_threshold': 0.005,
        'trend_r2_threshold': 0.8,
        'trend_strength_threshold': 0.01,
        'max_position_strong_trend': 0.1,
        'max_position_weak_trend': 0.3,
        'lookback_period': 20,
        'level1_threshold': 1.0
    }

    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(downtrend_data)

    # Trend filter should prevent all trading
    assert (signals == 0).all(), "Trend filter should block trading in strong downtrend"


def test_backward_compatibility():
    """Test that strategy works without new config options"""
    # Old style config - should work with defaults
    data = pd.DataFrame({
        'close': [100 + i * 2 for i in range(100)]
    })

    config = {
        'lookback_period': 20,
        'entry_threshold': 1.5
    }

    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(data)

    # Should generate signals with default settings
    # (trend filter enabled by default, so may block in uptrend)
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(data)
