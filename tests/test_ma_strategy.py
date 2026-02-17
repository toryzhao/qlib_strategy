# trading/tests/test_ma_strategy.py
import pytest
import pandas as pd
import numpy as np
from strategies.technical.ma_strategy import MAStrategy

def test_ma_strategy_initialization():
    """Test MA strategy initialization"""
    config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'fast_period': 5,
        'slow_period': 20
    }
    strategy = MAStrategy('TA', '2020-01-01', '2023-12-31', config)
    assert strategy.fast_period == 5
    assert strategy.slow_period == 20

def test_ma_strategy_generates_signals():
    """Test that MA strategy generates signals"""
    config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-01-31',
        'fast_period': 5,
        'slow_period': 20
    }
    strategy = MAStrategy('TA', '2020-01-01', '2020-01-31', config)

    # Create sample data
    df = pd.DataFrame({
        'close': [100 + i for i in range(100)]  # Uptrend
    })

    signals = strategy.generate_signals(df)

    assert isinstance(signals, pd.Series)
    assert len(signals) == 100
    assert signals.iloc[-1] in [-1, 0, 1]  # Last signal should be valid

def test_ma_strategy_default_params():
    """Test MA strategy with default parameters"""
    config = {'instrument': 'TA', 'start_date': '2020-01-01', 'end_date': '2020-01-31'}
    strategy = MAStrategy('TA', '2020-01-01', '2020-01-31', config)
    assert strategy.fast_period == 5  # Default
    assert strategy.slow_period == 20  # Default
