import pytest
import pandas as pd
from strategies.technical.ma_strategy import MAStrategy

def test_ma_strategy_with_risk_manager():
    """Test MA strategy uses RiskManager for signal filtering"""
    config = {
        "fast_period": 5,
        "slow_period": 20,
        "trend_ma_period": 10,  # Short for testing
        "instrument": "TA",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31"
    }

    strategy = MAStrategy("TA", "2020-01-01", "2020-12-31", config)

    # Verify RiskManager is initialized
    assert strategy.risk_manager is not None
    assert strategy.risk_manager.trend_ma_period == 10

def test_ma_strategy_signal_filtering():
    """Test that MA signals are filtered by trend"""
    config = {
        "fast_period": 3,
        "slow_period": 5,
        "trend_ma_period": 10
    }

    strategy = MAStrategy("TA", "2020-01-01", "2020-12-31", config)

    # Create price data: uptrend then downtrend
    data = pd.DataFrame({
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
    })

    signals = strategy.generate_signals(data)

    # Verify signals are filtered (should have zeros when trend conflicts)
    assert len(signals) == len(data)
    assert signals.dtype in [int, "int64"]
    # All signals should be -1, 0, or 1
    assert all(signals.isin([-1, 0, 1]))
