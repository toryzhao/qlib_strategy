# trading/tests/test_strategy_factory.py
import pytest
from strategies.strategy_factory import StrategyFactory

def test_create_ma_strategy():
    """Test creating MA strategy via factory"""
    config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'fast_period': 10,
        'slow_period': 30
    }
    strategy = StrategyFactory.create_strategy('ma_cross', config)
    assert strategy is not None
    assert strategy.instrument == 'TA'
    assert strategy.fast_period == 10
    assert strategy.slow_period == 30

def test_invalid_strategy_type():
    """Test that invalid strategy type raises error"""
    config = {'instrument': 'TA', 'start_date': '2020-01-01', 'end_date': '2020-12-31'}
    with pytest.raises(ValueError, match="Unknown strategy type"):
        StrategyFactory.create_strategy('invalid_strategy', config)
