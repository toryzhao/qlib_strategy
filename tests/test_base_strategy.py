# trading/tests/test_base_strategy.py
import pytest
from strategies.base.base_strategy import FuturesStrategy

class DummyStrategy(FuturesStrategy):
    """Dummy strategy for testing"""

    def generate_signals(self, data):
        import pandas as pd
        return pd.Series(0, index=data.index)

def test_base_strategy_initialization():
    """Test that base strategy can be initialized"""
    config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2023-12-31'
    }
    strategy = DummyStrategy('TA', '2020-01-01', '2023-12-31', config)
    assert strategy.instrument == 'TA'
    assert strategy.start_date == '2020-01-01'
    assert strategy.end_date == '2023-12-31'
    assert strategy.config == config

def test_generate_signals_abstract():
    """Test that generate_signals must be implemented"""
    from strategies.base.base_strategy import FuturesStrategy
    config = {'instrument': 'TA'}
    # Should raise TypeError when trying to instantiate without implementing abstract method
    with pytest.raises(TypeError, match="abstract"):
        FuturesStrategy('TA', '2020-01-01', '2023-12-31', config)
