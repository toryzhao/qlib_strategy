"""
Tests for RegimetryAdaptiveRangeStrategy
"""

import pytest
import pandas as pd
import numpy as np
from strategies.technical.regimetry_adaptive_range_strategy import RegimetryAdaptiveRangeStrategy


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'lookback_days': 20,
        'window_short': 10,
        'window_long': 30,
        'entry_buffer_atr': 1.0,
        'stop_loss_atr': 2.0,
        'risk_per_trade': 0.02
    }


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Generate realistic price movements
    base_price = 100.0
    prices = []

    for i in range(100):
        if i == 0:
            price = base_price
        else:
            # Random walk with some volatility
            change = np.random.normal(0, 2)
            price = max(50, prices[-1] + change)

        prices.append(price)

    # Create OHLCV
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.uniform(-0.01, 0)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 + np.random.uniform(-0.02, 0)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in range(100)]
    })

    return data


class TestRegimetryAdaptiveRangeStrategy:
    """Test suite for RegimetryAdaptiveRangeStrategy"""

    def test_strategy_initialization(self, sample_config):
        """Test strategy initialization with config"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        assert strategy.lookback_days == 20
        assert strategy.window_short == 10
        assert strategy.window_long == 30
        assert strategy.entry_buffer_atr == 1.0
        assert strategy.stop_loss_atr == 2.0
        assert strategy.risk_per_trade == 0.02
        assert strategy.pending_signal is None
        assert strategy.entry_price is None
        assert strategy.entry_atr is None
        assert strategy.regime_mapper is not None

    def test_atr_calculation(self, sample_config, sample_ohlcv):
        """Test ATR calculation"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        atr = strategy.calculate_atr(sample_ohlcv, period=14)

        # Check ATR properties
        assert len(atr) == len(sample_ohlcv)
        assert atr.isna().sum() >= 13  # First 13-14 values should be NaN (pandas rolling behavior)
        assert all(atr.dropna() > 0)  # All valid ATR values should be positive
        assert atr.dtype == np.float64

    def test_dynamic_window_calculation(self, sample_config, sample_ohlcv):
        """Test dynamic window calculation based on ATR"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        atr = strategy.calculate_atr(sample_ohlcv, period=14)

        # Test with current index (should have enough data)
        current_idx = 80
        window = strategy.calculate_dynamic_window(atr, current_idx)

        assert window in [strategy.window_short, strategy.window_long]
        assert isinstance(window, int)

    def test_position_size_calculation(self, sample_config):
        """Test position size calculation with 2% risk rule"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        account_value = 100000
        entry_price = 100.0
        atr = 2.0

        contracts, position_value = strategy.calculate_position_size(
            account_value, entry_price, atr
        )

        # Risk amount = 100000 * 0.02 = 2000
        # Stop loss distance = 2.0 * 2.0 = 4.0
        # Position value = 2000 / 4.0 = 500
        # Contracts = 500 / 100 = 5
        expected_risk = account_value * strategy.risk_per_trade
        expected_stop_distance = strategy.stop_loss_atr * atr
        expected_position_value = expected_risk / expected_stop_distance
        expected_contracts = int(expected_position_value / entry_price)

        assert contracts == expected_contracts
        assert position_value == expected_position_value
        assert contracts >= 0

    def test_position_size_with_different_risk(self, sample_config):
        """Test position size with different risk parameters"""
        config = sample_config.copy()
        config['risk_per_trade'] = 0.01  # 1% risk

        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=config
        )

        account_value = 100000
        entry_price = 100.0
        atr = 2.0

        contracts, position_value = strategy.calculate_position_size(
            account_value, entry_price, atr
        )

        # With 1% risk, position should be half the size
        assert contracts == 2  # Half of 4% risk case
        assert position_value == 250  # Half of 4% risk case

    def test_bull_signal_generation(self, sample_config, sample_ohlcv):
        """Test bull signal generation in bull market"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        # Test with no position
        signal = strategy.generate_bull_signal(sample_ohlcv, long_position=0)

        # Signal should be None or 'LONG'
        assert signal in [None, 'LONG']

        # Test when already in position
        signal = strategy.generate_bull_signal(sample_ohlcv, long_position=10)
        assert signal is None

    def test_bear_signal_generation(self, sample_config, sample_ohlcv):
        """Test bear signal generation in bear market"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        # Test with no position
        signal = strategy.generate_bear_signal(sample_ohlcv, short_position=0)

        # Signal should be None or 'SHORT'
        assert signal in [None, 'SHORT']

        # Test when already in position
        signal = strategy.generate_bear_signal(sample_ohlcv, short_position=10)
        assert signal is None

    def test_ranging_signal_no_pending(self, sample_config, sample_ohlcv):
        """Test ranging signal generation without pending signal"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        signal, pending = strategy.generate_ranging_signal(
            sample_ohlcv,
            long_position=0,
            short_position=0,
            pending_signal=None
        )

        # Should return (signal, pending_signal) tuple
        assert signal in [None, 'LONG', 'SHORT']
        assert pending in [None, 'LONG_PENDING', 'SHORT_PENDING']

    def test_ranging_signal_with_pending_long(self, sample_config, sample_ohlcv):
        """Test ranging signal with pending LONG signal"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        # Set pending signal
        signal, pending = strategy.generate_ranging_signal(
            sample_ohlcv,
            long_position=0,
            short_position=0,
            pending_signal='LONG_PENDING'
        )

        # Should confirm or cancel pending
        assert signal in [None, 'LONG']
        assert pending in [None, 'LONG_PENDING']

    def test_ranging_signal_with_pending_short(self, sample_config, sample_ohlcv):
        """Test ranging signal with pending SHORT signal"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        signal, pending = strategy.generate_ranging_signal(
            sample_ohlcv,
            long_position=0,
            short_position=0,
            pending_signal='SHORT_PENDING'
        )

        # Should confirm or cancel pending
        assert signal in [None, 'SHORT']
        assert pending in [None, 'SHORT_PENDING']

    def test_ranging_signal_with_positions(self, sample_config, sample_ohlcv):
        """Test ranging signal when already in positions"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        # Test when already long
        signal, pending = strategy.generate_ranging_signal(
            sample_ohlcv,
            long_position=10,
            short_position=0,
            pending_signal=None
        )

        # Should not add new long signals
        assert signal != 'LONG'
        assert pending != 'LONG_PENDING'

        # Test when already short
        signal, pending = strategy.generate_ranging_signal(
            sample_ohlcv,
            long_position=0,
            short_position=10,
            pending_signal=None
        )

        # Should not add new short signals
        assert signal != 'SHORT'
        assert pending != 'SHORT_PENDING'

    def test_stop_loss_long_position(self, sample_config):
        """Test stop loss checking for long position"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        # Set up long position
        strategy.entry_price = 100.0
        strategy.entry_atr = 2.0

        position = {'side': 'LONG'}

        # Price below stop loss
        stop_loss_price = 100.0 - (2.0 * 2.0) - 0.1
        assert strategy.check_stop_loss(position, stop_loss_price) is True

        # Price above stop loss
        safe_price = 100.0 - (2.0 * 2.0) + 0.1
        assert strategy.check_stop_loss(position, safe_price) is False

        # Price at stop loss
        exact_stop_loss = 100.0 - (2.0 * 2.0)
        assert strategy.check_stop_loss(position, exact_stop_loss) is False

    def test_stop_loss_short_position(self, sample_config):
        """Test stop loss checking for short position"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        # Set up short position
        strategy.entry_price = 100.0
        strategy.entry_atr = 2.0

        position = {'side': 'SHORT'}

        # Price above stop loss
        stop_loss_price = 100.0 + (2.0 * 2.0) + 0.1
        assert strategy.check_stop_loss(position, stop_loss_price) is True

        # Price below stop loss
        safe_price = 100.0 + (2.0 * 2.0) - 0.1
        assert strategy.check_stop_loss(position, safe_price) is False

        # Price at stop loss
        exact_stop_loss = 100.0 + (2.0 * 2.0)
        assert strategy.check_stop_loss(position, exact_stop_loss) is False

    def test_stop_loss_without_entry(self, sample_config):
        """Test stop loss when entry not set"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        position = {'side': 'LONG'}

        # No entry price set
        assert strategy.check_stop_loss(position, 95.0) is False

        # Entry price but no ATR
        strategy.entry_price = 100.0
        assert strategy.check_stop_loss(position, 95.0) is False

    def test_dynamic_window_with_high_volatility(self, sample_config, sample_ohlcv):
        """Test dynamic window returns short window in high volatility"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        atr = strategy.calculate_atr(sample_ohlcv, period=14)

        # Get baseline from earlier period
        baseline_idx = 50
        atr_baseline = atr.iloc[baseline_idx:80].median()

        # Create ATR series with high current value
        atr_high_vol = atr.copy()
        atr_high_vol.iloc[80] = atr_baseline * 1.5

        window = strategy.calculate_dynamic_window(atr_high_vol, 80)

        # Should use short window when volatility is high
        assert window == strategy.window_short

    def test_dynamic_window_with_low_volatility(self, sample_config, sample_ohlcv):
        """Test dynamic window returns long window in low volatility"""
        strategy = RegimetryAdaptiveRangeStrategy(
            instrument='TEST',
            start_date='2024-01-01',
            end_date='2024-12-31',
            config=sample_config
        )

        atr = strategy.calculate_atr(sample_ohlcv, period=14)

        # Get baseline from earlier period
        baseline_idx = 50
        atr_baseline = atr.iloc[baseline_idx:80].median()

        # Create ATR series with low current value
        atr_low_vol = atr.copy()
        atr_low_vol.iloc[80] = atr_baseline * 0.8

        window = strategy.calculate_dynamic_window(atr_low_vol, 80)

        # Should use long window when volatility is low
        assert window == strategy.window_long
