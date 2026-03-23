import pytest
import pandas as pd
import numpy as np
from strategies.technical.enhanced_ma_strategy import EnhancedMAStrategy
from strategies.regime.regime_detector import RegimeDetector

def test_enhanced_ma_initialization():
    """Test strategy initialization with default config"""
    config = {
        'ma_period': 20,
        'base_position': 1.00,
        'confidence_threshold': 0.60,
        'min_position': 0.30,
        'max_position': 1.50,
    }

    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    assert strategy.ma_period == 20
    assert strategy.base_position == 1.00
    assert strategy.confidence_threshold == 0.60

def test_generate_signals_no_filter():
    """Test signal generation without regime filter (baseline MA)"""
    config = {'ma_period': 20, 'use_regime_filter': False}
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # Create sample data with clear trend
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.arange(100) * 0.5  # Upward trend
    df = pd.DataFrame({
        'close': prices,
        'open': prices - 0.5,
        'high': prices + 0.5,
        'low': prices - 1.0
    }, index=dates)

    signals = strategy.generate_signals(df)

    # Should return DataFrame with signal and position_size columns
    assert 'signal' in signals.columns
    assert 'position_size' in signals.columns

    # Should have long signals in uptrend
    assert (signals['signal'] == 1).sum() > 50

def test_calculate_position_size():
    """Test dynamic position sizing based on confidence and ADX"""
    config = {
        'ma_period': 20,
        'base_position': 1.00,
        'min_position': 0.30,
        'max_position': 1.50,
    }
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # High confidence, strong trend
    size1 = strategy.calculate_position_size(confidence=0.90, adx=35)
    assert 1.0 <= size1 <= 1.5

    # Medium confidence, medium trend
    size2 = strategy.calculate_position_size(confidence=0.70, adx=25)
    assert 0.5 <= size2 <= 1.0

    # Low confidence, weak trend
    size3 = strategy.calculate_position_size(confidence=0.65, adx=15)
    assert 0.3 <= size3 <= 0.7

def test_should_enter_market():
    """Test entry conditions with regime filter"""
    config = {
        'ma_period': 20,
        'use_regime_filter': True,
        'confidence_threshold': 0.60,
    }
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # Bull regime, high confidence - should enter
    assert strategy.should_enter_market(regime=2, confidence=0.80) == True

    # Bull regime, low confidence - should not enter
    assert strategy.should_enter_market(regime=2, confidence=0.50) == False

    # Ranging regime - should not enter
    assert strategy.should_enter_market(regime=1, confidence=0.80) == False

    # Bear regime - should not enter
    assert strategy.should_enter_market(regime=0, confidence=0.80) == False

def test_should_exit_market():
    """Test exit conditions"""
    config = {
        'ma_period': 20,
        'use_regime_filter': True,
        'confidence_threshold': 0.60,
        'stop_loss': 0.10,
    }
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)
    strategy.entry_price = 100.0

    # Price below MA - should exit
    assert strategy.should_exit_market(price=99, ma=100, regime=2, confidence=0.80) == True

    # Regime changed to bear - should exit
    assert strategy.should_exit_market(price=105, ma=100, regime=0, confidence=0.80) == True

    # Confidence dropped - should exit
    assert strategy.should_exit_market(price=105, ma=100, regime=2, confidence=0.40) == True

    # Stop loss hit - should exit
    assert strategy.should_exit_market(price=89, ma=100, regime=2, confidence=0.80) == True

    # All good - should not exit
    assert strategy.should_exit_market(price=105, ma=100, regime=2, confidence=0.80) == False

def test_enhanced_signals_with_regime_filter():
    """Test full signal generation with regime filter"""
    config = {
        'ma_period': 20,
        'use_regime_filter': True,
        'confidence_threshold': 0.60,
    }

    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(200).cumsum()
    df = pd.DataFrame({
        'close': prices,
        'open': prices - 0.5,
        'high': prices + 0.5,
        'low': prices - 1.0
    }, index=dates)

    # Train regime detector
    regime_detector = RegimeDetector(n_states=3, random_state=42)
    regime_detector.fit(df)
    strategy.regime_detector = regime_detector

    # Generate signals
    signals = strategy.generate_signals(df)

    # Should have signal and position_size columns
    assert 'signal' in signals.columns
    assert 'position_size' in signals.columns

    # Signals should be 0, 1 only (long-only)
    assert set(signals['signal'].unique()).issubset({0, 1})
