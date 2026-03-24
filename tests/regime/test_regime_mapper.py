import pytest
import pandas as pd
import numpy as np
from strategies.regime.regime_mapper import RegimeMapper


def test_regime_mapper_initialization():
    """Test RegimeMapper can be initialized"""
    mapper = RegimeMapper()
    assert mapper.lookback_days == 20
    assert mapper.threshold_multiplier == 1.5


def test_calculate_cluster_return():
    """Test calculating cluster returns"""
    mapper = RegimeMapper()

    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.arange(100) * 0.5
    df = pd.DataFrame({'close': prices}, index=dates)

    cluster_returns = mapper._calculate_cluster_returns(df, lookback=20)

    assert len(cluster_returns) == 81
    assert cluster_returns.iloc[-1] > 0


def test_map_cluster_to_state_bull():
    """Test mapping cluster to bull state"""
    mapper = RegimeMapper()

    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.arange(100) * 2
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    state = mapper.map_cluster_to_state(df)
    assert state == 'BULL'


def test_map_cluster_to_state_bear():
    """Test mapping cluster to bear state"""
    mapper = RegimeMapper()

    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 - np.arange(100) * 2
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    state = mapper.map_cluster_to_state(df)
    assert state == 'BEAR'


def test_map_cluster_to_state_ranging():
    """Test mapping cluster to ranging state"""
    mapper = RegimeMapper()

    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(100) * 2
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    state = mapper.map_cluster_to_state(df)
    assert state == 'RANGING'


def test_map_all_clusters():
    """Test mapping multiple clusters"""
    mapper = RegimeMapper()

    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    cluster_ids = np.array([0] * 30 + [1] * 40 + [2] * 30)
    assignments = pd.DataFrame({
        'Date': dates,
        'Cluster_ID': cluster_ids
    })

    prices = np.concatenate([
        100 + np.arange(30) * 2,
        100 - np.arange(40),
        np.ones(30) * 100 + np.random.randn(30) * 2
    ])

    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    cluster_states = mapper.map_all_clusters(assignments, df)

    assert 0 in cluster_states
    assert 1 in cluster_states
    assert 2 in cluster_states
