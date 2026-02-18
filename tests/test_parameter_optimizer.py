# trading/tests/test_parameter_optimizer.py
import pytest
import pandas as pd
import numpy as np
from executors.parameter_optimizer import ParameterOptimizer
from strategies.technical.ma_strategy import MAStrategy

def test_parameter_optimizer_initialization():
    """Test optimizer initialization"""
    data = pd.DataFrame({'close': [100 + i for i in range(100)]})
    base_config = {'instrument': 'TA', 'start_date': '2020-01-01', 'end_date': '2020-12-31'}
    optimizer = ParameterOptimizer(MAStrategy, data, base_config)
    assert optimizer.strategy_class == MAStrategy
    assert optimizer.base_config == base_config

def test_grid_search():
    """Test grid search optimization"""
    # Create sample data
    data = pd.DataFrame({'close': [100 + i * 0.1 for i in range(100)]})
    base_config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'initial_cash': 1000000,
        'position_ratio': 0.3
    }

    optimizer = ParameterOptimizer(MAStrategy, data, base_config)

    # Small parameter grid for testing
    param_grid = {
        'fast_period': [5, 10],
        'slow_period': [20, 30]
    }

    best_result, results_df = optimizer.grid_search(param_grid, metric='sharpe_ratio', verbose=False)

    assert best_result is not None
    assert 'params' in best_result
    assert len(results_df) > 0
    assert 'sharpe_ratio' in results_df.columns

def test_random_search():
    """Test random search optimization"""
    data = pd.DataFrame({'close': [100 + i * 0.1 for i in range(100)]})
    base_config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'initial_cash': 1000000,
        'position_ratio': 0.3
    }

    optimizer = ParameterOptimizer(MAStrategy, data, base_config)

    param_distributions = {
        'fast_period': (5, 20),
        'slow_period': (20, 60)
    }

    best_result, results_df = optimizer.random_search(
        param_distributions,
        n_iter=5,
        metric='sharpe_ratio'
    )

    assert best_result is not None
    assert 'params' in best_result
    assert len(results_df) > 0
