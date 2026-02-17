# trading/tests/test_backtest_executor.py
import pytest
import pandas as pd
import numpy as np
from executors.backtest_executor import BacktestExecutor
from executors.futures_config import FuturesBacktestConfig
from strategies.technical.ma_strategy import MAStrategy

def test_backtest_executor_initialization():
    """Test backtest executor initialization"""
    config = {
        'initial_cash': 1000000,
        'position_ratio': 0.3
    }
    strategy = MAStrategy('TA', '2020-01-01', '2020-01-31', {})
    executor = BacktestExecutor(strategy, config)
    assert executor.strategy == strategy
    assert executor.config == config

def test_backtest_executor_runs():
    """Test that backtest executes"""
    config = {
        'initial_cash': 1000000,
        'position_ratio': 0.3
    }
    strategy = MAStrategy('TA', '2020-01-01', '2020-01-31', {'fast_period': 5, 'slow_period': 20})

    # Create sample data
    df = pd.DataFrame({
        'close': [100 + i * 0.1 for i in range(100)]
    })

    executor = BacktestExecutor(strategy, config)
    portfolio = executor.run_backtest(df)

    assert portfolio is not None
    assert 'portfolio_value' in portfolio.columns
    assert 'returns' in portfolio.columns

def test_calculate_metrics():
    """Test metrics calculation"""
    config = {'initial_cash': 1000000, 'position_ratio': 0.3}
    strategy = MAStrategy('TA', '2020-01-01', '2020-01-31', {})

    df = pd.DataFrame({
        'close': [100 + i * 0.1 for i in range(100)]
    })

    executor = BacktestExecutor(strategy, config)
    executor.run_backtest(df)
    metrics = executor.get_metrics()

    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'volatility' in metrics
