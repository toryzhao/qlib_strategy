# trading/tests/test_integration.py
import pytest
import pandas as pd
import os
from utils.data_processor import ContinuousContractProcessor
from utils.feature_engineering import FeatureEngineer
from strategies.strategy_factory import StrategyFactory
from executors.backtest_executor import BacktestExecutor
from analyzers.performance_analyzer import PerformanceAnalyzer

def test_full_backtest_workflow():
    """Test complete backtest workflow"""
    # Skip if data file doesn't exist
    csv_path = 'data/raw/TA.csv'
    if not os.path.exists(csv_path):
        pytest.skip(f"Data file not found: {csv_path}")

    # 1. Load and process data
    processor = ContinuousContractProcessor(csv_path)
    data = processor.process(adjust_price=True)
    data = processor.load_data(start_date='2020-01-01', end_date='2020-12-31')
    assert len(data) > 0

    # 2. Add features
    data = FeatureEngineer.add_technical_features(data)
    assert 'MA5' in data.columns
    assert 'MACD' in data.columns

    # 3. Create strategy
    config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'fast_period': 5,
        'slow_period': 20,
        'initial_cash': 1000000,
        'position_ratio': 0.3
    }
    strategy = StrategyFactory.create_strategy('ma_cross', config)

    # 4. Run backtest
    executor = BacktestExecutor(strategy, config)
    portfolio = executor.run_backtest(data)
    assert portfolio is not None
    assert 'returns' in portfolio.columns

    # 5. Calculate metrics
    metrics = executor.get_metrics()
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics

    # 6. Generate report
    analyzer = PerformanceAnalyzer(executor.portfolio)
    output_path = 'reports/test_integration'
    report_metrics = analyzer.generate_report(output_path)
    assert os.path.exists(f'{output_path}/performance_report.txt')
    assert os.path.exists(f'{output_path}/equity_curve.png')
