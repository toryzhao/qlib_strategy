# trading/tests/test_performance_analyzer.py
import pytest
import pandas as pd
import numpy as np
import os
import shutil
from analyzers.performance_analyzer import PerformanceAnalyzer

def test_performance_analyzer_initialization():
    """Test analyzer initialization"""
    portfolio = pd.DataFrame({
        'returns': [0.01, -0.005, 0.02, -0.01, 0.015]
    })
    analyzer = PerformanceAnalyzer(portfolio)
    assert analyzer.portfolio is not None

def test_calculate_metrics():
    """Test metrics calculation"""
    portfolio = pd.DataFrame({
        'returns': [0.01, -0.005, 0.02, -0.01, 0.015]
    })
    analyzer = PerformanceAnalyzer(portfolio)
    metrics = analyzer._calculate_metrics()

    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'volatility' in metrics

def test_generate_report(tmp_path):
    """Test report generation"""
    portfolio = pd.DataFrame({
        'returns': np.random.randn(1000) * 0.01
    })
    portfolio.index = pd.date_range('2020-01-01', periods=1000, freq='D')

    analyzer = PerformanceAnalyzer(portfolio)
    output_path = tmp_path / "reports"
    metrics = analyzer.generate_report(str(output_path))

    assert os.path.exists(output_path / 'performance_report.txt')
    assert 'total_return' in metrics
