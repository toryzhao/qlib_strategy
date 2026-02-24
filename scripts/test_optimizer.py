#!/usr/bin/env python
# Test script to validate optimizer with risk management parameters

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from strategies.technical.ma_strategy import MAStrategy
from executors.parameter_optimizer import ParameterOptimizer

# Create sample data with more realistic patterns
np.random.seed(42)
n = 500

# Generate price data with trend and cycles
trend = np.linspace(100, 120, n)
cycle = 5 * np.sin(np.linspace(0, 4*np.pi, n))
noise = np.random.randn(n) * 2

close = trend + cycle + noise
data = pd.DataFrame({
    'open': close + np.random.randn(n) * 0.5,
    'high': close + np.abs(np.random.randn(n)) * 1.5,
    'low': close - np.abs(np.random.randn(n)) * 1.5,
    'close': close,
    'volume': np.random.randint(1000, 10000, n)
})

print(f"Created test data: {len(data)} records")

# Base configuration
base_config = {
    'instrument': 'TEST',
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'initial_cash': 100000,
    'position_ratio': 0.3,
    'commission_rate': 0.0001,
}

# Small parameter grid for testing
param_grid = {
    'fast_period': [3, 5],
    'slow_period': [10, 15],
    'trend_ma_period': [50, 100],
    'swing_period': [10, 15],
}

print("\nStarting grid search optimization...")
print(f"Parameter combinations to test: 2 * 2 * 2 * 2 = 8")

# Create optimizer
optimizer = ParameterOptimizer(MAStrategy, data, base_config)

# Run grid search
best_result, results_df = optimizer.grid_search(
    param_grid,
    metric='sharpe_ratio',
    verbose=True
)

# Print results
optimizer.print_summary(best_result)

# Save results
output_path = 'reports/test_optimization_results.csv'
os.makedirs('reports', exist_ok=True)
optimizer.save_results(results_df, output_path)

print("\n✓ Optimizer test completed successfully!")
print(f"✓ Results saved to: {output_path}")
