#!/usr/bin/env python
"""
Optimize mean reversion strategy parameters

Usage:
    python scripts/optimize_mean_reversion.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import ContinuousContractProcessor
from strategies.statistical.mean_reversion import MeanReversionStrategy
from executors.parameter_optimizer import ParameterOptimizer
import pandas as pd


def main():
    # Load data
    csv_path = 'data/raw/TA.csv'
    print(f"Loading data: {csv_path}")

    processor = ContinuousContractProcessor(csv_path)
    data = processor.process(adjust_price=True)
    data = processor.load_data(start_date='2020-01-01', end_date='2020-12-31')

    print(f"Data loaded: {len(data)} records")

    # Base configuration
    base_config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'initial_cash': 1000000,
        'position_ratio': 0.3,
        'commission_rate': 0.0001,
    }

    # Parameter grid for mean reversion
    param_grid = {
        'lookback_period': [10, 15, 20, 25, 30],
        'entry_threshold': [1.2, 1.5, 1.8, 2.0],
        'exit_threshold': [0.3, 0.5, 0.7],
    }

    # Calculate total combinations
    from sklearn.model_selection import ParameterGrid
    total_combinations = len(list(ParameterGrid(param_grid)))

    # Create optimizer
    optimizer = ParameterOptimizer(MeanReversionStrategy, data, base_config)

    # Run optimization
    print("\n" + "=" * 60)
    print("Mean Reversion Parameter Optimization")
    print("=" * 60)
    print(f"\nParameter grid: {total_combinations} combinations")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    best_result, results_df = optimizer.grid_search(
        param_grid,
        metric='sharpe_ratio',
        verbose=True
    )

    # Print results
    optimizer.print_summary(best_result)
    optimizer.save_results(results_df, 'reports/TA_mean_reversion_optimization.csv')

    # Save best parameters to JSON
    import json
    with open('reports/TA_mean_reversion_best_params.json', 'w') as f:
        json.dump(best_result['params'], f, indent=2)
    print(f"\nBest parameters saved to: reports/TA_mean_reversion_best_params.json")


if __name__ == '__main__':
    main()
