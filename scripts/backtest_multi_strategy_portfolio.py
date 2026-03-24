#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtest script for improved multi-strategy portfolio
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor
from strategies.portfolio.improved_multi_strategy_portfolio import ImprovedMultiStrategyPortfolio


def main():
    """Main backtest function"""
    print("=" * 80)
    print("Multi-Strategy Portfolio Backtest")
    print("=" * 80)

    # Load data
    print("\nLoading TA futures data...")
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    df = processor.process(adjust_price=True)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Resample to daily
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Test period (same as RARS for comparison)
    test_start = '2019-01-01'
    test_end = '2025-03-21'
    df_test = daily_df.loc[test_start:test_end].copy()

    print(f"Data loaded: {len(df_test)} trading days")
    print(f"Test period: {test_start} to {test_end}")

    # Initialize portfolio
    print("\nInitializing portfolio...")
    portfolio = ImprovedMultiStrategyPortfolio(
        initial_cash=1000000,
        config={
            'rebalance_days': 30,  # Monthly rebalancing
            'max_drawdown': 0.15,  # 15% max drawdown
        }
    )

    # Run backtest with dynamic weights
    print("\n" + "-" * 80)
    print("Testing DYNAMIC WEIGHT allocation (regime-based)...")
    print("-" * 80)

    result_dynamic = portfolio.run_backtest(df_test, use_dynamic_weights=True)
    metrics_dynamic = portfolio.calculate_metrics(result_dynamic)
    portfolio.print_results(metrics_dynamic, "Dynamic Allocation Portfolio (Regime-Based)")

    # Run backtest with static weights for comparison
    print("\n" + "-" * 80)
    print("Testing STATIC WEIGHT allocation (40/30/30)...")
    print("-" * 80)

    portfolio_static = ImprovedMultiStrategyPortfolio(
        initial_cash=1000000,
        config={
            'base_weights': {'RARS': 0.40, 'LongOnlyMA': 0.30, 'EnhancedMA': 0.30},
            'rebalance_days': 30,
        }
    )

    result_static = portfolio_static.run_backtest(df_test, use_dynamic_weights=False)
    metrics_static = portfolio_static.calculate_metrics(result_static)
    portfolio_static.print_results(metrics_static, "Static Allocation Portfolio (40/30/30)")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    comparison = pd.DataFrame({
        'Metric': ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Final Value'],
        'Dynamic (Regime-Based)': [
            f"{metrics_dynamic['annual_return']:.2%}",
            f"{metrics_dynamic['sharpe_ratio']:.2f}",
            f"{metrics_dynamic['max_drawdown']:.2%}",
            f"${metrics_dynamic['final_value']:,.2f}"
        ],
        'Static (40/30/30)': [
            f"{metrics_static['annual_return']:.2%}",
            f"{metrics_static['sharpe_ratio']:.2f}",
            f"{metrics_static['max_drawdown']:.2%}",
            f"${metrics_static['final_value']:,.2f}"
        ],
    })

    print(comparison.to_string(index=False))

    # Compare with individual strategies
    print("\n" + "=" * 80)
    print("COMPARISON WITH INDIVIDUAL STRATEGIES")
    print("=" * 80)

    individual_results = pd.DataFrame({
        'Strategy': ['Buy & Hold', 'RARS (Best)', 'Multi-Strategy (Dynamic)', 'Multi-Strategy (Static)'],
        'Annual Return': [
            '-5.50%',  # Approximate from -33% over 6 years
            '2.00%',
            f"{metrics_dynamic['annual_return']:.2%}",
            f"{metrics_static['annual_return']:.2%}"
        ],
        'Notes': [
            'Baseline',
            'MA200 regimes',
            'Regime-based weights',
            'Fixed 40/30/30'
        ]
    })

    print(individual_results.to_string(index=False))

    # Regime analysis
    print("\n" + "=" * 80)
    print("REGIME ANALYSIS")
    print("=" * 80)

    regime_counts = result_dynamic['regime_history'].value_counts()
    total_days = len(result_dynamic['regime_history'])

    print(f"\nTotal days: {total_days}")
    for regime in ['BULL', 'BEAR', 'RANGING']:
        if regime in regime_counts:
            count = regime_counts[regime]
            pct = count / total_days * 100
            print(f"{regime}: {count} days ({pct:.1f}%)")

    print("\n" + "=" * 80)
    print("Backtest complete!")
    print("=" * 80)

    return {
        'dynamic': metrics_dynamic,
        'static': metrics_static,
        'regime_distribution': regime_counts.to_dict()
    }


if __name__ == '__main__':
    results = main()
