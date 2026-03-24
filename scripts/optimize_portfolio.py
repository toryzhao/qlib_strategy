#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimize Simple Regime-Based Portfolio parameters

Grid search over:
- Fast MA period: 3-15
- Slow MA period: 15-50
- Position size: 10-50%
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor
from strategies.portfolio.simple_regime_portfolio import detect_regime


def run_backtest_with_params(data, fast_ma, slow_ma, position_size, initial_cash=1000000):
    """Run backtest with specific parameters"""
    cash = initial_cash
    position = 0
    position_entry_price = None
    portfolio_values = []

    ma_fast = data['close'].rolling(window=fast_ma).mean()
    ma_slow = data['close'].rolling(window=slow_ma).mean()

    for i in range(max(200, slow_ma), len(data)):
        current_price = data['close'].iloc[i]
        regime = detect_regime(data.iloc[:i+1])

        # Only trade if NOT in BEAR regime
        if regime != 'BEAR':
            ma_cross_up = ma_fast.iloc[i] > ma_slow.iloc[i]
            ma_cross_down = ma_fast.iloc[i] < ma_slow.iloc[i]

            if ma_cross_up and position == 0:
                position_value = cash * position_size
                position = position_value / current_price
                cash -= position_value
                position_entry_price = current_price

            elif ma_cross_down and position > 0:
                cash += position * current_price
                position = 0
        else:
            if position > 0:
                cash += position * current_price
                position = 0

        portfolio_value = cash + (position * current_price if position > 0 else 0)
        portfolio_values.append(portfolio_value)

    equity_curve = pd.Series(portfolio_values, index=data.index[max(200, slow_ma):])
    returns = equity_curve.pct_change().dropna()

    total_return = (equity_curve.iloc[-1] / initial_cash) - 1
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': equity_curve.iloc[-1],
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'position_size': position_size
    }


def main():
    """Optimization function"""
    print("=" * 80)
    print("Simple Regime Portfolio - Parameter Optimization")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    df = processor.process(adjust_price=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    df_test = daily_df.loc['2019-01-01':'2025-03-21'].copy()
    print(f"Test data: {len(df_test)} days")

    # Grid search
    print("\n" + "-" * 80)
    print("Running grid search...")
    print("-" * 80)

    results = []
    total_tests = 0

    for fast_ma in [3, 5, 8, 10, 12]:
        for slow_ma in [15, 20, 25, 30, 40, 50]:
            if fast_ma >= slow_ma:
                continue

            for position_size in [0.10, 0.20, 0.30, 0.40, 0.50]:
                total_tests += 1

                if total_tests % 10 == 0:
                    print(f"Testing {total_tests}: MA({fast_ma}/{slow_ma}), pos={position_size:.0%}...")

                try:
                    result = run_backtest_with_params(
                        df_test, fast_ma, slow_ma, position_size
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  Error: {e}")
                    continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by annual return
    results_df = results_df.sort_values('annual_return', ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by Annual Return)")
    print("=" * 80)

    for i, row in results_df.head(10).iterrows():
        print(f"\n#{i+1}:")
        print(f"  MA({int(row['fast_ma'])}/{int(row['slow_ma'])})")
        print(f"  Position Size: {row['position_size']:.0%}")
        print(f"  Annual Return: {row['annual_return']:.2%}")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {row['max_drawdown']:.2%}")
        print(f"  Final Value: ${row['final_value']:,.2f}")

    # Best by Sharpe ratio
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS BY DIFFERENT METRICS")
    print("=" * 80)

    best_return = results_df.loc[results_df['annual_return'].idxmax()]
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    best_drawdown = results_df.loc[results_df['max_drawdown'].idxmax()]

    print(f"\nBest Annual Return ({best_return['annual_return']:.2%}):")
    print(f"  MA({int(best_return['fast_ma'])}/{int(best_return['slow_ma'])}), {best_return['position_size']:.0%} position")
    print(f"  Sharpe: {best_return['sharpe_ratio']:.2f}, Drawdown: {best_return['max_drawdown']:.2%}")

    print(f"\nBest Sharpe Ratio ({best_sharpe['sharpe_ratio']:.2f}):")
    print(f"  MA({int(best_sharpe['fast_ma'])}/{int(best_sharpe['slow_ma'])}), {best_sharpe['position_size']:.0%} position")
    print(f"  Return: {best_sharpe['annual_return']:.2%}, Drawdown: {best_sharpe['max_drawdown']:.2%}")

    print(f"\nBest Max Drawdown ({best_drawdown['max_drawdown']:.2%}):")
    print(f"  MA({int(best_drawdown['fast_ma'])}/{int(best_drawdown['slow_ma'])}), {best_drawdown['position_size']:.0%} position")
    print(f"  Return: {best_drawdown['annual_return']:.2%}, Sharpe: {best_drawdown['sharpe_ratio']:.2f}")

    # Save results
    results_df.to_csv('portfolio_optimization_results.csv', index=False)
    print(f"\nResults saved to portfolio_optimization_results.csv ({len(results_df)} tests)")

    print("\n" + "=" * 80)

    return results_df


if __name__ == '__main__':
    results = main()
