#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Regime-Based Portfolio

If BEAR market: Hold cash (don't trade)
If BULL or RANGING: Use simple MA crossover trend following

This is much simpler and more robust than the complex multi-strategy approach.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor


def detect_regime(data_slice):
    """Detect market regime using MA200 methodology"""
    if len(data_slice) < 200:
        return 'RANGING'

    # Calculate MA200
    ma200 = data_slice['close'].rolling(window=200).mean()
    ma_slope = ma200.diff(5)

    # Current values
    price = data_slice['close'].iloc[-1]
    current_ma = ma200.iloc[-1]
    current_slope = ma_slope.iloc[-1]

    if pd.isna(current_ma) or pd.isna(current_slope):
        return 'RANGING'

    # 2% threshold
    threshold = price * 0.02

    if price > (current_ma + threshold) and current_slope > 0:
        return 'BULL'
    elif price < (current_ma - threshold) and current_slope < 0:
        return 'BEAR'
    else:
        return 'RANGING'


def run_simple_portfolio_backtest(data, initial_cash=1000000, fast_ma=5, slow_ma=20):
    """
    Run simple regime-based portfolio backtest

    Strategy:
    - BEAR regime: Hold cash (0% position)
    - BULL/RANGING: MA crossover (long when fast > slow, cash when fast < slow)
    """
    print("\n" + "=" * 80)
    print("Simple Regime-Based Portfolio Backtest")
    print("=" * 80)
    print(f"\nStrategy:")
    print(f"  - BEAR market: Hold cash (avoid losses)")
    print(f"  - BULL/RANGING: MA({fast_ma}/{slow_ma}) trend following")
    print(f"\nTest period: {data.index[0]} to {data.index[-1]}")

    # Initialize
    cash = initial_cash
    position = 0  # Number of contracts
    position_entry_price = None
    portfolio_values = []

    # Calculate MAs
    ma_fast = data['close'].rolling(window=fast_ma).mean()
    ma_slow = data['close'].rolling(window=slow_ma).mean()

    # Track regime and trades
    regimes = []
    trades = []

    for i in range(max(200, slow_ma), len(data)):
        current_date = data.index[i]
        current_price = data['close'].iloc[i]

        # Detect regime
        regime = detect_regime(data.iloc[:i+1])
        regimes.append(regime)

        # Check for MA crossover signal (only if NOT in BEAR regime)
        if regime != 'BEAR':
            ma_cross_up = ma_fast.iloc[i] > ma_slow.iloc[i]
            ma_cross_down = ma_fast.iloc[i] < ma_slow.iloc[i]

            # Enter long
            if ma_cross_up and position == 0:
                # Use 20% of capital (conservative)
                position_value = cash * 0.20
                position = position_value / current_price
                cash -= position_value
                position_entry_price = current_price

                trades.append({
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'regime': regime,
                    'action': 'BUY'
                })

            # Exit long
            elif ma_cross_down and position > 0:
                cash += position * current_price
                position = 0

                if trades:
                    trades[-1]['exit_date'] = current_date
                    trades[-1]['exit_price'] = current_price
                    pnl = (current_price - position_entry_price) / position_entry_price
                    trades[-1]['pnl'] = pnl

        else:
            # BEAR regime - exit any position and hold cash
            if position > 0:
                cash += position * current_price
                position = 0

                if trades:
                    trades[-1]['exit_date'] = current_date
                    trades[-1]['exit_price'] = current_price
                    pnl = (current_price - position_entry_price) / position_entry_price
                    trades[-1]['pnl'] = pnl

        # Calculate portfolio value
        if position > 0:
            portfolio_value = cash + position * current_price
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)

    # Create results
    equity_curve = pd.Series(portfolio_values, index=data.index[max(200, slow_ma):])
    regime_series = pd.Series(regimes, index=data.index[max(200, slow_ma):])

    # Calculate metrics
    returns = equity_curve.pct_change().dropna()

    total_return = (equity_curve.iloc[-1] / initial_cash) - 1
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nInitial Capital: ${initial_cash:,.2f}")
    print(f"Final Value: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")

    # Trade analysis
    completed_trades = [t for t in trades if 'pnl' in t]
    if completed_trades:
        print(f"\nTotal Trades: {len(completed_trades)}")
        winning_trades = sum(1 for t in completed_trades if t['pnl'] > 0)
        print(f"Winning Trades: {winning_trades} ({winning_trades/len(completed_trades):.1%})")

        total_pnl = sum(t['pnl'] for t in completed_trades)
        print(f"Average PnL: {total_pnl/len(completed_trades):.2%}")

    # Regime analysis
    regime_counts = regime_series.value_counts()
    print(f"\nRegime Distribution:")
    for regime in ['BULL', 'BEAR', 'RANGING']:
        if regime in regime_counts:
            count = regime_counts[regime]
            pct = count / len(regime_series) * 100
            print(f"  {regime}: {count} days ({pct:.1f}%)")

    print("\n" + "=" * 80)

    return {
        'equity_curve': equity_curve,
        'regime_series': regime_series,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': equity_curve.iloc[-1]
        },
        'trades': trades
    }


def main():
    """Main function"""
    # Load data
    print("Loading TA futures data...")
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

    # Test period
    test_start = '2019-01-01'
    test_end = '2025-03-21'
    df_test = daily_df.loc[test_start:test_end].copy()

    print(f"Data loaded: {len(df_test)} trading days")
    print(f"Test period: {test_start} to {test_end}")

    # Run backtest
    results = run_simple_portfolio_backtest(df_test, fast_ma=5, slow_ma=20)

    return results


if __name__ == '__main__':
    results = main()
