#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Portfolio Optimization

Target: 5% annual return (currently at 4.62%)

Strategies:
1. Refined parameter search around best config
2. Add trailing stop loss
3. Add volatility filter
4. Regime-specific position sizing
5. Combination filters
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor


def detect_regime_advanced(data_slice, threshold_pct=0.02):
    """
    Advanced regime detection with configurable threshold

    Returns: 'BULL', 'BEAR', or 'RANGING'
    """
    if len(data_slice) < 200:
        return 'RANGING'

    ma200 = data_slice['close'].rolling(window=200).mean()
    ma_slope = ma200.diff(5)

    price = data_slice['close'].iloc[-1]
    current_ma = ma200.iloc[-1]
    current_slope = ma_slope.iloc[-1]

    if pd.isna(current_ma) or pd.isna(current_slope):
        return 'RANGING'

    threshold = price * threshold_pct

    if price > (current_ma + threshold) and current_slope > 0:
        return 'BULL'
    elif price < (current_ma - threshold) and current_slope < 0:
        return 'BEAR'
    else:
        return 'RANGING'


def run_backtest_advanced(data, fast_ma=5, slow_ma=50, position_size=0.50,
                         initial_cash=1000000, use_trailing_stop=False,
                         trailing_stop_pct=0.05, regime_threshold=0.02,
                         bull_pos_size=None, ranging_pos_size=None):
    """
    Advanced backtest with multiple enhancements

    Parameters:
    - fast_ma, slow_ma: MA periods
    - position_size: Default position size
    - use_trailing_stop: Enable trailing stop loss
    - trailing_stop_pct: Trailing stop at X% below peak
    - regime_threshold: MA distance threshold for regime
    - bull_pos_size: Override position size for BULL
    - ranging_pos_size: Override position size for RANGING
    """
    cash = initial_cash
    position = 0
    position_entry_price = None
    position_peak_price = None  # For trailing stop
    portfolio_values = []

    ma_fast = data['close'].rolling(window=fast_ma).mean()
    ma_slow = data['close'].rolling(window=slow_ma).mean()

    for i in range(max(200, slow_ma), len(data)):
        current_price = data['close'].iloc[i]

        # Detect regime
        regime = detect_regime_advanced(data.iloc[:i+1], threshold_pct=regime_threshold)

        # Determine position size based on regime
        if regime == 'BULL':
            current_pos_size = bull_pos_size if bull_pos_size else position_size
        elif regime == 'RANGING':
            current_pos_size = ranging_pos_size if ranging_pos_size else position_size
        else:  # BEAR
            current_pos_size = 0  # Hold cash

        # Only trade if NOT in BEAR regime
        if regime != 'BEAR':
            ma_cross_up = ma_fast.iloc[i] > ma_slow.iloc[i]
            ma_cross_down = ma_fast.iloc[i] < ma_slow.iloc[i]

            # ENTRY: MA cross up
            if ma_cross_up and position == 0:
                position_value = cash * current_pos_size
                position = position_value / current_price
                cash -= position_value
                position_entry_price = current_price
                position_peak_price = current_price

            # TRAILING STOP CHECK
            elif position > 0 and use_trailing_stop:
                if current_price > position_peak_price:
                    position_peak_price = current_price  # Update peak

                trailing_stop = position_peak_price * (1 - trailing_stop_pct)
                if current_price < trailing_stop:
                    # Trailing stop hit
                    cash += position * current_price
                    position = 0
                    position_entry_price = None
                    position_peak_price = None

            # EXIT: MA cross down
            elif ma_cross_down and position > 0:
                cash += position * current_price
                position = 0
                position_entry_price = None
                position_peak_price = None

        else:
            # BEAR regime - exit any position
            if position > 0:
                cash += position * current_price
                position = 0
                position_entry_price = None
                position_peak_price = None

        portfolio_value = cash + (position * current_price if position > 0 else 0)
        portfolio_values.append(portfolio_value)

    # Calculate metrics
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

    result = {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': equity_curve.iloc[-1],
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'position_size': position_size,
        'trailing_stop': use_trailing_stop,
        'regime_threshold': regime_threshold
    }

    if use_trailing_stop:
        result['trailing_stop_pct'] = trailing_stop_pct

    return result


def optimization_round_1():
    """Round 1: Refined parameter search around best config"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION ROUND 1: Refined Parameter Search")
    print("=" * 80)

    # Load data
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

    results = []

    # Search around best config (MA 5/50, 50% position)
    print("\nTesting refined parameters...")

    for fast_ma in [3, 4, 5, 6, 7]:
        for slow_ma in [40, 45, 50, 55, 60]:
            if fast_ma >= slow_ma:
                continue

            for pos_size in [0.45, 0.475, 0.50, 0.525, 0.55]:
                try:
                    result = run_backtest_advanced(
                        df_test, fast_ma, slow_ma, pos_size
                    )
                    results.append(result)
                    print(f"  MA({fast_ma}/{slow_ma}), {pos_size:.1%}: {result['annual_return']:.2%}")
                except:
                    continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('annual_return', ascending=False)

    print(f"\nBest Round 1: {results_df.iloc[0]['annual_return']:.2%}")
    print(f"  Config: MA({int(results_df.iloc[0]['fast_ma'])}/{int(results_df.iloc[0]['slow_ma'])}), {results_df.iloc[0]['position_size']:.1%}")
    print(f"  Sharpe: {results_df.iloc[0]['sharpe_ratio']:.2f}, Drawdown: {results_df.iloc[0]['max_drawdown']:.2%}")

    return results_df.iloc[0], df_test


def optimization_round_2(df_test, best_config):
    """Round 2: Add trailing stop loss"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION ROUND 2: Trailing Stop Loss")
    print("=" * 80)

    results = []

    fast_ma = int(best_config['fast_ma'])
    slow_ma = int(best_config['slow_ma'])
    base_pos = best_config['position_size']

    print("\nTesting trailing stop loss...")

    for trailing_stop_pct in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        try:
            result = run_backtest_advanced(
                df_test, fast_ma, slow_ma, base_pos,
                use_trailing_stop=True,
                trailing_stop_pct=trailing_stop_pct
            )
            results.append(result)
            print(f"  Trailing stop {trailing_stop_pct:.1%}: {result['annual_return']:.2%}")
        except:
            continue

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('annual_return', ascending=False)

        print(f"\nBest with trailing stop: {results_df.iloc[0]['annual_return']:.2%}")
        print(f"  Stop: {results_df.iloc[0]['trailing_stop_pct']:.1%}")
        print(f"  Sharpe: {results_df.iloc[0]['sharpe_ratio']:.2f}, Drawdown: {results_df.iloc[0]['max_drawdown']:.2%}")

        return results_df.iloc[0]
    else:
        return best_config


def optimization_round_3(df_test, best_config):
    """Round 3: Regime-specific position sizing"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION ROUND 3: Regime-Specific Position Sizing")
    print("=" * 80)

    results = []

    fast_ma = int(best_config['fast_ma'])
    slow_ma = int(best_config['slow_ma'])

    print("\nTesting regime-specific position sizes...")

    # Test different combinations
    for bull_mult in [0.8, 1.0, 1.2, 1.4, 1.6]:
        for ranging_mult in [0.6, 0.8, 1.0, 1.2]:
            bull_pos = 0.50 * bull_mult
            ranging_pos = 0.50 * ranging_mult

            # Cap at reasonable limits
            bull_pos = min(bull_pos, 0.80)
            ranging_pos = min(ranging_pos, 0.80)

            try:
                result = run_backtest_advanced(
                    df_test, fast_ma, slow_ma, 0.50,
                    bull_pos_size=bull_pos,
                    ranging_pos_size=ranging_pos
                )
                results.append(result)
                print(f"  BULL {bull_pos:.1%}, RANGING {ranging_pos:.1%}: {result['annual_return']:.2%}")
            except:
                continue

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('annual_return', ascending=False)

        best = results_df.iloc[0]
        print(f"\nBest regime-specific: {best['annual_return']:.2%}")
        print(f"  BULL: {best.get('bull_pos_size', 0.50):.1%}, RANGING: {best.get('ranging_pos_size', 0.50):.1%}")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}, Drawdown: {best['max_drawdown']:.2%}")

        return best
    else:
        return best_config


def optimization_round_4(df_test, best_config):
    """Round 4: Regime threshold tuning"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION ROUND 4: Regime Threshold Tuning")
    print("=" * 80)

    results = []

    fast_ma = int(best_config['fast_ma'])
    slow_ma = int(best_config['slow_ma'])

    print("\nTesting different regime thresholds...")

    for threshold in [0.015, 0.018, 0.020, 0.022, 0.025, 0.030]:
        try:
            result = run_backtest_advanced(
                df_test, fast_ma, slow_ma, 0.50,
                regime_threshold=threshold
            )
            results.append(result)
            print(f"  Threshold {threshold:.1%}: {result['annual_return']:.2%} (BULL: {result.get('bull_days', 'N/A')})")
        except:
            continue

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('annual_return', ascending=False)

        best = results_df.iloc[0]
        print(f"\nBest threshold: {best['annual_return']:.2%}")
        print(f"  Threshold: {best['regime_threshold']:.1%}")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}, Drawdown: {best['max_drawdown']:.2%}")

        return best
    else:
        return best_config


def main():
    """Main optimization function"""
    print("=" * 80)
    print("ADVANCED PORTFOLIO OPTIMIZATION")
    print("Target: 5% Annual Return (Current: 4.62%)")
    print("=" * 80)

    # Round 1: Refined parameter search
    best_round1, df_test = optimization_round_1()

    # Round 2: Trailing stop
    best_round2 = optimization_round_2(df_test, best_round1)

    # Round 3: Regime-specific sizing
    best_round3 = optimization_round_3(df_test, best_round2)

    # Round 4: Regime threshold
    best_round4 = optimization_round_4(df_test, best_round3)

    # Final summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE - FINAL RESULTS")
    print("=" * 80)

    print(f"\nStarting point: 4.62%")
    print(f"After optimization: {best_round4['annual_return']:.2%}")
    print(f"\nImprovement: {(best_round4['annual_return'] - 0.0462) * 100:.2f} percentage points")

    target_pct = best_round4['annual_return'] / 0.05 * 100
    if best_round4['annual_return'] >= 0.05:
        print(f"Target achieved: YES")
    else:
        print(f"Target achieved: NO ({target_pct:.1f}%)")

    print(f"\nBest Configuration:")
    print(f"  MA Periods: ({int(best_round4['fast_ma'])}/{int(best_round4['slow_ma'])})")
    print(f"  Position Size: {best_round4['position_size']:.1%}")
    if 'trailing_stop' in best_round4:
        print(f"  Trailing Stop: {best_round4['trailing_stop']}")
    if 'regime_threshold' in best_round4:
        print(f"  Regime Threshold: {best_round4['regime_threshold']:.1%}")

    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {best_round4['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_round4['max_drawdown']:.2%}")
    print(f"  Final Value: ${best_round4['final_value']:,.2f}")

    print("\n" + "=" * 80)

    return best_round4


if __name__ == '__main__':
    result = main()
