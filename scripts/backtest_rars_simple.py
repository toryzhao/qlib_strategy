#!/usr/bin/env python
"""
Simplified RARS Backtest with Real Regime Labels
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor


def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high = data['high']
    low = data['low']
    close = data['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_dynamic_window(atr_series, current_idx, window_short=10, window_long=30):
    """Calculate dynamic window based on volatility"""
    baseline_idx = max(0, current_idx - 60)
    atr_baseline = atr_series.iloc[baseline_idx:current_idx].median()
    current_atr = atr_series.iloc[current_idx]

    if current_atr > atr_baseline:
        return window_short
    else:
        return window_long


def run_backtest(data, assignments, initial_cash=1000000):
    """
    Run RARS backtest

    Strategy logic:
    - BULL: Buy when price <= (recent_low + 1*ATR)
    - BEAR: Short when price >= (recent_high - 1*ATR)
    - RANGING: Buy on breakout (high + 1*ATR) with confirmation, short on breakdown
    """
    print("\n执行 RARS 策略回测...")

    # Calculate ATR
    atr = calculate_atr(data)

    # Initialize
    cash = initial_cash
    position = 0  # Positive = long, negative = short
    position_entry_price = None
    position_side = None
    portfolio_values = []
    trades = []

    pending_signal = None

    # Merge assignments with data
    assignments['Date'] = pd.to_datetime(assignments['Date'])
    data_merged = data.reset_index().merge(
        assignments[['Date', 'Trading_State', 'Confidence']],
        left_on='datetime',
        right_on='Date',
        how='inner'
    ).set_index('Date')

    for i in range(60, len(data_merged)):
        current_data = data_merged.iloc[:i+1]
        current_date = current_data.index[-1]
        current_price = current_data['close'].iloc[-1]
        current_atr = atr.iloc[i]  # Use integer index
        current_regime = current_data['Trading_State'].iloc[-1]

        # Check stop loss first
        if position != 0 and position_entry_price is not None:
            if position > 0:  # Long position
                stop_loss = position_entry_price - (2 * current_atr)
                if current_price < stop_loss:
                    # Stop loss hit
                    cash += position * current_price
                    if trades:
                        pnl = (current_price - position_entry_price) / position_entry_price
                        trades[-1]['exit_date'] = current_date
                        trades[-1]['exit_price'] = current_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_reason'] = 'STOP_LOSS'
                    position = 0
                    position_entry_price = None
                    position_side = None
                    continue

            elif position < 0:  # Short position
                stop_loss = position_entry_price + (2 * current_atr)
                if current_price > stop_loss:
                    # Stop loss hit
                    cash -= abs(position) * current_price
                    if trades:
                        pnl = (position_entry_price - current_price) / position_entry_price
                        trades[-1]['exit_date'] = current_date
                        trades[-1]['exit_price'] = current_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_reason'] = 'STOP_LOSS'
                    position = 0
                    position_entry_price = None
                    position_side = None
                    continue

        # Generate signals based on regime
        signal = None

        if current_regime == 'BULL':
            # Buy on pullback to recent low
            window = calculate_dynamic_window(current_data['close'], len(current_data) - 1)
            recent_low = current_data['low'].rolling(window).min().iloc[-1]
            entry_zone = recent_low + (1 * current_atr)

            if position == 0 and current_price <= entry_zone:
                signal = 'LONG'

        elif current_regime == 'BEAR':
            # Short on rally to recent high
            window = calculate_dynamic_window(current_data['close'], len(current_data) - 1)
            recent_high = current_data['high'].rolling(window).max().iloc[-1]
            entry_zone = recent_high - (1 * current_atr)

            if position == 0 and current_price >= entry_zone:
                signal = 'SHORT'

        elif current_regime == 'RANGING':
            # Trade breakouts with confirmation
            window = calculate_dynamic_window(current_data['close'], len(current_data) - 1)
            recent_high = current_data['high'].rolling(window).max().iloc[-1]
            recent_low = current_data['low'].rolling(window).min().iloc[-1]

            # Check for pending signal confirmation
            if pending_signal == 'LONG_PENDING':
                breakout_level = recent_high + (1 * current_atr)
                if current_price > breakout_level:
                    signal = 'LONG'
                else:
                    pending_signal = None  # Failed confirmation

            elif pending_signal == 'SHORT_PENDING':
                breakdown_level = recent_low - (1 * current_atr)
                if current_price < breakdown_level:
                    signal = 'SHORT'
                else:
                    pending_signal = None  # Failed confirmation

            # Check for new breakouts
            if position == 0 and pending_signal is None:
                breakout_level = recent_high + (1 * current_atr)
                if current_price > breakout_level:
                    pending_signal = 'LONG_PENDING'

                breakdown_level = recent_low - (1 * current_atr)
                if current_price < breakdown_level:
                    pending_signal = 'SHORT_PENDING'

        # Execute signal
        if signal == 'LONG' and position == 0:
            # Enter long
            contracts = int(initial_cash * 0.02 / (2 * current_atr) / current_price)
            contracts = max(1, contracts)  # At least 1 contract

            position_value = contracts * current_price
            cash -= position_value
            position = contracts
            position_entry_price = current_price
            position_side = 'LONG'

            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': current_price,
                'size': contracts,
                'regime': current_regime
            })

        elif signal == 'SHORT' and position == 0:
            # Enter short
            contracts = int(initial_cash * 0.02 / (2 * current_atr) / current_price)
            contracts = max(1, contracts)

            position_value = contracts * current_price
            cash += position_value
            position = -contracts
            position_entry_price = current_price
            position_side = 'SHORT'

            trades.append({
                'date': current_date,
                'action': 'SHORT',
                'price': current_price,
                'size': contracts,
                'regime': current_regime
            })

        # Calculate portfolio value
        if position > 0:
            portfolio_value = cash + position * current_price
        elif position < 0:
            portfolio_value = cash - abs(position) * current_price
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)

    # Close final position
    if position != 0:
        final_price = data_merged['close'].iloc[-1]
        if position > 0:
            cash += position * final_price
        else:
            cash -= abs(position) * final_price
        portfolio_values[-1] = cash

    return calculate_metrics(portfolio_values, trades, data)


def calculate_metrics(portfolio_values, trades, data):
    """Calculate performance metrics"""
    portfolio_series = pd.Series(portfolio_values)
    returns = portfolio_series.pct_change().dropna()

    years = (data.index[-1] - data.index[0]).days / 365.25

    completed_trades = [t for t in trades if 'pnl' in t]

    metrics = {
        'final_value': portfolio_values[-1],
        'total_return': (portfolio_values[-1] / portfolio_values[0] - 1),
        'annual_return': returns.mean() * 252 if len(returns) > 0 else 0,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(returns),
        'num_trades': len(completed_trades),
        'win_rate': sum(1 for t in completed_trades if t['pnl'] > 0) / len(completed_trades) if completed_trades else 0,
        'portfolio_values': portfolio_series,
        'trades': trades
    }

    return metrics


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def print_results(metrics):
    """Print performance results"""
    print(f"\n{'=' * 80}")
    print("RARS 策略回测结果")
    print(f"{'=' * 80}")

    if metrics is None:
        print("无可用数据")
        return

    print(f"\n最终资金: ${metrics['final_value']:,.2f}")
    print(f"总收益率: {metrics['total_return']:.2%}")
    print(f"年化收益率: {metrics['annual_return']:.2%}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"交易次数: {metrics['num_trades']}")
    print(f"胜率: {metrics['win_rate']:.2%}")

    # Analyze by regime
    print(f"\n{'=' * 80}")
    print("各机制交易表现")
    print(f"{'=' * 80}")

    for regime in ['BULL', 'BEAR', 'RANGING']:
        regime_trades = [t for t in metrics['trades'] if t.get('regime') == regime and 'pnl' in t]

        if len(regime_trades) > 0:
            pnls = [t['pnl'] for t in regime_trades]
            avg_pnl = np.mean(pnls) * 100
            win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)

            print(f"\n{regime}:")
            print(f"  交易次数: {len(regime_trades)}")
            print(f"  平均收益率: {avg_pnl:.2f}%")
            print(f"  胜率: {win_rate:.2%}")


def main():
    """Main function"""
    print("=" * 80)
    print("RARS 策略回测 - TA期货（2019-2025）")
    print("使用真实的 HMM 机制标签")
    print("=" * 80)

    # Load data
    print("\n加载 TA 期货数据...")
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

    print(f"数据加载完成: {len(df_test)} 个交易日")
    print(f"回测周期: {test_start} 到 {test_end}")

    # Load regime labels
    print("\n加载 HMM 机制标签...")
    assignments = pd.read_csv('data/regimetry/reports/HMM_TA/cluster_assignments.csv')

    print(f"机制标签加载完成: {len(assignments)} 个数据点")

    # Show regime distribution
    print(f"\n机制分布:")
    for regime in ['BULL', 'BEAR', 'RANGING']:
        count = (assignments['Trading_State'] == regime).sum()
        pct = count / len(assignments) * 100
        print(f"  {regime}: {count} 天 ({pct:.1f}%)")

    # Run backtest
    metrics = run_backtest(df_test, assignments)

    # Print results
    print_results(metrics)

    return metrics


if __name__ == '__main__':
    main()
