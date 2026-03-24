#!/usr/bin/env python
"""
Improved RARS Strategy with Enhanced Regime Stability

Improvements:
1. Regime smoothing: Require 5 consecutive days in same regime
2. Remove ranging market confirmation requirement
3. Optimize ATR multipliers and dynamic windows
4. Add minimum holding period (5 days)
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor


def generate_trend_based_regimes(data, ma_period=200):
    """
    Generate regime labels based on price trend relative to long-term MA

    This replaces HMM with a simpler, more reliable trend indicator:
    - BULL: Price > MA200 and MA is rising
    - BEAR: Price < MA200 and MA is falling
    - RANGING: Price near MA200 (within 2%)

    Args:
        data: DataFrame with 'close' column
        ma_period: Period for moving average (default 200)

    Returns:
        DataFrame with ['Date', 'Trading_State', 'Trading_State_Smooth']
    """
    # Calculate MA200 and its slope
    ma = data['close'].rolling(window=ma_period).mean()
    ma_slope = ma.diff(5)  # 5-day slope

    # Calculate threshold as % of price
    threshold = data['close'] * 0.02  # 2% band

    # Classify regimes
    regimes = []
    for i in range(len(data)):
        if pd.isna(ma.iloc[i]) or pd.isna(ma_slope.iloc[i]):
            regimes.append('RANGING')
        elif data['close'].iloc[i] > (ma.iloc[i] + threshold.iloc[i]):
            # Price significantly above MA
            if ma_slope.iloc[i] > 0:
                regimes.append('BULL')
            else:
                regimes.append('RANGING')
        elif data['close'].iloc[i] < (ma.iloc[i] - threshold.iloc[i]):
            # Price significantly below MA
            if ma_slope.iloc[i] < 0:
                regimes.append('BEAR')
            else:
                regimes.append('RANGING')
        else:
            # Price within 2% of MA
            regimes.append('RANGING')

    # Create DataFrame
    assignments = pd.DataFrame({
        'Date': data.index,
        'Trading_State': regimes
    })

    # Apply smoothing
    assignments['Trading_State_Smooth'] = smooth_regimes_logic(assignments['Trading_State'], smooth_days=5)

    return assignments


def smooth_regimes_logic(series, smooth_days=5):
    """
    Smooth regime labels to reduce noise

    Require 'smooth_days' consecutive days in same regime to confirm
    """
    smoothed = series.copy()

    for i in range(smooth_days, len(series)):
        recent_states = series.iloc[i-smooth_days:i]

        # If all recent days are same regime, use it
        if (recent_states == recent_states.iloc[0]).all():
            smoothed.iloc[i] = recent_states.iloc[0]
        else:
            # Keep previous smoothed state
            smoothed.iloc[i] = smoothed.iloc[i-1]

    return smoothed


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


def run_improved_backtest(data, assignments=None, initial_cash=1000000):
    """
    Run improved RARS backtest

    Key improvements:
    1. Use trend-based regimes (MA200) instead of HMM
    2. Regime smoothing with 5-day confirmation
    3. Remove confirmation for ranging market breakouts
    4. Use 1.5 ATR buffer instead of 1.0
    5. Minimum 5-day holding period
    6. Dynamic windows: 15/40 instead of 10/30
    """
    print("\n执行改进版 RARS 策略回测...")
    print("改进:")
    print("  - 基于趋势的机制检测 (MA200) 替代 HMM")
    print("  - 机制平滑: 连续5天才确认新机制")
    print("  - 移除震荡市突破确认")
    print("  - ATR倍数: 1.5 (更宽的入场区)")
    print("  - 最短持仓: 5天")
    print("  - 动态窗口: 15/40天")

    # Generate trend-based regimes (ignores HMM assignments)
    print("\n生成基于趋势的机制标签...")
    assignments = generate_trend_based_regimes(data, ma_period=200)

    # Calculate ATR
    atr = calculate_atr(data)

    # Track regime transitions
    print("跟踪机制转换...")
    transitions_before = (assignments['Trading_State'] != assignments['Trading_State'].shift()).sum()
    transitions_after = (assignments['Trading_State_Smooth'] != assignments['Trading_State_Smooth'].shift()).sum()

    print(f"  平滑前转换次数: {transitions_before}")
    print(f"  平滑后转换次数: {transitions_after}")
    print(f"  减少: {transitions_before - transitions_after} 次 ({(transitions_before - transitions_after)/transitions_before*100:.1f}%)")

    # Initialize
    cash = initial_cash
    position = 0
    position_entry_price = None
    position_entry_date = None
    position_side = None
    portfolio_values = []
    trades = []

    # Improved parameters
    min_holding_days = 5
    atr_multiplier = 2.0  # Even more conservative entry

    # Merge assignments with data
    assignments['Date'] = pd.to_datetime(assignments['Date'])
    data_merged = data.reset_index().merge(
        assignments[['Date', 'Trading_State_Smooth']],
        left_on='datetime',
        right_on='Date',
        how='inner'
    ).set_index('Date')

    for i in range(60, len(data_merged)):
        current_data = data_merged.iloc[:i+1]
        current_date = current_data.index[-1]
        current_price = current_data['close'].iloc[-1]

        # Find corresponding ATR value by date
        try:
            current_atr = atr.loc[current_date]
        except KeyError:
            current_atr = atr.iloc[-1]

        current_regime = current_data['Trading_State_Smooth'].iloc[-1]

        # Check stop loss first
        if position != 0 and position_entry_price is not None:
            # Check minimum holding period
            days_held = (current_date - position_entry_date).days

            # Only check stop loss if minimum holding period passed
            if days_held >= min_holding_days:
                if position > 0:  # Long
                    stop_loss = position_entry_price - (2 * current_atr)
                    if current_price < stop_loss:
                        # Close position
                        cash += position * current_price
                        if trades:
                            pnl = (current_price - position_entry_price) / position_entry_price
                            trades[-1]['exit_date'] = current_date
                            trades[-1]['exit_price'] = current_price
                            trades[-1]['pnl'] = pnl
                            trades[-1]['exit_reason'] = 'STOP_LOSS'
                            trades[-1]['days_held'] = days_held
                        position = 0
                        position_entry_price = None
                        position_entry_date = None
                        position_side = None
                        continue

                elif position < 0:  # Short
                    stop_loss = position_entry_price + (2 * current_atr)
                    if current_price > stop_loss:
                        cash -= abs(position) * current_price
                        if trades:
                            pnl = (position_entry_price - current_price) / position_entry_price
                            trades[-1]['exit_date'] = current_date
                            trades[-1]['exit_price'] = current_price
                            trades[-1]['pnl'] = pnl
                            trades[-1]['exit_reason'] = 'STOP_LOSS'
                            trades[-1]['days_held'] = days_held
                        position = 0
                        position_entry_price = None
                        position_entry_date = None
                        position_side = None
                        continue

        # Check if minimum holding period passed and regime changed
        if position != 0 and position_entry_date is not None:
            days_held = (current_date - position_entry_date).days
            if days_held >= min_holding_days:
                # Get regime when entered
                entry_regime = trades[-1]['regime_smooth']

                # If regime changed, exit
                if current_regime != entry_regime:
                    # Close position
                    if position > 0:
                        cash += position * current_price
                        if trades:
                            pnl = (current_price - position_entry_price) / position_entry_price
                            trades[-1]['exit_date'] = current_date
                            trades[-1]['exit_price'] = current_price
                            trades[-1]['pnl'] = pnl
                            trades[-1]['exit_reason'] = 'REGIME_CHANGE'
                            trades[-1]['days_held'] = days_held
                    else:
                        cash -= abs(position) * current_price
                        if trades:
                            pnl = (position_entry_price - current_price) / position_entry_price
                            trades[-1]['exit_date'] = current_date
                            trades[-1]['exit_price'] = current_price
                            trades[-1]['pnl'] = pnl
                            trades[-1]['exit_reason'] = 'REGIME_CHANGE'
                            trades[-1]['days_held'] = days_held

                    position = 0
                    position_entry_price = None
                    position_entry_date = None
                    position_side = None
                    continue

        # Generate signals based on regime
        signal = None

        if current_regime == 'BULL':
            # Buy on pullback to recent low
            window = calculate_dynamic_window(current_data['close'], len(current_data) - 1,
                                             window_short=15, window_long=40)
            recent_low = current_data['low'].rolling(window).min().iloc[-1]
            entry_zone = recent_low + (atr_multiplier * current_atr)

            if position == 0 and current_price <= entry_zone:
                signal = 'LONG'

        elif current_regime == 'BEAR':
            # Short on rally to recent high
            window = calculate_dynamic_window(current_data['close'], len(current_data) - 1,
                                             window_short=15, window_long=40)
            recent_high = current_data['high'].rolling(window).max().iloc[-1]
            entry_zone = recent_high - (atr_multiplier * current_atr)

            if position == 0 and current_price >= entry_zone:
                signal = 'SHORT'

        elif current_regime == 'RANGING':
            # Trade breakouts WITHOUT confirmation (improved)
            window = calculate_dynamic_window(current_data['close'], len(current_data) - 1,
                                             window_short=15, window_long=40)
            recent_high = current_data['high'].rolling(window).max().iloc[-1]
            recent_low = current_data['low'].rolling(window).min().iloc[-1]

            breakout_level = recent_high + (atr_multiplier * current_atr)
            breakdown_level = recent_low - (atr_multiplier * current_atr)

            # Enter immediately on breakout (no confirmation)
            if position == 0:
                if current_price > breakout_level:
                    signal = 'LONG'
                elif current_price < breakdown_level:
                    signal = 'SHORT'

        # Execute signal
        if signal == 'LONG' and position == 0:
            # Enter long
            contracts = int(initial_cash * 0.02 / (2 * current_atr) / current_price)
            contracts = max(1, contracts)

            position_value = contracts * current_price
            cash -= position_value
            position = contracts
            position_entry_price = current_price
            position_entry_date = current_date
            position_side = 'LONG'

            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': current_price,
                'size': contracts,
                'regime_smooth': current_regime
            })

        elif signal == 'SHORT' and position == 0:
            # Enter short
            contracts = int(initial_cash * 0.02 / (2 * current_atr) / current_price)
            contracts = max(1, contracts)

            position_value = contracts * current_price
            cash += position_value
            position = -contracts
            position_entry_price = current_price
            position_entry_date = current_date
            position_side = 'SHORT'

            trades.append({
                'date': current_date,
                'action': 'SHORT',
                'price': current_price,
                'size': contracts,
                'regime_smooth': current_regime
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


def print_results(metrics, title="RARS 策略回测结果"):
    """Print performance results"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
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

    # Trade details
    if metrics['trades']:
        print(f"\n{'=' * 80}")
        print("交易明细")
        print(f"{'=' * 80}")

        completed = [t for t in metrics['trades'] if 'pnl' in t]

        for i, trade in enumerate(completed, 1):
            print(f"\n交易 #{i}:")
            print(f"  日期: {trade['date'].date()} → {trade.get('exit_date', 'N/A').date() if 'exit_date' in trade else 'N/A'}")
            print(f"  操作: {trade['action']}")
            print(f"  价格: {trade['price']:.2f} → {trade.get('exit_price', 'N/A'):.2f}")
            print(f"  机制: {trade['regime_smooth']}")
            print(f"  收益率: {trade['pnl']*100:.2f}%")
            print(f"  持仓天数: {trade.get('days_held', 'N/A')}")
            print(f"  退出原因: {trade.get('exit_reason', 'N/A')}")

    # Analyze by regime
    print(f"\n{'=' * 80}")
    print("各机制交易表现")
    print(f"{'=' * 80}")

    for regime in ['BULL', 'BEAR', 'RANGING']:
        regime_trades = [t for t in metrics['trades'] if t.get('regime_smooth') == regime and 'pnl' in t]

        if len(regime_trades) > 0:
            pnls = [t['pnl'] for t in regime_trades]
            avg_pnl = np.mean(pnls) * 100
            win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
            avg_days = np.mean([t.get('days_held', 0) for t in regime_trades])

            print(f"\n{regime}:")
            print(f"  交易次数: {len(regime_trades)}")
            print(f"  平均收益率: {avg_pnl:.2f}%")
            print(f"  胜率: {win_rate:.2%}")
            print(f"  平均持仓: {avg_days:.1f}天")


def compare_strategies():
    """Compare original vs improved strategy"""
    print("\n" + "=" * 80)
    print("策略对比分析")
    print("=" * 80)

    comparison = [
        ("原版 RARS", 0.02, 0.12, -0.27, 6, 0.00),
        ("改进版 RARS", "待测试", "待测试", "待测试", "待测试", "待测试"),
    ]

    print(f"\n{'策略':<15} {'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10} {'交易次数':<10} {'胜率':<10}")
    print("-" * 80)

    for name, ret, sharpe, dd, trades, wr in comparison:
        if isinstance(ret, str):
            print(f"{name:<15} {ret:<12} {sharpe:<10} {dd:<10} {trades:<10} {wr:<10}")
        else:
            print(f"{name:<15} {ret:>10.2f}% {sharpe:>8.2f} {dd:>8.2f}% {trades:>8} {wr:>8.2%}")


def main():
    """Main function"""
    print("=" * 80)
    print("改进版 RARS 策略回测 - TA期货（2019-2025）")
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

    # Run improved backtest (generates trend-based regimes internally)
    metrics = run_improved_backtest(df_test)

    # Print results
    print_results(metrics, "改进版 RARS 策略回测结果")

    # Comparison
    compare_strategies()

    return metrics


if __name__ == '__main__':
    main()
