#!/usr/bin/env python
"""
Generate real regime labels using HMM RegimeDetector
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor
from strategies.regime.regime_detector import RegimeDetector


def main():
    """Generate regime labels using HMM"""

    print("=" * 80)
    print("使用 HMM 生成真实机制标签 - TA期货")
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

    print(f"数据加载完成: {len(daily_df)} 个交易日")
    print(f"日期范围: {daily_df.index[0].date()} 到 {daily_df.index[-1].date()}")

    # Test period for backtest
    test_start = '2019-01-01'
    test_end = '2025-03-21'

    # Use data before test for training
    train_df = daily_df.loc[:test_start].copy()
    test_df = daily_df.loc[test_start:test_end].copy()

    print(f"\n训练数据: {len(train_df)} 天 ({train_df.index[0].date()} 到 {train_df.index[-1].date()})")
    print(f"测试数据: {len(test_df)} 天 ({test_df.index[0].date()} 到 {test_df.index[-1].date()})")

    # Initialize and train HMM regime detector
    print("\n训练 HMM 机制检测器...")
    detector = RegimeDetector(
        n_states=3,
        covariance_type='full',
        random_state=42
    )

    detector.fit(train_df)
    print("HMM 训练完成")

    # Predict regimes for test period
    print("\n生成测试期机制标签...")
    regimes, probs = detector.predict(test_df)

    # Create assignments dataframe
    assignments = pd.DataFrame({
        'Date': test_df.index,
        'Regime': regimes,
        'Confidence': [probs[i, r] for i, r in enumerate(regimes)]
    })

    # Filter out invalid regimes (-1 means insufficient data)
    valid_assignments = assignments[assignments['Regime'] != -1].copy()
    invalid_count = len(assignments) - len(valid_assignments)

    if invalid_count > 0:
        print(f"\n警告: {invalid_count} 个数据点因特征计算窗口期被标记为无效")

    assignments = valid_assignments

    # Map regime numbers to names
    regime_names = {
        0: 'BEAR',
        1: 'RANGING',
        2: 'BULL'
    }

    # Analyze regime characteristics
    print("\n" + "=" * 80)
    print("机制特征分析")
    print("=" * 80)

    for regime_id in sorted(assignments['Regime'].unique()):
        regime_name = regime_names[regime_id]
        regime_data = assignments[assignments['Regime'] == regime_id]

        # Calculate returns for this regime
        regime_prices = test_df.loc[regime_data['Date'], 'close']

        if len(regime_prices) > 1:
            returns = regime_prices.pct_change().dropna()
            avg_return = returns.mean() * 100
            volatility = returns.std() * 100
            total_return = (regime_prices.iloc[-1] / regime_prices.iloc[0] - 1) * 100

            print(f"\n{regime_name} (Regime {regime_id}):")
            print(f"  交易天数: {len(regime_data)} ({len(regime_data)/len(assignments)*100:.1f}%)")
            print(f"  平均日收益: {avg_return:.3f}%")
            print(f"  日波动率: {volatility:.3f}%")
            print(f"  总收益: {total_return:.2f}%")

    # Add trading state
    assignments['Trading_State'] = assignments['Regime'].map(regime_names)

    # Save assignments
    output_dir = 'data/regimetry/reports/HMM_TA'
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/cluster_assignments.csv'
    assignments.to_csv(output_path, index=False)
    print(f"\n机制标签已保存: {output_path}")

    # Regime transitions
    print("\n" + "=" * 80)
    print("机制转换分析")
    print("=" * 80)

    assignments_sorted = assignments.sort_values('Date')
    regime_changes = assignments_sorted['Trading_State'].ne(
        assignments_sorted['Trading_State'].shift()
    ).sum()

    print(f"总机制转换次数: {regime_changes}")
    print(f"平均每个机制持续时间: {len(assignments) / regime_changes:.1f} 天")

    # Display transitions
    transitions = assignments_sorted[assignments_sorted['Trading_State'].ne(
        assignments_sorted['Trading_State'].shift()
    )][['Date', 'Trading_State']]

    print(f"\n机制转换日期:")
    for _, row in transitions.iterrows():
        print(f"  {row['Date'].date()}: {row['Trading_State']}")

    # Recent sample
    print("\n" + "=" * 80)
    print("最近30天样本")
    print("=" * 80)
    recent = assignments[['Date', 'Regime', 'Trading_State', 'Confidence']].tail(30)
    print(recent.to_string(index=False))

    return assignments, test_df


if __name__ == '__main__':
    main()
