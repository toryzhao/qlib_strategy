#!/usr/bin/env python
"""
Analyze regime labels and map to trading states
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor
from strategies.regime.regime_mapper import RegimeMapper


def main():
    """Analyze regime labels and map to trading states"""

    print("=" * 80)
    print("机制标签分析 - TA期货")
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

    # Load cluster assignments
    print("\n加载机制标签...")
    assignments = pd.read_csv('data/regimetry/reports/TA/cluster_assignments.csv')
    assignments['Date'] = pd.to_datetime(assignments['Date'])

    print(f"机制标签加载完成: {len(assignments)} 个数据点")

    # Initialize regime mapper
    print("\n初始化机制映射器...")
    mapper = RegimeMapper(lookback_days=20, threshold_multiplier=1.5)

    # Map all clusters to trading states
    print("\n映射聚类到交易状态...")
    cluster_states = mapper.map_all_clusters(assignments, daily_df)

    print("\n" + "=" * 80)
    print("聚类 -> 交易状态映射")
    print("=" * 80)

    for cluster_id in sorted(cluster_states.keys()):
        state = cluster_states[cluster_id]
        print(f"聚类 {cluster_id}: {state}")

    # Add trading state to assignments
    assignments['Trading_State'] = assignments['Cluster_ID'].map(cluster_states)

    # Analyze trading states
    print("\n" + "=" * 80)
    print("交易状态分布")
    print("=" * 80)

    state_counts = assignments['Trading_State'].value_counts()
    for state, count in state_counts.items():
        pct = count / len(assignments) * 100
        print(f"{state}: {count} 天 ({pct:.1f}%)")

    # Calculate statistics by trading state
    print("\n" + "=" * 80)
    print("各交易状态的收益统计")
    print("=" * 80)

    assignments = assignments.merge(
        daily_df[['close']].reset_index(),
        left_on='Date',
        right_on='datetime',
        how='inner'
    )

    for state in ['BULL', 'BEAR', 'RANGING']:
        state_data = assignments[assignments['Trading_State'] == state]

        if len(state_data) > 1:
            returns = state_data['close'].pct_change().dropna()
            avg_daily_return = returns.mean() * 100
            volatility = returns.std() * 100
            total_return = (state_data['close'].iloc[-1] / state_data['close'].iloc[0] - 1) * 100

            print(f"\n{state}:")
            print(f"  平均日收益: {avg_daily_return:.3f}%")
            print(f"  日波动率: {volatility:.3f}%")
            print(f"  总收益: {total_return:.2f}%")
            print(f"  交易天数: {len(state_data)}")

    # Save enriched assignments
    output_path = 'data/regimetry/reports/TA/cluster_assignments_with_states.csv'
    assignments[['Date', 'Cluster_ID', 'Trading_State']].to_csv(output_path, index=False)
    print(f"\n增强版机制标签已保存: {output_path}")

    # Display recent sample
    print("\n" + "=" * 80)
    print("最近30天样本（带交易状态）")
    print("=" * 80)
    recent_sample = assignments[['Date', 'Cluster_ID', 'Trading_State']].tail(30)
    print(recent_sample.to_string(index=False))

    # Analyze regime transitions
    print("\n" + "=" * 80)
    print("机制转换分析")
    print("=" * 80)

    assignments_sorted = assignments.sort_values('Date')
    regime_changes = assignments_sorted['Trading_State'].ne(
        assignments_sorted['Trading_State'].shift()
    ).sum()

    print(f"\n总机制转换次数: {regime_changes}")
    print(f"平均每个机制持续时间: {len(assignments) / regime_changes:.1f} 天")

    return assignments


if __name__ == '__main__':
    main()
