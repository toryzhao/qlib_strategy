"""
日线级别回测脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.data_processor import ContinuousContractProcessor
from strategies.statistical.mean_reversion import MeanReversionStrategy
from executors.backtest_executor import BacktestExecutor

def resample_to_daily(data):
    """
    将分钟线数据重采样为日线数据

    Parameters:
        data: DataFrame with minute data

    Returns:
        DataFrame with daily data
    """
    # 使用日期分组，每天取最后一根K线的数据
    data['date'] = pd.to_datetime(data['datetime']).dt.date

    daily_data = data.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    # 设置日期为索引
    daily_data['datetime'] = pd.to_datetime(daily_data['date'])
    daily_data.set_index('datetime', inplace=True)
    daily_data.drop('date', axis=1, inplace=True)

    return daily_data

def main():
    print("=" * 60)
    print("日线级别均值回归策略回测")
    print("=" * 60)

    # Load data
    csv_path = 'data/raw/TA.csv'
    print(f"\n加载数据: {csv_path}")

    processor = ContinuousContractProcessor(csv_path)
    data = processor.process(adjust_price=True)

    # Filter for 2020 data
    print(f"原始数据量: {len(data)}条")
    data = data[(data['datetime'] >= '2020-01-01') & (data['datetime'] <= '2020-12-31')]
    print(f"2020年数据量: {len(data)}条")

    # Resample to daily
    print("\n重采样为日线数据...")
    daily_data = resample_to_daily(data)
    print(f"日线数据量: {len(daily_data)}条")

    # Show daily statistics
    print(f"\n日线数据统计:")
    print(f"日期范围: {daily_data.index[0].date()} 到 {daily_data.index[-1].date()}")
    print(f"价格范围: {daily_data['close'].min():.2f} - {daily_data['close'].max():.2f}")
    print(f"年初收盘: {daily_data.iloc[0]['close']:.2f}")
    print(f"年末收盘: {daily_data.iloc[-1]['close']:.2f}")
    print(f"年度涨跌: {(daily_data.iloc[-1]['close'] / daily_data.iloc[0]['close'] - 1) * 100:.2f}%")

    # Configure strategy
    config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'initial_cash': 1000000,
        'position_ratio': 0.3,
        'commission_rate': 0.0001,

        # Mean reversion parameters
        'lookback_period': 20,
        'entry_threshold': 1.5,
        'exit_threshold': 0.5,
        'max_hold_period': 50,
        'stop_multiplier': 1.5,

        # Trend filter (方案1)
        'use_trend_filter': True,
        'trend_filter_lookback': 60,
        'trend_slope_threshold': 0.005,
        'trend_r2_threshold': 0.3,

        # Dynamic position (方案3)
        'enable_dynamic_position': True,
        'trend_strength_threshold': 0.01,
        'max_position_strong_trend': 0.1,
        'max_position_weak_trend': 0.3,
    }

    # Create strategy
    print(f"\n创建策略...")
    strategy = MeanReversionStrategy('TA', '2020-01-01', '2020-12-31', config)

    # Generate signals
    print(f"生成信号...")
    signals = strategy.generate_signals(daily_data)

    print(f"\n信号统计:")
    print(f"总交易日数: {len(signals)}")
    print(f"做多信号: {(signals == 1).sum()}")
    print(f"做空信号: {(signals == -1).sum()}")
    print(f"空仓信号: {(signals == 0).sum()}")
    print(f"交易活跃度: {(signals != 0).sum() / len(signals) * 100:.1f}%")

    # Run backtest
    print(f"\n运行回测...")
    executor = BacktestExecutor(strategy, config)
    portfolio = executor.run_backtest(daily_data)

    # Get metrics
    metrics = executor.get_metrics()

    # Print results
    print("\n" + "=" * 60)
    print("回测结果（日线级别）")
    print("=" * 60)
    print(f"总收益率: {metrics['total_return']:.2%}")
    print(f"年化收益率: {metrics['annual_return']:.2%}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"年化波动率: {metrics['volatility']:.2%}")

    # Calculate additional stats
    print(f"\n交易统计:")
    if (signals != 0).sum() > 0:
        # Count trades (simplified)
        trades = 0
        prev_signal = 0
        for sig in signals:
            if sig != 0 and prev_signal == 0:
                trades += 1
            prev_signal = sig
        print(f"交易次数: {trades}")
    else:
        print(f"交易次数: 0 (全年空仓)")

    # Generate report
    output_path = 'reports/TA_mean_reversion_daily'
    print(f"\n生成报告...")
    try:
        from analyzers.performance_analyzer import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(executor.portfolio)
        analyzer.generate_report(output_path)
        print(f"报告已保存到: {output_path}")
    except Exception as e:
        print(f"生成报告时出错: {str(e)}")

    return metrics

if __name__ == '__main__':
    main()
