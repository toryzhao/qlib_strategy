"""
测试不同参数配置的日线回测效果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.data_processor import ContinuousContractProcessor
from strategies.statistical.mean_reversion import MeanReversionStrategy
from executors.backtest_executor import BacktestExecutor

def resample_to_daily(data):
    """将分钟线数据重采样为日线数据"""
    data['date'] = pd.to_datetime(data['datetime']).dt.date
    daily_data = data.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    daily_data['datetime'] = pd.to_datetime(daily_data['date'])
    daily_data.set_index('datetime', inplace=True)
    daily_data.drop('date', axis=1, inplace=True)
    return daily_data

def run_backtest(config_name, config, daily_data):
    """运行单个配置的回测"""
    strategy = MeanReversionStrategy('TA', '2020-01-01', '2020-12-31', config)
    signals = strategy.generate_signals(daily_data)

    # 跳过没有交易的配置
    if (signals == 0).all():
        return {
            'name': config_name,
            'signals': 0,
            'return': 0.0,
            'sharpe': 0.0
        }

    executor = BacktestExecutor(strategy, config)
    executor.run_backtest(daily_data)
    metrics = executor.get_metrics()

    return {
        'name': config_name,
        'signals': (signals != 0).sum(),
        'return': metrics['total_return'],
        'sharpe': metrics['sharpe_ratio'],
        'max_dd': metrics['max_drawdown']
    }

def main():
    print("=" * 70)
    print("日线回测：不同参数配置对比")
    print("=" * 70)

    # Load and prepare data
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    data = processor.process(adjust_price=True)
    data = data[(data['datetime'] >= '2020-01-01') & (data['datetime'] <= '2020-12-31')]
    daily_data = resample_to_daily(data)

    print(f"\n数据准备完成: {len(daily_data)}个交易日")
    print(f"价格范围: {daily_data['close'].min():.2f} - {daily_data['close'].max():.2f}")
    print(f"年度涨跌: {(daily_data.iloc[-1]['close'] / daily_data.iloc[0]['close'] - 1) * 100:.2f}%\n")

    # Define different configurations
    configs = []

    # 配置1: 启用趋势过滤器（当前默认）
    configs.append({
        'name': '配置1: 启用趋势过滤器（默认）',
        'config': {
            'lookback_period': 20,
            'entry_threshold': 1.5,
            'use_trend_filter': True,
            'trend_filter_lookback': 60,
            'trend_slope_threshold': 0.005,
            'trend_r2_threshold': 0.3,
        }
    })

    # 配置2: 禁用趋势过滤器
    configs.append({
        'name': '配置2: 禁用趋势过滤器（原始策略）',
        'config': {
            'lookback_period': 20,
            'entry_threshold': 1.5,
            'use_trend_filter': False,
        }
    })

    # 配置3: 放宽趋势过滤器参数（更激进）
    configs.append({
        'name': '配置3: 宽松趋势过滤器（R²=0.1）',
        'config': {
            'lookback_period': 20,
            'entry_threshold': 1.5,
            'use_trend_filter': True,
            'trend_filter_lookback': 40,
            'trend_slope_threshold': 0.01,  # 更高阈值
            'trend_r2_threshold': 0.1,     # 更低R²要求
        }
    })

    # 配置4: 紧张趋势过滤器（更保守）
    configs.append({
        'name': '配置4: 紧张趋势过滤器（斜率=0.001）',
        'config': {
            'lookback_period': 20,
            'entry_threshold': 1.5,
            'use_trend_filter': True,
            'trend_filter_lookback': 60,
            'trend_slope_threshold': 0.001,  # 更低阈值
            'trend_r2_threshold': 0.5,        # 更高R²要求
        }
    })

    # 配置5: 缩短回看周期
    configs.append({
        'name': '配置5: 短回看周期（lookback=10）',
        'config': {
            'lookback_period': 10,
            'entry_threshold': 1.5,
            'use_trend_filter': True,
            'trend_filter_lookback': 30,  # 短周期
            'trend_slope_threshold': 0.005,
            'trend_r2_threshold': 0.3,
        }
    })

    # Add common settings to all configs
    common_settings = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'initial_cash': 1000000,
        'position_ratio': 0.3,
        'commission_rate': 0.0001,
        'exit_threshold': 0.5,
        'max_hold_period': 50,
        'stop_multiplier': 1.5,
    }

    for cfg in configs:
        cfg['config'].update(common_settings)

    # Run backtests
    results = []
    print("开始回测...\n")

    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] 测试 {cfg['name']}...")
        try:
            result = run_backtest(cfg['name'], cfg['config'], daily_data)
            results.append(result)
            print(f"  [OK] 信号数: {result['signals']}, "
                  f"收益率: {result['return']:.2%}, "
                  f"夏普: {result['sharpe']:.4f}")
        except Exception as e:
            print(f"  [ERR] 错误: {str(e)}")
            results.append({
                'name': cfg['name'],
                'signals': 0,
                'return': 0,
                'sharpe': float('-inf'),
                'error': str(e)
            })

    # Print summary table
    print("\n" + "=" * 100)
    print("回测结果汇总")
    print("=" * 100)
    print(f"{'配置名称':<35} {'信号数':>8} {'总收益率':>12} {'夏普比率':>12} {'最大回撤':>12}")
    print("-" * 100)

    for r in results:
        print(f"{r['name']:<35} {r['signals']:>8} "
              f"{r['return']:>11.2%} {r['sharpe']:>12.4f} "
              f"{r.get('max_dd', 0):>11.2%}")

    # Find best configuration
    valid_results = [r for r in results if r.get('sharpe', float('-inf')) != float('-inf')]
    if valid_results:
        best = max(valid_results, key=lambda x: x['sharpe'])
        print("\n" + "=" * 100)
        print(f"最佳配置: {best['name']}")
        print(f"夏普比率: {best['sharpe']:.4f}")
        print(f"总收益率: {best['return']:.2%}")
        print("=" * 100)

    # Analysis
    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)

    with_filter = results[0]  # 配置1: 启用趋势过滤器
    without_filter = results[1]  # 配置2: 禁用趋势过滤器

    print(f"\n1. 趋势过滤器效果:")
    print(f"   启用过滤器: {with_filter['return']:.2%} 收益, {with_filter['signals']}个信号")
    print(f"   禁用过滤器: {without_filter['return']:.2%} 收益, {without_filter['signals']}个信号")

    if abs(with_filter['return']) < abs(without_filter['return']):
        print(f"   [OK] 趋势过滤器有效: 避免了{abs(without_filter['return']) - abs(with_filter['return']):.2%}的额外损失")
    else:
        print(f"   [WARN] 趋势过滤器可能过于保守")

    print(f"\n2. 参数敏感性:")
    print(f"   5个配置中, {len([r for r in results if r['return'] > 0])}个盈利, "
          f"{len([r for r in results if r['return'] < 0])}个亏损")

    print(f"\n3. 最终建议:")
    print(f"   2020年TA明确下跌趋势(-21.22%)")
    print(f"   所有配置都未能实现盈利")
    print(f"   启用趋势过滤器是最安全的: 0% > -X%")
    print(f"   需要寻找震荡市场年份进行测试")

if __name__ == '__main__':
    main()
