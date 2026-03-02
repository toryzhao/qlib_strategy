"""
测试均值回归策略在2021-2023年的表现
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

def analyze_year(year_data, year):
    """分析单一年份的市场特征"""
    print(f"\n{'='*70}")
    print(f"{year}年市场分析")
    print(f"{'='*70}")

    print(f"交易日数: {len(year_data)}")
    print(f"日期范围: {year_data.index[0].date()} 到 {year_data.index[-1].date()}")
    print(f"价格范围: {year_data['close'].min():.2f} - {year_data['close'].max():.2f}")
    print(f"年初价格: {year_data.iloc[0]['close']:.2f}")
    print(f"年末价格: {year_data.iloc[-1]['close']:.2f}")

    yearly_return = (year_data.iloc[-1]['close'] / year_data.iloc[0]['close'] - 1) * 100
    print(f"年度涨跌: {yearly_return:.2f}%")

    # 计算趋势（使用线性回归）
    from scipy import stats
    x = range(len(year_data))
    y = year_data['close'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    print(f"\n趋势分析:")
    print(f"线性回归斜率: {slope:.6f}")
    print(f"R-squared (拟合度): {r_squared:.4f}")

    if abs(slope) > 0.5 and r_squared > 0.3:
        trend = "强趋势" + ("(上涨)" if slope > 0 else "(下跌)")
    elif abs(slope) > 0.1 and r_squared > 0.1:
        trend = "温和趋势" + ("(上涨)" if slope > 0 else "(下跌)")
    else:
        trend = "震荡/无明显趋势"

    print(f"市场状态: {trend}")

    return {
        'return': yearly_return,
        'slope': slope,
        'r_squared': r_squared,
        'trend': trend
    }

def backtest_year(year, data, config_with_filter, config_without_filter):
    """对单一年份进行回测对比"""
    print(f"\n{'='*70}")
    print(f"{year}年回测")
    print(f"{'='*70}")

    results = {}

    # 配置1: 启用趋势过滤器
    print(f"\n[配置1] 启用趋势过滤器...")
    try:
        strategy1 = MeanReversionStrategy('TA', f'{year}-01-01', f'{year}-12-31', config_with_filter)
        signals1 = strategy1.generate_signals(data)

        if (signals1 != 0).sum() == 0:
            print(f"  全年空仓，0个交易信号")
            results['with_filter'] = {'signals': 0, 'return': 0.0, 'sharpe': 0.0, 'dd': 0.0}
        else:
            print(f"  产生{(signals1 != 0).sum()}个交易信号，运行回测...")
            executor1 = BacktestExecutor(strategy1, config_with_filter)
            executor1.run_backtest(data)
            metrics1 = executor1.get_metrics()
            results['with_filter'] = {
                'signals': (signals1 != 0).sum(),
                'return': metrics1['total_return'],
                'sharpe': metrics1['sharpe_ratio'],
                'dd': metrics1['max_drawdown']
            }
    except Exception as e:
        print(f"  错误: {str(e)}")
        results['with_filter'] = {'signals': 0, 'return': 0.0, 'sharpe': float('-inf'), 'dd': 0.0}

    # 配置2: 禁用趋势过滤器
    print(f"\n[配置2] 禁用趋势过滤器...")
    try:
        strategy2 = MeanReversionStrategy('TA', f'{year}-01-01', f'{year}-12-31', config_without_filter)
        signals2 = strategy2.generate_signals(data)

        if (signals2 != 0).sum() == 0:
            print(f"  全年空仓，0个交易信号")
            results['without_filter'] = {'signals': 0, 'return': 0.0, 'sharpe': 0.0, 'dd': 0.0}
        else:
            print(f"  产生{(signals2 != 0).sum()}个交易信号，运行回测...")
            executor2 = BacktestExecutor(strategy2, config_without_filter)
            executor2.run_backtest(data)
            metrics2 = executor2.get_metrics()
            results['without_filter'] = {
                'signals': (signals2 != 0).sum(),
                'return': metrics2['total_return'],
                'sharpe': metrics2['sharpe_ratio'],
                'dd': metrics2['max_drawdown']
            }
    except Exception as e:
        print(f"  错误: {str(e)}")
        results['without_filter'] = {'signals': 0, 'return': 0.0, 'sharpe': float('-inf'), 'dd': 0.0}

    return results

def main():
    print("=" * 70)
    print("均值回归策略：2021-2023年回测")
    print("=" * 70)

    # Load data
    csv_path = 'data/raw/TA.csv'
    print(f"\n加载数据: {csv_path}")

    processor = ContinuousContractProcessor(csv_path)
    data = processor.process(adjust_price=True)
    print(f"原始数据量: {len(data)}条")

    # Common config
    base_config = {
        'instrument': 'TA',
        'initial_cash': 1000000,
        'position_ratio': 0.3,
        'commission_rate': 0.0001,
        'lookback_period': 20,
        'entry_threshold': 1.5,
        'exit_threshold': 0.5,
        'max_hold_period': 50,
        'stop_multiplier': 1.5,
    }

    # Config with trend filter
    config_with_filter = base_config.copy()
    config_with_filter.update({
        'use_trend_filter': True,
        'trend_filter_lookback': 60,
        'trend_slope_threshold': 0.005,
        'trend_r2_threshold': 0.3,
        'enable_dynamic_position': True,
        'trend_strength_threshold': 0.01,
        'max_position_strong_trend': 0.1,
        'max_position_weak_trend': 0.3,
    })

    # Config without trend filter
    config_without_filter = base_config.copy()
    config_without_filter.update({
        'use_trend_filter': False,
    })

    # Test each year
    years = [2021, 2022, 2023]
    all_results = {}

    for year in years:
        # Filter for this year
        year_data = data[(data['datetime'] >= f'{year}-01-01') & (data['datetime'] <= f'{year}-12-31')]

        if len(year_data) == 0:
            print(f"\n[WARNING] {year}年无数据，跳过")
            continue

        # Resample to daily
        daily_data = resample_to_daily(year_data)

        # Analyze market
        market_analysis = analyze_year(daily_data, year)

        # Backtest
        backtest_results = backtest_year(year, daily_data, config_with_filter, config_without_filter)

        all_results[year] = {
            'market': market_analysis,
            'backtest': backtest_results
        }

    # Print summary
    print(f"\n\n{'='*100}")
    print("综合对比报告")
    print(f"{'='*100}")

    print(f"\n{'年份':<8} {'市场涨跌':<12} {'市场状态':<20} {'启用过滤器':<20} {'禁用过滤器':<20}")
    print("-" * 100)

    for year in years:
        if year not in all_results:
            continue

        r = all_results[year]
        m = r['market']
        b = r['backtest']

        with_filter = f"{b['with_filter']['return']:>6.2%} ({b['with_filter']['signals']}信号)"
        without_filter = f"{b['without_filter']['return']:>6.2%} ({b['without_filter']['signals']}信号)"

        print(f"{year:<8} {m['return']:>8.2f}% {m['trend']:<20} {with_filter:<20} {without_filter:<20}")

    # Detailed analysis
    print(f"\n\n{'='*100}")
    print("详细分析")
    print(f"{'='*100}")

    for year in years:
        if year not in all_results:
            continue

        r = all_results[year]
        m = r['market']
        b = r['backtest']

        print(f"\n{'='*70}")
        print(f"{year}年详细报告")
        print(f"{'='*70}")

        print(f"\n市场特征:")
        print(f"  年度涨跌: {m['return']:.2f}%")
        print(f"  线性斜率: {m['slope']:.6f}")
        print(f"  R²拟合度: {m['r_squared']:.4f}")
        print(f"  市场状态: {m['trend']}")

        print(f"\n配置1（启用趋势过滤器）:")
        print(f"  交易信号: {b['with_filter']['signals']}个")
        print(f"  总收益率: {b['with_filter']['return']:.2f}%")
        print(f"  夏普比率: {b['with_filter']['sharpe']:.4f}")
        print(f"  最大回撤: {b['with_filter']['dd']:.2f}%")

        print(f"\n配置2（禁用趋势过滤器）:")
        print(f"  交易信号: {b['without_filter']['signals']}个")
        print(f"  总收益率: {b['without_filter']['return']:.2f}%")
        print(f"  夏普比率: {b['without_filter']['sharpe']:.4f}")
        print(f"  最大回撤: {b['without_filter']['dd']:.2f}%")

        # Calculate filter benefit
        if b['without_filter']['return'] < 0:
            benefit = abs(b['without_filter']['return']) - abs(b['with_filter']['return'])
            print(f"\n趋势过滤器效果: 避免{benefit:.2f}%损失")

    # Find best year for mean reversion
    print(f"\n\n{'='*100}")
    print("最适合均值回归的年份")
    print(f"{'='*100}")

    best_years = []
    for year in years:
        if year not in all_results:
            continue
        r = all_results[year]

        # 判断标准：震荡市场 + 禁用过滤器有正向收益
        if ('震荡' in r['market']['trend'] and
            r['backtest']['without_filter']['return'] > 0):
            best_years.append(year)

    if best_years:
        print(f"\n发现适合的年份: {best_years}")
        for year in best_years:
            r = all_results[year]
            print(f"\n{year}年:")
            print(f"  市场状态: {r['market']['trend']}")
            print(f"  禁用过滤器收益率: {r['backtest']['without_filter']['return']:.2f}%")
            print(f"  推荐配置: 禁用趋势过滤器")
    else:
        print(f"\n未发现明显适合均值回归的年份")
        print(f"建议: 继续测试其他品种或寻找更长的震荡期")

if __name__ == '__main__':
    main()
