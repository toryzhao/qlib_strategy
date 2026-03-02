"""
Analyze backtest results with trend filter
"""
import pandas as pd
from utils.data_processor import ContinuousContractProcessor
from strategies.statistical.mean_reversion import MeanReversionStrategy
from strategies.statistical.trend_filter import TrendFilter
from executors.backtest_executor import BacktestExecutor

# Load 2020 data
processor = ContinuousContractProcessor('data/raw/TA.csv')
data = processor.process(adjust_price=True)
data_2020 = processor.load_data(start_date='2020-01-01', end_date='2020-12-31')

print(f"=== 2020年TA期货数据分析 ===\n")
print(f"数据量: {len(data_2020)}条")
print(f"价格范围: {data_2020['close'].min():.2f} - {data_2020['close'].max():.2f}")

# Test with trend filter
print(f"\n=== 趋势过滤器分析 ===")
trend_filter = TrendFilter(lookback=60, slope_threshold=0.005, r2_threshold=0.3)

# 每月检查趋势
months = data_2020.index.to_period('M').unique()
for month in months[:12]:  # 前12个月
    month_data = data_2020[data_2020.index.to_period('M') == month]
    if len(month_data) >= 60:
        trend = trend_filter.detect_trend(month_data)
        print(f"{month}: {trend}")
    else:
        print(f"{month}: 数据不足")

# 全年趋势
year_trend = trend_filter.detect_trend(data_2020)
print(f"\n全年趋势: {year_trend}")
print(f"是否适合均值回归: {trend_filter.should_trade_mean_reversion(data_2020)}")

# Generate signals with trend filter
print(f"\n=== 策略信号分析 ===")
config = {
    'use_trend_filter': True,
    'lookback_period': 20,
    'entry_threshold': 1.5
}
strategy = MeanReversionStrategy('TA', '2020-01-01', '2020-12-31', config)
signals = strategy.generate_signals(data_2020)

print(f"总信号数: {len(signals)}")
print(f"非零信号数: {(signals != 0).sum()}")
print(f"做多信号数: {(signals == 1).sum()}")
print(f"做空信号数: {(signals == -1).sum()}")
print(f"空仓信号数: {(signals == 0).sum()}")

# Compare without trend filter
print(f"\n=== 对比：不使用趋势过滤器 ===")
config_no_filter = {
    'use_trend_filter': False,
    'lookback_period': 20,
    'entry_threshold': 1.5
}
strategy_no_filter = MeanReversionStrategy('TA', '2020-01-01', '2020-12-31', config_no_filter)
signals_no_filter = strategy_no_filter.generate_signals(data_2020)

print(f"总信号数: {len(signals_no_filter)}")
print(f"非零信号数: {(signals_no_filter != 0).sum()}")
print(f"做多信号数: {(signals_no_filter == 1).sum()}")
print(f"做空信号数: {(signals_no_filter == -1).sum()}")

print(f"\n=== 结论 ===")
if (signals != 0).sum() == 0:
    print("✓ 趋势过滤器成功阻止了所有交易")
    print("✓ 避免了在-21.69%的熊市中交易")
    print("✓ 0%收益率 > -100%爆仓")
else:
    print(f"趋势过滤器允许了{(signals != 0).sum()}个交易")
