import pandas as pd
from utils.data_processor import ContinuousContractProcessor

# Load data
processor = ContinuousContractProcessor('data/raw/TA.csv')
data = processor.process(adjust_price=True)

# Check index type
print(f"Index type before filter: {type(data.index)}")
print(f"Index dtype: {data.index.dtype}")

# Try loading data for 2020
data_2020 = processor.load_data(start_date='2020-01-01', end_date='2020-12-31')
print(f"\n=== 2020年TA期货市场特征分析 ===\n")
print(f'总记录数: {len(data_2020)}')
print(f'价格范围: {data_2020["close"].min():.2f} - {data_2020["close"].max():.2f}')
print(f'年初价格: {data_2020.iloc[0]["close"]:.2f} (日期: {data_2020.index[0]})')
print(f'年末价格: {data_2020.iloc[-1]["close"]:.2f} (日期: {data_2020.index[-1]})')
print(f'年度涨跌: {(data_2020.iloc[-1]["close"] / data_2020.iloc[0]["close"] - 1) * 100:.2f}%')

# 计算价格变化
data_2020['price_change'] = data_2020['close'].pct_change()
print(f'\n收益率统计:')
print(f'日收益率均值: {data_2020["price_change"].mean() * 100:.4f}%')
print(f'日收益率标准差: {data_2020["price_change"].std() * 100:.4f}%')
print(f'最大单日涨幅: {data_2020["price_change"].max() * 100:.2f}%')
print(f'最大单日跌幅: {data_2020["price_change"].min() * 100:.2f}%')

# 检查趋势
from scipy import stats
x = range(len(data_2020))
y = data_2020['close'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f'\n=== 趋势分析 ===')
print(f'线性回归斜率: {slope:.4f}')
print(f'R²: {r_value**2:.4f}')
print(f'P值: {p_value:.2e}')

if abs(slope) > 0.5:
    direction = "强上涨" if slope > 0 else "强下跌"
    print(f'结论: 2020年TA呈现{direction}趋势行情')
elif abs(slope) > 0.1:
    direction = "温和上涨" if slope > 0 else "温和下跌"
    print(f'结论: 2020年TA呈现{direction}行情')
else:
    print(f'结论: 2020年TA呈现震荡行情')

# 分析波动性
rolling_std = data_2020['close'].rolling(20).std()
print(f'\n=== 波动性分析 ===')
print(f'20日滚动标准差均值: {rolling_std.mean():.2f}')
print(f'20日滚动标准差最大值: {rolling_std.max():.2f}')
print(f'20日滚动标准差最小值: {rolling_std.min():.2f}')

# 分析Z-Score分布
lookback = 20
data_2020['zscore'] = (data_2020['close'] - data_2020['close'].rolling(lookback).mean()) / data_2020['close'].rolling(lookback).std()
print(f'\n=== Z-Score分析（回看{lookback}天）===')
print(f'Z-Score有效样本数: {data_2020["zscore"].notna().sum()}')
print(f'Z-Score均值: {data_2020["zscore"].mean():.4f}')
print(f'Z-Score标准差: {data_2020["zscore"].std():.4f}')
print(f'Z-Score最小值: {data_2020["zscore"].min():.4f}')
print(f'Z-Score最大值: {data_2020["zscore"].max():.4f}')
print(f'Z-Score > 2的天数: {(data_2020["zscore"] > 2).sum()} ({(data_2020["zscore"] > 2).sum() / data_2020["zscore"].notna().sum() * 100:.1f}%)')
print(f'Z-Score < -2的天数: {(data_2020["zscore"] < -2).sum()} ({(data_2020["zscore"] < -2).sum() / data_2020["zscore"].notna().sum() * 100:.1f}%)')
print(f'|Z-Score| > 1.5的天数: {(abs(data_2020["zscore"]) > 1.5).sum()} ({(abs(data_2020["zscore"]) > 1.5).sum() / data_2020["zscore"].notna().sum() * 100:.1f}%)')

# 连续极端值分析
data_2020['extreme'] = abs(data_2020['zscore']) > 2
extreme_groups = data_2020['extreme'].astype(int).groupby((data_2020['extreme'] != data_2020['extreme'].shift()).cumsum()).sum()
max_consecutive = extreme_groups[extreme_groups > 0].max() if len(extreme_groups[extreme_groups > 0]) > 0 else 0
print(f'最大连续极端值天数: {max_consecutive}天')

# 均值回归速度分析
print(f'\n=== 均值回归速度分析 ===')
data_2020['above_mean'] = data_2020['zscore'] > 0
reversions = 0
total_periods = 0
for i in range(lookback, len(data_2020) - 1):
    if data_2020.iloc[i]['above_mean'] != data_2020.iloc[i+1]['above_mean']:
        reversions += 1
    total_periods += 1
print(f'均值回归次数: {reversions}')
print(f'平均回归周期: {total_periods / reversions if reversions > 0 else 0:.1f}天')
