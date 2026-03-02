import pandas as pd
from utils.data_processor import ContinuousContractProcessor

# Load data
processor = ContinuousContractProcessor('data/raw/TA.csv')
data = processor.process(adjust_price=True)
data_2020 = data[(data.index >= '2020-01-01') & (data.index <= '2020-12-31')].copy()

print(f'=== 2020年TA期货市场特征分析 ===\n')
print(f'总记录数: {len(data_2020)}')
print(f'价格范围: {data_2020["close"].min():.2f} - {data_2020["close"].max():.2f}')
print(f'年初价格: {data_2020.iloc[0]["close"]:.2f}')
print(f'年末价格: {data_2020.iloc[-1]["close"]:.2f}')
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
print(f'Z-Score均值: {data_2020["zscore"].mean():.4f}')
print(f'Z-Score标准差: {data_2020["zscore"].std():.4f}')
print(f'Z-Score最小值: {data_2020["zscore"].min():.4f}')
print(f'Z-Score最大值: {data_2020["zscore"].max():.4f}')
print(f'Z-Score > 2的天数: {(data_2020["zscore"] > 2).sum()}')
print(f'Z-Score < -2的天数: {(data_2020["zscore"] < -2).sum()}')

# 连续极端值分析
data_2020['extreme'] = abs(data_2020['zscore']) > 2
consecutive_extreme = data_2020['extreme'].astype(int).groupby((data_2020['extreme'] != data_2020['extreme'].shift()).cumsum()).sum()
max_consecutive = consecutive_extreme[consecutive_extreme > 0].max() if len(consecutive_extreme[consecutive_extreme > 0]) > 0 else 0
print(f'最大连续极端天数: {max_consecutive}天')
