"""
分析当前数据的时间周期
"""
import pandas as pd
from utils.data_processor import ContinuousContractProcessor

# Load data
processor = ContinuousContractProcessor('data/raw/TA.csv')
data = processor.process(adjust_price=True)
data_2020 = processor.load_data(start_date='2020-01-01', end_date='2020-12-31')

print("=== 当前数据分析 ===\n")
print(f"总数据量: {len(data_2020)}条")
print(f"数据索引类型: {type(data_2020.index)}")
print(f"日期范围: {data_2020.index[0]} 到 {data_2020.index[-1]}")

# 检查是否有重复的时间戳
duplicates = data_2020.index.duplicated()
print(f"重复时间戳数量: {duplicates.sum()}")

# 计算每分钟的平均数据量（假设是分钟线）
trading_days = len(data_2020.index.date) if hasattr(data_2020.index, 'date') else 252
bars_per_day = len(data_2020) / trading_days if trading_days > 0 else len(data_2020)
print(f"\n估算: 每个交易日约 {bars_per_day:.0f} 条数据")
print(f"推断数据级别: {'分钟线' if bars_per_day > 100 else '小时线' if bars_per_day > 10 else '日线'}")

# 查看实际的数据分布
print(f"\n=== 时间戳示例 ===")
print(data_2020.head(10))
print(data_2020.tail(10))
