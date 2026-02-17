# Qlib期货交易系统

基于Qlib框架的国内期货研究和回测系统，支持螺纹钢、豆粕、PTA等品种的一分钟级别历史数据回测。

## 功能特性

- ✅ 模块化设计：策略、执行、分析各层独立
- ✅ 多策略支持：技术指标策略（双均线、MACD、布林带等）
- ✅ 完整分析：详细的性能指标和可视化报告
- ✅ 适配国内期货：支持夜盘、保证金、手续费等特性
- ✅ 命令行工具：便捷的CLI回测脚本
- ✅ 测试覆盖：单元测试和集成测试

## 支持品种

- PTA (TA)
- 螺纹钢 (rb)
- 豆粕 (m)

## 快速开始

### 安装依赖

```bash
cd trading
pip install -r requirements.txt
```

### 准备数据

将CSV格式的历史数据文件放入 `data/raw/` 目录：
- `TA.csv` - PTA主力连续合约
- `rb.csv` - 螺纹钢主力连续合约
- `m.csv` - 豆粕主力连续合约

CSV格式要求：
```csv
datetime,open,high,low,close,volume,amount,position,symbol
2006-12-18 09:26:00,8980.0,8998.0,8980.0,8998.0,7186.0,645312680.0,3238.0,TA0705
```

### 运行回测

使用命令行脚本运行回测：

```bash
python scripts/run_backtest.py \
    --instrument TA \
    --strategy ma_cross \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --fast 5 \
    --slow 20
```

参数说明：
- `--instrument`: 品种代码 (TA, rb, m)
- `--strategy`: 策略类型 (ma_cross)
- `--start`: 开始日期 (YYYY-MM-DD)
- `--end`: 结束日期 (YYYY-MM-DD)
- `--fast`: 快速均线周期（可选）
- `--slow`: 慢速均线周期（可选）

### 查看报告

回测完成后，报告会保存在 `reports/` 目录：

```
reports/
└── TA_ma_cross/
    ├── equity_curve.png          # 资金曲线
    ├── drawdown.png              # 回撤图
    └── performance_report.txt    # 性能报告
```

## 运行测试

```bash
cd trading
pytest tests/ -v
```

## 项目结构

```
trading/
├── analyzers/           # 性能分析器
├── config/              # 配置文件
├── data/                # 数据目录
│   ├── raw/            # 原始CSV数据
│   ├── processed/      # 处理后的数据
│   └── cache/          # 特征缓存
├── executors/          # 回测执行器
├── scripts/            # 命令行脚本
├── strategies/         # 交易策略
│   ├── base/          # 策略基类
│   └── technical/     # 技术指标策略
├── tests/              # 测试文件
└── utils/              # 工具函数
```

## 性能指标

系统提供以下性能指标：

- **总收益率**: 整个回测期间的总收益
- **年化收益率**: 按年计算的平均收益率
- **夏普比率**: 风险调整后的收益指标
- **最大回撤**: 最大资产回撤百分比
- **年化波动率**: 收益率的标准差

## 扩展开发

### 添加新策略

1. 在 `strategies/technical/` 创建新的策略文件
2. 继承 `FuturesStrategy` 基类
3. 实现 `generate_signals()` 方法
4. 在 `StrategyFactory` 中注册新策略

示例：

```python
from strategies.base.base_strategy import FuturesStrategy
import pandas as pd

class MyStrategy(FuturesStrategy):
    def generate_signals(self, data):
        # 实现你的交易逻辑
        signals = pd.Series(0, index=data.index)
        # ... 你的代码 ...
        return signals
```

## 技术栈

- Python 3.8+
- Qlib
- Pandas, NumPy
- Matplotlib, Seaborn
- PyYAML

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue。
