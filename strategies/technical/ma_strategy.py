# trading/strategies/technical/ma_strategy.py
from strategies.base.base_strategy import FuturesStrategy
import pandas as pd

class MAStrategy(FuturesStrategy):
    """双均线策略"""

    def __init__(self, instrument, start_date, end_date, config):
        """
        初始化双均线策略

        参数:
            instrument: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            config: 配置字典，包含:
                - fast_period: 快速均线周期 (默认5)
                - slow_period: 慢速均线周期 (默认20)
        """
        super().__init__(instrument, start_date, end_date, config)
        self.fast_period = config.get('fast_period', 5)
        self.slow_period = config.get('slow_period', 20)

    def generate_signals(self, data):
        """
        生成交易信号

        策略逻辑:
        - 快速均线上穿慢速均线 → 金叉做多 (signal=1)
        - 快速均线下穿慢速均线 → 死叉做空 (signal=-1)

        参数:
            data: 市场数据DataFrame，必须包含'close'列

        返回:
            pd.Series: 信号序列
        """
        # 计算均线
        fast_ma = data['close'].rolling(self.fast_period).mean()
        slow_ma = data['close'].rolling(self.slow_period).mean()

        # 金叉死叉信号
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1   # 金叉做多
        signals[fast_ma < slow_ma] = -1  # 死叉做空

        return signals

    def get_features(self):
        """获取策略所需特征"""
        return {
            f'MA{self.fast_period}': 'close',
            f'MA{self.slow_period}': 'close'
        }
