# trading/strategies/base/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd

class FuturesStrategy(ABC):
    """期货策略基类"""

    def __init__(self, instrument, start_date, end_date, config):
        """
        初始化策略

        参数:
            instrument: 品种代码 (如 'TA', 'rb', 'm')
            start_date: 回测开始日期
            end_date: 回测结束日期
            config: 策略配置字典
        """
        self.instrument = instrument
        self.start_date = start_date
        self.end_date = end_date
        self.config = config

    @abstractmethod
    def generate_signals(self, data):
        """
        生成交易信号 (抽象方法，子类必须实现)

        参数:
            data: 市场数据DataFrame

        返回:
            pd.Series: 信号序列 (1=做多, -1=做空, 0=平仓)
        """
        pass

    def get_features(self):
        """
        获取策略所需特征

        返回:
            dict: 特征字典 {feature_name: source_column}
        """
        return {}

    def validate_signal(self, signal):
        """
        信号验证 (可选实现)

        参数:
            signal: 信号值

        返回:
            bool: 信号是否有效
        """
        return signal in [-1, 0, 1]
