# trading/strategies/technical/ma_strategy.py
from strategies.base.base_strategy import FuturesStrategy
from strategies.risk.risk_manager import RiskManager
import pandas as pd

class MAStrategy(FuturesStrategy):
    """双均线策略with risk management"""

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
                - trend_ma_period: 趋势过滤MA周期 (默认200)
                - atr_period: ATR周期 (默认14)
                - atr_lookback: ATR回看期 (默认100)
                - volatility_threshold: 波动率阈值 (默认80)
                - swing_period: 摆荡周期 (默认20)
        """
        super().__init__(instrument, start_date, end_date, config)
        self.fast_period = config.get("fast_period", 5)
        self.slow_period = config.get("slow_period", 20)

        # Check if risk management is enabled
        self.use_risk_management = any(key in config for key in [
            'trend_ma_period', 'atr_period', 'atr_lookback',
            'volatility_threshold', 'swing_period'
        ])

        # Initialize RiskManager only if risk management is enabled
        if self.use_risk_management:
            self.risk_manager = RiskManager(config)
        else:
            self.risk_manager = None

    def generate_signals(self, data):
        """
        生成交易信号 (可选趋势过滤器)

        策略逻辑:
        - 计算快速MA和慢速MA
        - 如果启用风险管理，应用趋势过滤器 (200-MA)
        - 只在趋势方向上交易

        参数:
            data: 市场数据DataFrame，必须包含"close"列

        返回:
            pd.Series: 信号序列 (1=做多, -1=做空, 0=无信号)
        """
        # 计算均线
        fast_ma = data["close"].rolling(self.fast_period).mean()
        slow_ma = data["close"].rolling(self.slow_period).mean()

        # 生成原始信号
        raw_signals = pd.Series(0, index=data.index)
        raw_signals[fast_ma > slow_ma] = 1   # 金叉做多
        raw_signals[fast_ma < slow_ma] = -1  # 死叉做空

        # 应用趋势过滤器 (如果启用风险管理)
        if self.use_risk_management and self.risk_manager:
            trend = self.risk_manager.calculate_trend_filter(data)
            signals = pd.Series(0, index=data.index)
            signals[(raw_signals == 1) & (trend == 1)] = 1   # 做多且上升趋势
            signals[(raw_signals == -1) & (trend == -1)] = -1  # 做空且下降趋势
            return signals
        else:
            return raw_signals

    def get_features(self):
        """获取策略所需特征"""
        return {
            f"MA{self.fast_period}": "close",
            f"MA{self.slow_period}": "close"
        }
