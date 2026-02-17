# trading/strategies/strategy_factory.py
from strategies.technical.ma_strategy import MAStrategy

class StrategyFactory:
    """策略工厂"""

    @staticmethod
    def create_strategy(strategy_type, config):
        """
        创建策略实例

        参数:
            strategy_type: 策略类型 ('ma_cross', 'macd', 'boll', etc.)
            config: 策略配置字典，必须包含:
                - instrument: 品种代码
                - start_date: 开始日期
                - end_date: 结束日期

        返回:
            策略实例

        异常:
            ValueError: 当策略类型不存在时
        """
        strategies = {
            'ma_cross': MAStrategy,
        }

        strategy_class = strategies.get(strategy_type)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        return strategy_class(
            instrument=config['instrument'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            config=config
        )
