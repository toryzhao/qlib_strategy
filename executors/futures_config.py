# trading/executors/futures_config.py
class FuturesBacktestConfig:
    """期货回测配置"""

    @staticmethod
    def get_default_config():
        """默认配置"""
        return {
            'initial_cash': 1000000,   # 初始资金100万
            'position_ratio': 0.3,     # 单个品种仓位比例30%
            'max_position_ratio': 0.8, # 最大总仓位80%
            'commission_rate': 0.0001, # 手续费率万分之一
            'slippage': 0,             # 滑点（跳数）
        }

    @staticmethod
    def get_instrument_config(instrument):
        """获取品种特定配置"""
        configs = {
            'TA': {
                'contract_size': 5,
                'margin_rate': 0.08,
                'commission_rate': 0.0001,
            },
            'rb': {
                'contract_size': 10,
                'margin_rate': 0.10,
                'commission_rate': 0.0001,
            },
            'm': {
                'contract_size': 10,
                'margin_rate': 0.07,
                'commission_rate': 0.0001,
            }
        }
        return configs.get(instrument, {})
