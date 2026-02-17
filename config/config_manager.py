# trading/config/config_manager.py
import yaml
import os

class ConfigManager:
    """Configuration manager for the trading system"""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get_instrument_config(self, instrument):
        """Get instrument-specific configuration"""
        return self.config['instruments'].get(instrument, {})

    def get_data_config(self):
        """Get data configuration"""
        return self.config.get('data', {})

    def get_backtest_config(self):
        """Get backtest configuration"""
        return self.config.get('backtest', {})

    def get_qlib_config(self):
        """Get Qlib configuration"""
        return self.config.get('qlib', {})
