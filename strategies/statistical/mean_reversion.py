"""
Mean Reversion Strategy using Z-Score
"""
from strategies.base.base_strategy import FuturesStrategy
import pandas as pd
import numpy as np


class MeanReversionStrategy(FuturesStrategy):
    """Z-Score based mean reversion strategy for range-bound markets"""

    def __init__(self, instrument, start_date, end_date, config):
        """
        Initialize Mean Reversion Strategy

        Parameters:
            instrument: Instrument code
            start_date: Start date
            end_date: End date
            config: Configuration dictionary containing:
                - lookback_period: Rolling window for Z-Score calculation
                - entry_threshold: Z-Score threshold for entry
                - level1_threshold: 50% position threshold
                - level2_threshold: 75% position threshold
                - level3_threshold: 100% position threshold
                - exit_threshold: Z-Score threshold for exit
                - max_hold_period: Maximum holding period in bars
                - stop_multiplier: Stop loss multiplier
        """
        super().__init__(instrument, start_date, end_date, config)
        self.lookback_period = config.get("lookback_period", 20)
        self.entry_threshold = config.get("entry_threshold", 1.5)
        self.level1_threshold = config.get("level1_threshold", 1.5)
        self.level2_threshold = config.get("level2_threshold", 2.0)
        self.level3_threshold = config.get("level3_threshold", 2.5)
        self.exit_threshold = config.get("exit_threshold", 0.5)
        self.max_hold_period = config.get("max_hold_period", 50)
        self.stop_multiplier = config.get("stop_multiplier", 1.5)

    def calculate_zscore(self, data):
        """
        Calculate Z-Score for price series

        Z-Score = (Price - Rolling Mean) / Rolling Std Dev

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            Series: Z-Score values (NaN for insufficient data)
        """
        close = data['close']
        rolling_mean = close.rolling(self.lookback_period).mean()
        rolling_std = close.rolling(self.lookback_period).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (close - rolling_mean) / rolling_std

        # Clip extreme values
        zscore = zscore.clip(-5, 5)

        return zscore

    def generate_signals(self, data):
        """
        Generate trading signals based on Z-Score mean reversion

        Returns DataFrame with:
        - signal: 1 (long), -1 (short), 0 (no position)
        - target_position: 0.0 to 1.0 (position size ratio)

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            DataFrame: Signals and target positions
        """
        # Placeholder implementation - will be fully implemented later
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['target_position'] = 0.0
        return signals
