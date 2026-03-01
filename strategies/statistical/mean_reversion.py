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

        # Check if volatility adjustment should be used
        self.use_volatility_adjustment = config.get("use_volatility_adjustment", False)

        # Initialize RiskManager if volatility adjustment enabled
        if self.use_volatility_adjustment:
            from strategies.risk.risk_manager import RiskManager
            self.risk_manager = RiskManager(config)
        else:
            self.risk_manager = None

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

    def get_position_size(self, zscore):
        """
        Calculate position size based on Z-Score deviation magnitude

        Multi-layer position sizing:
        - |z| < level1: 0% (no trade)
        - level1 ≤ |z| < level2: 50%
        - level2 ≤ |z| < level3: 75%
        - |z| ≥ level3: 100%

        Parameters:
            zscore: Current Z-Score value (can be float or Series)

        Returns:
            float or Series: Position size ratio (0.0 to 1.0)
        """
        abs_z = abs(zscore)

        if abs_z < self.level1_threshold:
            return 0.0
        elif abs_z < self.level2_threshold:
            return 0.5
        elif abs_z < self.level3_threshold:
            return 0.75
        else:
            return 1.0

    def should_exit(self, current_zscore, entry_zscore, bars_held):
        """
        Check if position should be closed

        Exit conditions (priority order):
        1. Stop loss hit (Z-Score moved against position)
        2. Z-Score reverted to mean (profit taking)
        3. Max holding period exceeded (time-based exit)

        Parameters:
            current_zscore: Current Z-Score value
            entry_zscore: Z-Score at position entry
            bars_held: Number of bars since entry

        Returns:
            bool: True if should exit, False otherwise
        """
        # Check max holding period first (time-based exit)
        if bars_held > self.max_hold_period:
            return True

        # Calculate stop loss threshold
        stop_threshold = abs(entry_zscore) * self.stop_multiplier

        # Check stop loss (price moved further against us)
        if abs(current_zscore) > stop_threshold:
            return True

        # Check reversion to mean (profit taking)
        if abs(current_zscore) < self.exit_threshold:
            return True

        return False

    def get_position_with_volatility(self, data, current_bar, base_position):
        """
        Apply volatility adjustment to position size

        Uses RiskManager to reduce position during high volatility periods.

        Parameters:
            data: DataFrame with OHLC data
            current_bar: Current bar index
            base_position: Base position ratio

        Returns:
            float: Adjusted position ratio
        """
        if self.risk_manager is not None:
            data_slice = data.iloc[:current_bar + 1]
            adjusted_ratio = self.risk_manager.calculate_volatility_adjustment(
                data_slice, base_position
            )
            return adjusted_ratio
        else:
            return base_position

    def generate_signals(self, data):
        """
        Generate trading signals based on Z-Score mean reversion

        Strategy logic:
        - Calculate Z-Score for price series
        - Short when price above mean (positive Z-Score)
        - Long when price below mean (negative Z-Score)
        - Position size based on deviation magnitude

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            pd.Series: Signal series (1=long, -1=short, 0=no position)
        """
        # Calculate Z-Score
        zscore = self.calculate_zscore(data)

        # Initialize signals Series
        signals = pd.Series(0, index=data.index)

        # Generate signals for valid Z-Scores
        valid_mask = ~pd.isna(zscore)

        for i in range(len(data)):
            if not valid_mask.iloc[i]:
                continue

            current_z = zscore.iloc[i]

            # Determine signal direction and position size
            if current_z > 0:
                # Price above mean → potential short
                position_size = self.get_position_size(current_z)
                if position_size > 0:
                    signals.iloc[i] = -1
            else:
                # Price below mean → potential long
                position_size = self.get_position_size(current_z)
                if position_size > 0:
                    signals.iloc[i] = 1

        return signals
