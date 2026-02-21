import pandas as pd

class RiskManager:
    """Risk management for trading strategies"""

    def __init__(self, config):
        """
        Initialize RiskManager

        Parameters:
            config: Configuration dictionary containing:
                - trend_ma_period: Period for trend filter MA
                - atr_period: ATR calculation period
                - atr_lookback: Lookback for ATR percentile
                - volatility_threshold: Percentile threshold (0-100)
                - swing_period: Lookback for swing high/low
        """
        self.trend_ma_period = config.get('trend_ma_period', 200)
        self.atr_period = config.get('atr_period', 14)
        self.atr_lookback = config.get('atr_lookback', 100)
        self.volatility_threshold = config.get('volatility_threshold', 80)
        self.swing_period = config.get('swing_period', 20)

    def _calculate_atr(self, data):
        """
        Calculate Average True Range

        Parameters:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR using exponential moving average
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()

        return atr

    def calculate_trend_filter(self, data):
        """
        Calculate trend filter based on long-term MA

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            Series: 1 for uptrend (close > MA), -1 for downtrend, 0 for neutral/no data
        """
        if len(data) < self.trend_ma_period:
            return pd.Series(0, index=data.index)

        # Calculate long-term MA
        trend_ma = data['close'].rolling(self.trend_ma_period).mean()

        # Determine trend
        trend = pd.Series(0, index=data.index)
        trend[data['close'] > trend_ma] = 1
        trend[data['close'] < trend_ma] = -1

        return trend

    def calculate_volatility_adjustment(self, data, base_position_ratio):
        """
        Calculate volatility-adjusted position size

        Reduces position size by 50% when ATR is in top 20% of historical range

        Parameters:
            data: DataFrame with price data (must have 'high', 'low', 'close')
            base_position_ratio: Base position ratio (e.g., 0.3 for 30%)

        Returns:
            float: Adjusted position ratio
        """
        if len(data) < self.atr_lookback:
            # Not enough data, use base ratio
            return base_position_ratio

        # Calculate ATR
        atr = self._calculate_atr(data)

        # Calculate ATR percentile over lookback period
        current_atr = atr.iloc[-1]
        atr_lookback_values = atr.iloc[-self.atr_lookback:]

        # Calculate percentile rank
        percentile = (atr_lookback_values < current_atr).sum() / len(atr_lookback_values) * 100

        # Reduce position size if volatility is high
        if percentile >= self.volatility_threshold:
            return base_position_ratio * 0.5
        else:
            return base_position_ratio

    def should_exit_trailing_stop(self, data, entry_bar, current_bar, position_type):
        """
        Check if position should be closed based on trailing stop

        Uses swing high/low method: exit when price penetrates
        the highest high (short) or lowest low (long) since entry

        Parameters:
            data: DataFrame with 'close', 'high', 'low' columns
            entry_bar: Index of bar when position was entered
            current_bar: Index of current bar
            position_type: 1 for long, -1 for short

        Returns:
            bool: True if should exit, False otherwise
        """
        # Determine lookback period (bars since entry)
        bars_since_entry = current_bar - entry_bar
        lookback = min(bars_since_entry, self.swing_period)

        # Minimum 5 bars required
        if lookback < 5:
            # Use entry price as stop
            entry_price = data['close'].iloc[entry_bar]
            current_price = data['close'].iloc[current_bar]

            if position_type == 1:  # Long
                return current_price < entry_price
            else:  # Short
                return current_price > entry_price

        # Get data since entry
        data_since_entry = data.iloc[entry_bar:current_bar + 1]

        if position_type == 1:  # Long position
            # Exit if close falls below lowest low
            lowest_low = data_since_entry['low'].min()
            current_price = data['close'].iloc[current_bar]
            return current_price < lowest_low

        else:  # Short position
            # Exit if close rises above highest high
            highest_high = data_since_entry['high'].max()
            current_price = data['close'].iloc[current_bar]
            return current_price > highest_high
