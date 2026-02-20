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
