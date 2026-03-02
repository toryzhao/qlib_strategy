"""
Trend Filter Module - Detect market trend using linear regression
"""
import numpy as np
from scipy import stats


class TrendFilter:
    """Detect market trends to avoid mean reversion in trending markets"""

    def __init__(self, lookback=60, slope_threshold=0.005, r2_threshold=0.3):
        """
        Initialize Trend Filter

        Parameters:
            lookback: Number of bars to analyze for trend (default 60)
            slope_threshold: Minimum slope magnitude to consider trending (default 0.005)
            r2_threshold: Minimum R² to consider trend significant (default 0.3)
        """
        self.lookback = lookback
        self.slope_threshold = slope_threshold
        self.r2_threshold = r2_threshold

    def detect_trend(self, data):
        """
        Detect market trend using linear regression

        Parameters:
            data: DataFrame with 'close' column (at least lookback bars)

        Returns:
            str: 'uptrend', 'downtrend', or 'sideways'
        """
        if len(data) < self.lookback:
            return 'sideways'  # Not enough data

        # Get recent price data
        recent_prices = data['close'].tail(self.lookback).values

        # Perform linear regression: price = slope * time + intercept
        x = np.arange(len(recent_prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)

        # Calculate R²
        r_squared = r_value ** 2

        # Determine trend
        if r_squared < self.r2_threshold:
            # Low correlation = sideways/choppy market
            return 'sideways'
        elif slope > self.slope_threshold:
            # Positive slope with good correlation = uptrend
            return 'uptrend'
        elif slope < -self.slope_threshold:
            # Negative slope with good correlation = downtrend
            return 'downtrend'
        else:
            # Shallow slope = sideways
            return 'sideways'

    def get_trend_strength(self, data):
        """
        Get trend strength (absolute slope value)

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            float: Trend strength (higher = stronger trend)
        """
        if len(data) < self.lookback:
            return 0.0

        recent_prices = data['close'].tail(self.lookback).values
        x = np.arange(len(recent_prices))
        slope, _, r_value, _, _ = stats.linregress(x, recent_prices)

        # Return absolute slope adjusted by R²
        r_squared = r_value ** 2
        return abs(slope) * r_squared

    def should_trade_mean_reversion(self, data):
        """
        Check if conditions are suitable for mean reversion strategy

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            bool: True if sideways market (suitable for mean reversion)
        """
        trend = self.detect_trend(data)
        return trend == 'sideways'

    def get_trend_signal(self, data):
        """
        Get trend direction as signal

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            int: 1 (uptrend), -1 (downtrend), 0 (sideways)
        """
        trend = self.detect_trend(data)
        trend_map = {
            'uptrend': 1,
            'downtrend': -1,
            'sideways': 0
        }
        return trend_map.get(trend, 0)
