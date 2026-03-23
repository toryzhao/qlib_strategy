"""
Enhanced MA Strategy with Regime Filtering

This module implements an enhanced moving average strategy that combines:
- MA(20) signal generation
- HMM-based regime filtering (only trade in bull markets)
- Dynamic position sizing based on confidence and ADX
- Multi-layered exit conditions
"""

import numpy as np
import pandas as pd
from strategies.base.base_strategy import FuturesStrategy


class EnhancedMAStrategy(FuturesStrategy):
    """
    Enhanced Moving Average Strategy with Regime Filtering.

    A long-only strategy that:
    - Uses MA(20) for baseline trend signals
    - Filters trades using HMM regime detection (only enter in bull regimes)
    - Dynamically sizes positions based on regime confidence and ADX trend strength
    - Exits on multiple conditions: MA cross, regime change, confidence drop, or stop-loss

    Parameters
    ----------
    instrument : str
        Instrument code (e.g., 'TA', 'rb', 'm')
    start_date : str
        Backtest start date
    end_date : str
        Backtest end date
    config : dict
        Strategy configuration with keys:
        - ma_period: MA period (default: 20)
        - base_position: Base position size (default: 1.0)
        - confidence_threshold: Minimum confidence to enter (default: 0.60)
        - min_position: Minimum position size (default: 0.30)
        - max_position: Maximum position size (default: 1.50)
        - use_regime_filter: Enable regime filtering (default: True)
        - stop_loss: Stop loss percentage (default: 0.10)
        - adx_period: ADX calculation period (default: 14)
    """

    def __init__(self, instrument, start_date, end_date, config):
        super().__init__(instrument, start_date, end_date, config)

        # Strategy parameters
        self.ma_period = config.get('ma_period', 20)
        self.base_position = config.get('base_position', 1.0)
        self.confidence_threshold = config.get('confidence_threshold', 0.60)
        self.min_position = config.get('min_position', 0.30)
        self.max_position = config.get('max_position', 1.50)
        self.use_regime_filter = config.get('use_regime_filter', True)
        self.stop_loss = config.get('stop_loss', 0.10)
        self.adx_period = config.get('adx_period', 14)

        # Regime detector (will be set externally)
        self.regime_detector = None

        # Track entry price for stop-loss
        self.entry_price = None

    def calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) for trend strength.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'high', 'low', 'close' columns
        period : int, optional
            ADX period (defaults to self.adx_period)

        Returns
        -------
        pd.Series
            ADX values
        """
        if period is None:
            period = self.adx_period

        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Convert to Series
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # Calculate smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def calculate_position_size(self, confidence: float, adx: float) -> float:
        """
        Calculate dynamic position size based on confidence and trend strength.

        Formula:
        - trend_multiplier scales ADX to [0.5, 1.5] range
        - position = base_position * confidence * trend_multiplier

        Parameters
        ----------
        confidence : float
            Regime confidence (0 to 1)
        adx : float
            ADX value (trend strength indicator)

        Returns
        -------
        float
            Position size clipped to [min_position, max_position]
        """
        # ADX typically ranges 0-100, with 25+ indicating trending
        # Map ADX to multiplier: weak trend (0.5x) to strong trend (1.5x)
        trend_multiplier = 0.5 + (adx / 50.0)  # ADX 0->0.5x, ADX 50->1.5x
        trend_multiplier = np.clip(trend_multiplier, 0.5, 1.5)

        # Calculate position size
        position = self.base_position * confidence * trend_multiplier

        # Clip to bounds
        position = max(self.min_position, min(self.max_position, position))

        return position

    def should_enter_market(self, regime: int, confidence: float) -> bool:
        """
        Determine if market conditions are suitable for entry.

        Entry conditions:
        - Regime must be Bull (state 2)
        - Confidence must exceed threshold

        Parameters
        ----------
        regime : int
            Current regime (0=Bear, 1=Ranging, 2=Bull)
        confidence : float
            Regime confidence level

        Returns
        -------
        bool
            True if should enter market
        """
        if not self.use_regime_filter:
            return True

        # Only enter in bull regime with sufficient confidence
        return regime == 2 and confidence >= self.confidence_threshold

    def should_exit_market(
        self,
        price: float,
        ma: float,
        regime: int,
        confidence: float
    ) -> bool:
        """
        Determine if should exit current position.

        Exit conditions (any one triggers exit):
        1. Price crosses below MA
        2. Regime changes from Bull to anything else
        3. Confidence drops below threshold
        4. Stop loss hit

        Parameters
        ----------
        price : float
            Current price
        ma : float
            Moving average value
        regime : int
            Current regime (0=Bear, 1=Ranging, 2=Bull)
        confidence : float
            Regime confidence level

        Returns
        -------
        bool
            True if should exit market
        """
        # Condition 1: Price below MA
        if price < ma:
            return True

        # Condition 2: Regime changed from Bull
        if self.use_regime_filter and regime != 2:
            return True

        # Condition 3: Confidence dropped
        if self.use_regime_filter and confidence < self.confidence_threshold:
            return True

        # Condition 4: Stop loss hit
        if self.entry_price is not None:
            if price < self.entry_price * (1 - self.stop_loss):
                return True

        return False

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with position sizes.

        Strategy logic:
        1. Calculate MA(20)
        2. Calculate ADX for trend strength
        3. If regime filter enabled, get regime and confidence
        4. Generate signals based on entry/exit conditions
        5. Calculate dynamic position sizes

        Parameters
        ----------
        data : pd.DataFrame
            Market data with 'close', 'high', 'low' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - signal: 1 (long), 0 (flat)
            - position_size: Position size for signals
        """
        # Calculate indicators
        ma = data['close'].rolling(window=self.ma_period).mean()
        adx = self.calculate_adx(data)

        # Initialize output
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['position_size'] = 0.0

        # Get regime predictions if filter is enabled
        if self.use_regime_filter and self.regime_detector is not None:
            try:
                regimes, probabilities = self.regime_detector.predict(data)

                # Extract confidence for bull regime (state 2)
                confidence = pd.Series(
                    [prob[2] if len(prob) > 2 else 0.0 for prob in probabilities],
                    index=data.index
                )
            except Exception:
                # If regime prediction fails, disable filtering
                regimes = np.full(len(data), 2)  # Assume bull regime
                confidence = pd.Series(1.0, index=data.index)
        else:
            # No regime filtering - assume always bull regime with max confidence
            regimes = np.full(len(data), 2)
            confidence = pd.Series(1.0, index=data.index)

        # Generate signals
        in_position = False

        for i in range(self.ma_period, len(data)):
            idx = data.index[i]
            current_price = data['close'].iloc[i]
            current_ma = ma.iloc[i]
            current_regime = regimes[i]
            current_confidence = confidence.iloc[i]
            current_adx = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 25.0

            if not in_position:
                # Check entry conditions
                if self.should_enter_market(current_regime, current_confidence):
                    # Also require price above MA for entry
                    if current_price > current_ma:
                        signals.loc[idx, 'signal'] = 1
                        signals.loc[idx, 'position_size'] = self.calculate_position_size(
                            current_confidence,
                            current_adx
                        )
                        self.entry_price = current_price
                        in_position = True
            else:
                # Already in position - check exit conditions
                if self.should_exit_market(current_price, current_ma, current_regime, current_confidence):
                    # Exit (signal stays 0)
                    self.entry_price = None
                    in_position = False
                else:
                    # Maintain position
                    signals.loc[idx, 'signal'] = 1
                    signals.loc[idx, 'position_size'] = self.calculate_position_size(
                        current_confidence,
                        current_adx
                    )

        return signals

    def get_features(self):
        """Get strategy features."""
        return {
            f'MA{self.ma_period}': 'close',
            'ADX': 'high'
        }
