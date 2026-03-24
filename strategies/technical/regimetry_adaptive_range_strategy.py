"""
Regimetry Adaptive Range Strategy (RARS)
"""

import pandas as pd
import numpy as np
from strategies.base.base_strategy import FuturesStrategy
from strategies.regime.regime_mapper import RegimeMapper


class RegimetryAdaptiveRangeStrategy(FuturesStrategy):
    """
    Adaptive strategy that changes behavior based on market regime
    """

    def __init__(self, instrument, start_date, end_date, config):
        super().__init__(instrument, start_date, end_date, config)

        self.lookback_days = config.get('lookback_days', 20)
        self.window_short = config.get('window_short', 10)
        self.window_long = config.get('window_long', 30)
        self.entry_buffer_atr = config.get('entry_buffer_atr', 1.0)
        self.stop_loss_atr = config.get('stop_loss_atr', 2.0)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)

        self.regime_mapper = RegimeMapper(
            lookback_days=self.lookback_days,
            threshold_multiplier=1.5
        )

        self.pending_signal = None
        self.entry_price = None
        self.entry_atr = None

    def calculate_atr(self, data, period=14):
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_dynamic_window(self, atr_series, current_idx):
        baseline_idx = max(0, current_idx - 60)
        atr_baseline = atr_series.iloc[baseline_idx:current_idx].median()
        current_atr = atr_series.iloc[current_idx]

        if current_atr > atr_baseline:
            return self.window_short
        else:
            return self.window_long

    def calculate_position_size(self, account_value, entry_price, atr):
        risk_amount = account_value * self.risk_per_trade
        stop_loss_distance = self.stop_loss_atr * atr

        position_value = risk_amount / stop_loss_distance
        contracts = int(position_value / entry_price)

        return contracts, position_value

    def generate_bull_signal(self, data, long_position):
        atr = self.calculate_atr(data)
        current_idx = len(data) - 1

        window = self.calculate_dynamic_window(atr, current_idx)

        recent_low = data['low'].rolling(window).min().iloc[-1]
        current_atr = atr.iloc[-1]
        entry_zone = recent_low + (self.entry_buffer_atr * current_atr)
        current_price = data['close'].iloc[-1]

        if long_position == 0 and current_price <= entry_zone:
            return 'LONG'

        return None

    def generate_bear_signal(self, data, short_position):
        atr = self.calculate_atr(data)
        current_idx = len(data) - 1

        window = self.calculate_dynamic_window(atr, current_idx)

        recent_high = data['high'].rolling(window).max().iloc[-1]
        current_atr = atr.iloc[-1]
        entry_zone = recent_high - (self.entry_buffer_atr * current_atr)
        current_price = data['close'].iloc[-1]

        if short_position == 0 and current_price >= entry_zone:
            return 'SHORT'

        return None

    def generate_ranging_signal(self, data, long_position, short_position, pending_signal=None):
        atr = self.calculate_atr(data)
        current_idx = len(data) - 1

        window = self.calculate_dynamic_window(atr, current_idx)

        recent_high = data['high'].rolling(window).max().iloc[-1]
        recent_low = data['low'].rolling(window).min().iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]

        if pending_signal == 'LONG_PENDING':
            breakout_level = recent_high + (self.entry_buffer_atr * current_atr)
            if current_price > breakout_level:
                return 'LONG', None
            else:
                return None, None

        if pending_signal == 'SHORT_PENDING':
            breakdown_level = recent_low - (self.entry_buffer_atr * current_atr)
            if current_price < breakdown_level:
                return 'SHORT', None
            else:
                return None, None

        if long_position == 0:
            breakout_level = recent_high + (self.entry_buffer_atr * current_atr)
            if current_price > breakout_level:
                return None, 'LONG_PENDING'

        if short_position == 0:
            breakdown_level = recent_low - (self.entry_buffer_atr * current_atr)
            if current_price < breakdown_level:
                return None, 'SHORT_PENDING'

        return None, None

    def check_stop_loss(self, position, current_price):
        if self.entry_price is None or self.entry_atr is None:
            return False

        if position['side'] == 'LONG':
            stop_loss = self.entry_price - (self.stop_loss_atr * self.entry_atr)
            return current_price < stop_loss

        elif position['side'] == 'SHORT':
            stop_loss = self.entry_price + (self.stop_loss_atr * self.entry_atr)
            return current_price > stop_loss

        return False

    def generate_signals(self, data):
        """
        Generate trading signals based on market regime

        Parameters:
            data: Market data DataFrame with OHLCV columns

        Returns:
            pd.Series: Signal series (1=LONG, -1=SHORT, 0=HOLD)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate regime
        regime = self.regime_mapper.map_regime(data)

        # Generate signals based on regime
        for i in range(len(data)):
            if i < self.lookback_days:
                continue

            current_data = data.iloc[:i+1]

            if regime == 'BULL':
                signal = self.generate_bull_signal(current_data, 0)
                if signal == 'LONG':
                    signals.iloc[i] = 1

            elif regime == 'BEAR':
                signal = self.generate_bear_signal(current_data, 0)
                if signal == 'SHORT':
                    signals.iloc[i] = -1

            elif regime == 'RANGING':
                signal, pending = self.generate_ranging_signal(
                    current_data, 0, 0, self.pending_signal
                )
                if signal == 'LONG':
                    signals.iloc[i] = 1
                elif signal == 'SHORT':
                    signals.iloc[i] = -1

                # Update pending signal
                if pending:
                    self.pending_signal = pending

        return signals
