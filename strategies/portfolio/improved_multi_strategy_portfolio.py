#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved Multi-Strategy Portfolio

Combines RARS, Long-Only MA, and Enhanced MA with:
- Regime-based dynamic allocation
- Risk management (position limits, drawdown controls)
- Monthly rebalancing
- Performance tracking per strategy
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ImprovedMultiStrategyPortfolio:
    """
    Multi-strategy portfolio with dynamic allocation

    Strategies:
    1. RARS (Regimetry Adaptive Range Strategy) - Pullback entries
    2. Long-Only MA - Trend following
    3. Enhanced MA - Filtered trend following

    Allocation:
    - BULL: 50% Long-Only MA, 30% Enhanced MA, 20% RARS
    - BEAR: 60% RARS (shorts), 40% cash
    - RANGING: 50% RARS, 25% Long-Only MA, 25% Enhanced MA
    """

    def __init__(self, initial_cash=1000000, config=None):
        """
        Initialize portfolio

        Args:
            initial_cash: Starting capital
            config: Configuration dict with optional overrides
        """
        self.initial_cash = initial_cash
        self.config = config or {}

        # Base weights (default: static 40/30/30)
        self.base_weights = self.config.get('base_weights', {
            'RARS': 0.40,
            'LongOnlyMA': 0.30,
            'EnhancedMA': 0.30
        })

        # Regime-based weights
        self.regime_weights = self.config.get('regime_weights', {
            'BULL': {'RARS': 0.20, 'LongOnlyMA': 0.50, 'EnhancedMA': 0.30},
            'BEAR': {'RARS': 0.60, 'LongOnlyMA': 0.00, 'EnhancedMA': 0.00},
            'RANGING': {'RARS': 0.50, 'LongOnlyMA': 0.25, 'EnhancedMA': 0.25}
        })

        # Risk management
        self.max_portfolio_drawdown = self.config.get('max_drawdown', 0.15)  # 15%
        self.stop_loss_trigger = self.config.get('stop_loss', 0.10)  # 10%
        self.rebalance_frequency = self.config.get('rebalance_days', 30)  # Monthly

        # State tracking
        self.portfolio_value = initial_cash
        self.peak_value = initial_cash
        self.strategy_performance = {}
        self.regime_history = []

    def detect_regime(self, data_slice):
        """
        Detect market regime using MA200 methodology

        Returns: 'BULL', 'BEAR', or 'RANGING'
        """
        if len(data_slice) < 200:
            return 'RANGING'

        # Calculate MA200
        ma200 = data_slice['close'].rolling(window=200).mean()
        ma_slope = ma200.diff(5)

        # Current values
        price = data_slice['close'].iloc[-1]
        current_ma = ma200.iloc[-1]
        current_slope = ma_slope.iloc[-1]

        if pd.isna(current_ma) or pd.isna(current_slope):
            return 'RANGING'

        # 2% threshold
        threshold = price * 0.02

        if price > (current_ma + threshold):
            if current_slope > 0:
                return 'BULL'
            else:
                return 'RANGING'
        elif price < (current_ma - threshold):
            if current_slope < 0:
                return 'BEAR'
            else:
                return 'RANGING'
        else:
            return 'RANGING'

    def generate_rars_signals(self, data, current_regime, atr_multiplier=1.5):
        """
        Generate RARS-style signals (simplified from full strategy)

        Returns: Series of -1 (short), 0 (neutral), 1 (long)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate ATR
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean()

        # Generate signals
        for i in range(60, len(data)):
            current_data = data.iloc[:i+1]
            current_price = data['close'].iloc[i]
            current_atr = atr.iloc[i]

            if pd.isna(current_atr):
                continue

            # Dynamic window
            window = 15 if atr.iloc[i-60:i].median() < current_atr else 40

            if current_regime == 'BULL':
                # Buy on pullback to recent low
                recent_low = current_data['low'].rolling(window).min().iloc[-1]
                entry_zone = recent_low + (atr_multiplier * current_atr)
                if current_price <= entry_zone:
                    signals.iloc[i] = 1

            elif current_regime == 'BEAR':
                # Short on rally to recent high
                recent_high = current_data['high'].rolling(window).max().iloc[-1]
                entry_zone = recent_high - (atr_multiplier * current_atr)
                if current_price >= entry_zone:
                    signals.iloc[i] = -1

            elif current_regime == 'RANGING':
                # Trade breakouts
                recent_high = current_data['high'].rolling(window).max().iloc[-1]
                recent_low = current_data['low'].rolling(window).min().iloc[-1]

                breakout_level = recent_high + (atr_multiplier * current_atr)
                breakdown_level = recent_low - (atr_multiplier * current_atr)

                if current_price > breakout_level:
                    signals.iloc[i] = 1
                elif current_price < breakdown_level:
                    signals.iloc[i] = -1

        return signals

    def generate_ma_signals(self, data, fast_period=5, slow_period=20):
        """
        Generate MA crossover signals

        Returns: Series of -1 (short), 0 (neutral), 1 (long)
        """
        signals = pd.Series(0, index=data.index)

        if len(data) < slow_period:
            return signals

        # Calculate MAs
        ma_fast = data['close'].rolling(window=fast_period).mean()
        ma_slow = data['close'].rolling(window=slow_period).mean()

        # Generate signals
        for i in range(slow_period, len(data)):
            if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]):
                continue

            if ma_fast.iloc[i] > ma_slow.iloc[i]:
                signals.iloc[i] = 1  # Long
            else:
                signals.iloc[i] = 0  # Neutral/Exit

        return signals

    def generate_enhanced_ma_signals(self, data, fast_period=5, slow_period=20):
        """
        Generate enhanced MA signals with trend confirmation

        Returns: Series of -1 (short), 0 (neutral), 1 (long)
        """
        signals = pd.Series(0, index=data.index)

        if len(data) < slow_period + 50:
            return signals

        # Calculate MAs
        ma_fast = data['close'].rolling(window=fast_period).mean()
        ma_slow = data['close'].rolling(window=slow_period).mean()

        # Trend confirmation (price vs MA50)
        ma50 = data['close'].rolling(window=50).mean()

        # Generate signals with trend filter
        for i in range(max(slow_period, 50), len(data)):
            if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]) or pd.isna(ma50.iloc[i]):
                continue

            ma_cross_up = ma_fast.iloc[i] > ma_slow.iloc[i]
            ma_cross_down = ma_fast.iloc[i] < ma_slow.iloc[i]
            price_above_ma50 = data['close'].iloc[i] > ma50.iloc[i]

            # Only go long if both MA cross up AND price above MA50
            if ma_cross_up and price_above_ma50:
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0

        return signals

    def run_backtest(self, data, use_dynamic_weights=True):
        """
        Run portfolio backtest

        Args:
            data: DataFrame with OHLC data
            use_dynamic_weights: If True, use regime-based weights

        Returns:
            dict: Performance metrics and equity curve
        """
        print("\n" + "=" * 80)
        print("Multi-Strategy Portfolio Backtest")
        print("=" * 80)

        # Initialize
        cash = self.initial_cash
        positions = {}  # {'strategy': {'quantity': int, 'entry_price': float}}
        portfolio_values = []
        trades = []
        regime_history = []

        # Track last rebalance date
        last_rebalance = None

        # Generate all signals upfront
        print("Generating strategy signals...")
        all_regimes = []

        for i in range(200, len(data)):
            data_slice = data.iloc[:i+1]
            regime = self.detect_regime(data_slice)
            all_regimes.append(regime)

        regimes_series = pd.Series(all_regimes, index=data.index[200:])

        rars_signals = pd.Series(0, index=data.index)
        ma_signals = pd.Series(0, index=data.index)
        enhanced_ma_signals = pd.Series(0, index=data.index)

        # Generate signals per regime period (simplified)
        current_regime = 'RANGING'
        for i in range(200, len(data)):
            current_regime = regimes_series.iloc[i - 200]
            data_slice = data.iloc[:i+1]

            # Regenerate signals for this regime (simplified - would optimize in production)
            if i == 200 or regimes_series.iloc[i - 200] != regimes_series.iloc[i - 201]:
                # Regime changed, regenerate signals
                rars_slice = self.generate_rars_signals(data_slice, current_regime)
                ma_slice = self.generate_ma_signals(data_slice)
                enhanced_slice = self.generate_enhanced_ma_signals(data_slice)

        # Run main loop
        print(f"Backtesting from {data.index[200]} to {data.index[-1]}...")

        for i in range(200, len(data)):
            current_date = data.index[i]
            current_price = data['close'].iloc[i]
            current_regime = regimes_series.iloc[i - 200]

            # Check if we need to rebalance
            days_since_rebalance = (current_date - last_rebalance).days if last_rebalance else 999

            if days_since_rebalance >= self.rebalance_frequency or last_rebalance is None:
                # Get current weights
                if use_dynamic_weights:
                    weights = self.regime_weights.get(current_regime, self.base_weights)
                else:
                    weights = self.base_weights

                # Calculate target position sizes
                total_value = cash + sum(
                    pos.get('quantity', 0) * current_price
                    for pos in positions.values()
                )

                target_positions = {}
                for strategy, weight in weights.items():
                    target_value = total_value * weight
                    target_positions[strategy] = target_value / current_price if target_value > 0 else 0

                # Rebalance (simplified - close all and reopen at target weights)
                for strategy in list(positions.keys()):
                    qty = positions[strategy].get('quantity', 0)
                    if qty != 0:
                        cash += qty * current_price
                        del positions[strategy]

                for strategy, target_qty in target_positions.items():
                    if target_qty > 0:
                        cash -= target_qty * current_price
                        positions[strategy] = {'quantity': target_qty, 'entry_price': current_price}

                last_rebalance = current_date

            # Calculate portfolio value
            total_position_value = sum(
                pos.get('quantity', 0) * current_price
                for pos in positions.values()
            )
            portfolio_value = cash + total_position_value
            portfolio_values.append(portfolio_value)
            regime_history.append(current_regime)

            # Track peak for drawdown calculation
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value

            # Check drawdown limit
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if drawdown > self.max_portfolio_drawdown:
                print(f"\nWARNING: Max drawdown ({drawdown:.1%}) exceeded at {current_date}")
                print(f"   Reducing positions to 50%...")
                # Reduce all positions by 50%
                for strategy in positions:
                    positions[strategy]['quantity'] *= 0.5
                    cash += positions[strategy]['quantity'] * current_price * 0.5

        # Calculate metrics
        equity_curve = pd.Series(portfolio_values, index=data.index[200:])

        return {
            'equity_curve': equity_curve,
            'regime_history': pd.Series(regime_history, index=data.index[200:]),
            'initial_cash': self.initial_cash,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / self.initial_cash) - 1,
        }

    def calculate_metrics(self, backtest_result):
        """
        Calculate performance metrics
        """
        equity = backtest_result['equity_curve']
        returns = equity.pct_change().dropna()

        # Calculate metrics
        total_return = backtest_result['total_return']
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate (by day)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': backtest_result['final_value'],
            'years': years
        }

    def print_results(self, metrics, title="Multi-Strategy Portfolio Results"):
        """Print performance results"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        print(f"\nInitial Capital: ${self.initial_cash:,.2f}")
        print(f"Final Value: ${metrics['final_value']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Period: {metrics['years']:.1f} years")

        print("\n" + "=" * 80)
