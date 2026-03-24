#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版组合策略: MA趋势 + 市场环境过滤

策略:
- 只在明确的牛市中使用MA(20)策略
- 震荡市和熊市空仓
- 目标: 提升夏普比率，降低回撤
"""

import pandas as pd
import numpy as np
from strategies.regime.market_regime_detector import MarketRegimeDetector
from strategies.technical.simple_trend_strategy import SimpleTrendStrategy


class FilteredMAStrategy:
    """
    过滤版MA策略

    只在趋势明确的牛市中交易
    """

    def __init__(self, instrument, start_date, end_date, config):
        self.instrument = instrument
        self.start_date = start_date
        self.end_date = end_date
        self.config = config

        # 市场环境识别器
        self.regime_detector = MarketRegimeDetector(
            adx_period=config.get('adx_period', 14),
            trend_period=config.get('trend_period', 50),
            adx_trend_threshold=config.get('adx_trend_threshold', 25),
            adx_ranging_threshold=config.get('adx_ranging_threshold', 20)
        )

        # MA策略
        self.ma_strategy = SimpleTrendStrategy(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
            config={'ma_period': config.get('ma_period', 20)}
        )

    def run_backtest(self, data):
        """运行回测"""
        initial_cash = self.config.get('initial_cash', 1000000)
        commission_rate = self.config.get('commission_rate', 0.0001)
        position_ratio = self.config.get('position_ratio', 1.0)

        # 识别市场环境
        regimes = self.regime_detector.detect_regime(data)

        # 生成MA信号
        ma_signals = self.ma_strategy.generate_signals(data)

        # 组合信号: 只在BULL时使用MA信号，否则为0
        combined_signals = pd.Series(0, index=data.index)

        for i in range(len(data)):
            if regimes.iloc[i] == 'BULL':
                combined_signals.iloc[i] = ma_signals.iloc[i]
            else:
                combined_signals.iloc[i] = 0

        # 执行回测
        return self._execute_backtest(data, combined_signals, initial_cash, commission_rate, position_ratio)

    def _execute_backtest(self, data, signals, initial_cash, commission_rate, position_ratio):
        """执行回测"""
        cash = initial_cash
        position = 0
        portfolio_values = []

        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]

            if signal == 1 and position == 0:  # 开多仓
                position_value = cash * position_ratio
                position = position_value / current_price
                cash -= position_value

            elif signal == 0 and position != 0:  # 平仓
                cash += position * current_price
                cash -= abs(position) * current_price * commission_rate
                position = 0

            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)

        portfolio_df = pd.DataFrame({
            'portfolio_value': portfolio_values
        }, index=data.index[1:])

        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()

        return portfolio_df

    def get_metrics(self, portfolio_df):
        """计算指标"""
        returns = portfolio_df['returns'].dropna()

        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        volatility = returns.std() * np.sqrt(252)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
        }

    def _calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
