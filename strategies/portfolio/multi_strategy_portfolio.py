#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多策略组合管理器

根据市场环境动态选择策略:
- 牛市: MA趋势策略 (100%仓位)
- 震荡市: 均值回归策略 (50%仓位)
- 熊市: 现金/空仓 (0%仓位)
"""

import pandas as pd
import numpy as np
from strategies.regime.market_regime_detector import MarketRegimeDetector
from strategies.technical.simple_trend_strategy import SimpleTrendStrategy
from strategies.technical.mean_reversion_regime import MeanReversionRegimeStrategy


class MultiStrategyPortfolio:
    """
    多策略组合管理器

    功能:
    1. 识别市场环境 (牛市/熊市/震荡市)
    2. 根据环境切换策略
    3. 动态调整仓位
    4. 风险控制
    """

    def __init__(self, instrument, start_date, end_date, config):
        """
        初始化组合管理器

        参数:
            instrument: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            config: 配置字典
        """
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

        # 子策略
        self.ma_strategy = SimpleTrendStrategy(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
            config={
                'ma_period': config.get('ma_period', 20),
                'position_ratio': 1.0,  # 内部100%仓位
            }
        )

        self.mr_strategy = MeanReversionRegimeStrategy(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
            config={
                'lookback_period': config.get('mr_lookback', 20),
                'z_entry_threshold': config.get('z_entry', 2.0),
                'z_exit_threshold': config.get('z_exit', 0.0),
            }
        )

        # 仓位配置
        self.bull_position_ratio = config.get('bull_position_ratio', 1.0)  # 牛市100%
        self.ranging_position_ratio = config.get('ranging_position_ratio', 0.5)  # 震荡50%
        self.bear_position_ratio = config.get('bear_position_ratio', 0.0)  # 熊市0%

    def run_backtest(self, data):
        """
        运行多策略回测

        参数:
            data: 市场数据DataFrame (OHLC)

        返回:
            DataFrame: 组合净值
        """
        initial_cash = self.config.get('initial_cash', 1000000)
        commission_rate = self.config.get('commission_rate', 0.0001)

        # 1. 识别市场环境
        regimes = self.regime_detector.detect_regime(data)

        # 2. 生成各策略信号
        ma_signals = self.ma_strategy.generate_signals(data)
        mr_signals = self.mr_strategy.generate_signals(data)

        # 3. 组合信号生成
        portfolio_signals = self._generate_portfolio_signals(
            regimes, ma_signals, mr_signals
        )

        # 4. 执行回测
        return self._execute_backtest(data, portfolio_signals, initial_cash, commission_rate)

    def _generate_portfolio_signals(self, regimes, ma_signals, mr_signals):
        """
        生成组合信号

        逻辑:
        - 牛市: 使用MA信号
        - 震荡市: 使用均值回归信号
        - 熊市: 平仓
        """
        signals = pd.Series(0, index=regimes.index)

        for i in range(len(regimes)):
            regime = regimes.iloc[i]

            if regime == 'BULL':
                # 牛市: 使用MA策略
                signals.iloc[i] = ma_signals.iloc[i]

            elif regime == 'RANGING':
                # 震荡市: 使用均值回归策略
                signals.iloc[i] = mr_signals.iloc[i]

            else:  # BEAR
                # 熊市: 空仓
                signals.iloc[i] = 0

        return signals

    def _execute_backtest(self, data, signals, initial_cash, commission_rate):
        """
        执行回测

        使用简化的回测逻辑，不考虑复杂止损
        """
        cash = initial_cash
        position = 0  # 正数=多仓，负数=空仓，0=空仓
        portfolio_values = []

        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]

            # 获取当前市场环境以确定仓位比例
            regime = self.regime_detector.detect_regime(data.iloc[:i+1]).iloc[-1]

            # 确定仓位比例
            if regime == 'BULL':
                position_ratio = self.bull_position_ratio
            elif regime == 'RANGING':
                position_ratio = self.ranging_position_ratio
            else:  # BEAR
                position_ratio = self.bear_position_ratio

            # 交易执行
            if signal == 1 and position == 0:  # 开多仓
                position_value = cash * position_ratio
                position = position_value / current_price
                cash -= position_value

            elif signal == -1 and position == 0:  # 开空仓
                position_value = cash * position_ratio
                position = -position_value / current_price
                cash -= abs(position) * current_price * commission_rate

            elif signal == 0 and position != 0:  # 平仓
                cash += position * current_price
                cash -= abs(position) * current_price * commission_rate
                position = 0

            # 计算当前资产
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)

        # 创建DataFrame
        portfolio_df = pd.DataFrame({
            'portfolio_value': portfolio_values
        }, index=data.index[1:])

        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()

        return portfolio_df

    def get_metrics(self, portfolio_df):
        """
        计算组合指标
        """
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

    def get_regime_analysis(self, data):
        """
        获取市场环境分析

        返回:
            dict: 环境统计信息
        """
        regimes = self.regime_detector.detect_regime(data)
        stats = self.regime_detector.get_regime_statistics(data)

        return {
            'regimes': regimes,
            'statistics': stats,
        }
