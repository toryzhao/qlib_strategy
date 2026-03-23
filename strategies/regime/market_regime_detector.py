#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场环境识别器

识别当前市场处于：
- 牛市 (Bull): 强趋势向上
- 熊市 (Bear): 强趋势向下
- 震荡市 (Ranging): 横盘震荡

使用指标:
- ADX: 趋势强度 (>25为趋势, <20为震荡)
- MA斜率: 趋势方向
- 价格相对MA位置
"""

import pandas as pd
import numpy as np


class MarketRegimeDetector:
    """
    市场环境识别器

    识别三种市场状态:
    1. BULL: 牛市 - 使用趋势跟随策略
    2. BEAR: 熊市 - 使用防御或空仓
    3. RANGING: 震荡市 - 使用均值回归策略
    """

    def __init__(self,
                 adx_period=14,
                 trend_period=50,
                 adx_trend_threshold=25,
                 adx_ranging_threshold=20):
        """
        初始化识别器

        参数:
            adx_period: ADX计算周期
            trend_period: 趋势判断MA周期
            adx_trend_threshold: ADX趋势阈值 (>=此值为趋势市)
            adx_ranging_threshold: ADX震荡阈值 (<=此值为震荡市)
        """
        self.adx_period = adx_period
        self.trend_period = trend_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_ranging_threshold = adx_ranging_threshold

    def detect_regime(self, data):
        """
        识别市场环境

        参数:
            data: DataFrame with OHLC columns

        返回:
            pd.Series: 'BULL', 'BEAR', 或 'RANGING'
        """
        close = data['close']
        high = data['high']
        low = data['low']

        # 1. 计算ADX (趋势强度)
        adx = self._calculate_adx(high, low, close, self.adx_period)

        # 2. 计算MA斜率 (趋势方向)
        ma = close.rolling(self.trend_period).mean()
        ma_slope = ma.diff(5)  # 5日MA斜率

        # 3. 计算价格相对MA位置
        price_vs_ma = (close - ma) / ma  # 偏离度

        # 初始化regime系列
        regimes = pd.Series('RANGING', index=data.index)

        # 判断逻辑
        for i in range(len(data)):
            current_adx = adx.iloc[i]
            current_slope = ma_slope.iloc[i]
            current_price_vs_ma = price_vs_ma.iloc[i]

            # 趋势市判断
            if current_adx >= self.adx_trend_threshold:
                # ADX高，确定为趋势市
                if current_slope > 0 and current_price_vs_ma > 0:
                    regimes.iloc[i] = 'BULL'
                elif current_slope < 0 and current_price_vs_ma < 0:
                    regimes.iloc[i] = 'BEAR'
                else:
                    # ADX高但方向不明确，用价格位置判断
                    if current_price_vs_ma > 0.02:  # 价格高于MA 2%
                        regimes.iloc[i] = 'BULL'
                    elif current_price_vs_ma < -0.02:  # 价格低于MA 2%
                        regimes.iloc[i] = 'BEAR'
                    else:
                        regimes.iloc[i] = 'RANGING'

            elif current_adx <= self.adx_ranging_threshold:
                # ADX低，确定为震荡市
                regimes.iloc[i] = 'RANGING'

            else:
                # ADX在中间区间，结合其他指标判断
                if abs(current_price_vs_ma) < 0.01:  # 价格接近MA
                    regimes.iloc[i] = 'RANGING'
                elif current_slope > 0:
                    regimes.iloc[i] = 'BULL'
                else:
                    regimes.iloc[i] = 'BEAR'

        return regimes

    def get_regime_signal(self, data):
        """
        获取市场环境信号 (数值形式)

        返回:
            pd.Series: 1 (牛市), 0 (震荡), -1 (熊市)
        """
        regimes = self.detect_regime(data)

        # 转换为数值信号
        signal_map = {'BULL': 1, 'RANGING': 0, 'BEAR': -1}
        signals = regimes.map(signal_map)

        return signals

    def _calculate_adx(self, high, low, close, period=14):
        """
        计算ADX (Average Directional Index)

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期

        返回:
            pd.Series: ADX值
        """
        # 计算+DM和-DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        # 只保留正向变动
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > -plus_dm), 0)

        # 计算TR (True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 平滑计算
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
        tr_smooth = tr.ewm(span=period, adjust=False).mean()

        # 计算+DI和-DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # 计算DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        # 计算ADX
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def get_regime_statistics(self, data):
        """
        获取市场环境统计信息

        返回:
            dict: 包含各环境占比的统计
        """
        regimes = self.detect_regime(data)

        stats = {
            'bull_pct': (regimes == 'BULL').sum() / len(regimes),
            'bear_pct': (regimes == 'BEAR').sum() / len(regimes),
            'ranging_pct': (regimes == 'RANGING').sum() / len(regimes),
            'total_days': len(regimes)
        }

        return stats
