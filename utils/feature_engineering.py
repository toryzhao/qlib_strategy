# trading/utils/feature_engineering.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    """特征工程工具类"""

    @staticmethod
    def add_technical_features(df):
        """
        添加技术指标特征

        参数:
            df: 原始数据DataFrame

        返回:
            添加了技术指标的DataFrame
        """

        # 价格特征
        df['return_1m'] = df['close'].pct_change()
        df['return_5m'] = df['close'].pct_change(5)
        df['volatility_20'] = df['close'].rolling(20).std()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # 移动平均
        for period in [5, 10, 20, 60, 120]:
            df[f'MA{period}'] = df['close'].rolling(period).mean()
            df[f'EMA{period}'] = df['close'].ewm(span=period).mean()

        # MACD
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # 布林带
        df['BOLL_mid'] = df['close'].rolling(20).mean()
        df['BOLL_std'] = df['close'].rolling(20).std()
        df['BOLL_upper'] = df['BOLL_mid'] + 2 * df['BOLL_std']
        df['BOLL_lower'] = df['BOLL_mid'] - 2 * df['BOLL_std']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # 成交量特征
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        df['volume_std'] = df['volume'].rolling(20).std()

        return df

    @staticmethod
    def add_time_features(df):
        """
        添加时间特征

        参数:
            df: 原始数据DataFrame，必须包含datetime列

        返回:
            添加了时间特征的DataFrame
        """
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['quarter'] = df['datetime'].dt.quarter

        return df
