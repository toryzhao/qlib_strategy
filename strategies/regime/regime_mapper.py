"""
Dynamic regime mapping - maps cluster IDs to trading states
"""

import pandas as pd
import numpy as np


class RegimeMapper:
    """
    Maps Regimetry cluster IDs to trading states (BULL/BEAR/RANGING)

    Uses dynamic thresholds based on volatility (ATR) to classify clusters.
    """

    def __init__(self, lookback_days=20, threshold_multiplier=1.5):
        self.lookback_days = lookback_days
        self.threshold_multiplier = threshold_multiplier

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_cluster_returns(self, df, lookback=None):
        if lookback is None:
            lookback = self.lookback_days

        returns = (df['close'] / df['close'].shift(lookback - 1)) - 1
        return returns.dropna()

    def map_cluster_to_state(self, df):
        # Calculate return from start to end of the cluster
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        recent_return = (end_price / start_price) - 1

        atr = self._calculate_atr(df)
        atr_20 = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else atr.iloc[-2]
        price = df['close'].iloc[-1]

        atr_ratio = (atr_20 * self.threshold_multiplier) / price
        threshold_bull = atr_ratio
        threshold_bear = -atr_ratio

        if recent_return > threshold_bull:
            return 'BULL'
        elif recent_return < threshold_bear:
            return 'BEAR'
        else:
            return 'RANGING'

    def map_all_clusters(self, assignments, df):
        cluster_states = {}

        assignments['Date'] = pd.to_datetime(assignments['Date'])

        # Get the index name from the dataframe
        index_name = df.index.name if df.index.name else 'index'

        df_merged = df.reset_index().merge(
            assignments,
            left_on=index_name,
            right_on='Date',
            how='inner'
        )
        df_merged = df_merged.set_index('Date')

        for cluster_id in df_merged['Cluster_ID'].unique():
            cluster_data = df_merged[df_merged['Cluster_ID'] == cluster_id]
            state = self.map_cluster_to_state(cluster_data)
            cluster_states[cluster_id] = state

        return cluster_states

    def get_market_state(self, assignments, df, date):
        assignments['Date'] = pd.to_datetime(assignments['Date'])
        date = pd.to_datetime(date)

        cluster_row = assignments[assignments['Date'] == date]

        if len(cluster_row) == 0:
            previous_dates = assignments[assignments['Date'] < date]['Date']
            if len(previous_dates) == 0:
                return 'RANGING'

            last_date = previous_dates.max()
            cluster_row = assignments[assignments['Date'] == last_date]

        cluster_id = cluster_row['Cluster_ID'].iloc[0]

        if not hasattr(self, '_cluster_states'):
            self._cluster_states = self.map_all_clusters(assignments, df)

        return self._cluster_states.get(cluster_id, 'RANGING')
