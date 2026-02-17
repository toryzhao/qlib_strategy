# trading/utils/data_processor.py
import pandas as pd
import numpy as np

class ContinuousContractProcessor:
    """主力连续合约处理器"""

    def __init__(self, csv_path):
        """
        初始化

        参数:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])

    def process(self, adjust_price=True):
        """
        处理流程：
        1. 数据清洗
        2. 识别换月点
        3. 价格复权（可选）

        参数:
            adjust_price: 是否进行价格复权

        返回:
            处理后的DataFrame
        """
        # 1. 数据清洗
        self.df = self._clean_data()

        # 2. 识别换月点
        switch_points = self._find_contract_switches()

        # 3. 价格复权
        if adjust_price and len(switch_points) > 0:
            self.df = self._adjust_price(switch_points)

        return self.df

    def _clean_data(self):
        """数据清洗"""
        # 删除缺失值
        self.df = self.df.dropna(subset=['open', 'high', 'low', 'close'])

        # 删除异常值
        for col in ['open', 'high', 'low', 'close']:
            self.df = self.df[self.df[col] > 0]

        # 时间排序
        self.df = self.df.sort_values('datetime')

        # 重置索引
        self.df = self.df.reset_index(drop=True)

        return self.df

    def _find_contract_switches(self):
        """识别合约换月点（symbol变化）"""
        self.df['symbol_changed'] = self.df['symbol'].ne(self.df['symbol'].shift())
        switch_dates = self.df[self.df['symbol_changed']]['datetime'].tolist()
        return switch_dates

    def _adjust_price(self, switch_points):
        """
        价格复权处理（后退复权）

        策略：
        1. 检测换月日的价格跳空
        2. 计算复权因子
        3. 对历史数据进行向后复权
        """
        df_copy = self.df.copy()
        adjustment_factor = 1.0

        # 按时间倒序处理
        for i in range(len(switch_points) - 1, 0, -1):
            switch_date = switch_points[i]

            # 获取换月前后的收盘价
            before_mask = df_copy['datetime'] <= switch_date
            after_mask = df_copy['datetime'] > switch_date

            if before_mask.sum() == 0 or after_mask.sum() == 0:
                continue

            before_close = df_copy[before_mask]['close'].iloc[-1]
            after_close = df_copy[after_mask]['close'].iloc[0]

            # 计算跳空比例
            gap_ratio = after_close / before_close
            adjustment_factor *= gap_ratio

            # 对换月之前的所有价格进行调整
            for col in ['open', 'high', 'low', 'close']:
                df_copy.loc[before_mask, col] *= adjustment_factor

        return df_copy

    def load_data(self, start_date=None, end_date=None):
        """
        加载指定时间范围的数据

        参数:
            start_date: 开始日期 (str or datetime)
            end_date: 结束日期 (str or datetime)

        返回:
            过滤后的DataFrame
        """
        if start_date:
            self.df = self.df[self.df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            self.df = self.df[self.df['datetime'] <= pd.to_datetime(end_date)]
        return self.df
