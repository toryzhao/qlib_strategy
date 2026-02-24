# trading/executors/backtest_executor.py
import pandas as pd
import numpy as np

class BacktestExecutor:
    """回测执行器"""

    def __init__(self, strategy, config):
        """
        初始化回测执行器

        参数:
            strategy: 策略实例
            config: 回测配置字典
        """
        self.strategy = strategy
        self.config = config
        self.portfolio = None

    def run_backtest(self, data):
        """
        运行回测

        参数:
            data: 市场数据DataFrame

        返回:
            组合收益DataFrame
        """
        # 生成交易信号
        signals = self.strategy.generate_signals(data)

        # 检查是否有完整的数据（high, low列）用于移动止损
        has_ohlcn = all(col in data.columns for col in ['high', 'low'])
        
        if has_ohlcn:
            # 运行带移动止损的回测
            self.portfolio = self._backtest_with_trailing_stop(data, signals)
        else:
            # 运行简化版回测（向后兼容）
            self.portfolio = self._simple_backtest(data, signals)

        return self.portfolio

    def _simple_backtest(self, data, signals):
        """简化版回测实现"""
        initial_cash = self.config.get('initial_cash', 1000000)
        position_ratio = self.config.get('position_ratio', 0.3)
        commission_rate = self.config.get('commission_rate', 0.0001)

        cash = initial_cash
        position = 0
        portfolio_values = []

        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]

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
                position = 0

            # 计算当前资产
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)

        # 计算收益率
        portfolio_df = pd.DataFrame({
            'portfolio_value': portfolio_values
        }, index=data.index[1:])

        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()

        return portfolio_df



    def _backtest_with_trailing_stop(self, data, signals):
        """
        带移动止损的回测实现
        
        参数:
            data: 市场数据DataFrame，包含close, high, low列
            signals: 交易信号Series (1=做多, -1=做空, 0=无信号)
        
        返回:
            组合收益DataFrame
        """
        initial_cash = self.config.get('initial_cash', 1000000)
        position_ratio = self.config.get('position_ratio', 0.3)
        commission_rate = self.config.get('commission_rate', 0.0001)
        swing_period = self.config.get('swing_period', 20)
        
        cash = initial_cash
        position = 0  # 正数=多仓，负数=空仓，0=无仓位
        entry_bar = None  # 入场bar索引
        position_type = None  # 'long' 或 'short'
        
        portfolio_values = []
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            signal = signals.iloc[i]
            
            # 检查移动止损（如果持仓中）
            if position != 0 and entry_bar is not None:
                should_exit = self._check_trailing_stop(
                    data, i, entry_bar, position_type, swing_period
                )
                
                if should_exit:
                    # 触发止损，平仓
                    cash += position * current_price
                    cash -= abs(position) * current_price * commission_rate
                    position = 0
                    entry_bar = None
                    position_type = None
                    # 计算当前资产
                    portfolio_value = cash + position * current_price
                    portfolio_values.append(portfolio_value)
                    continue
            
            # 交易执行
            if signal == 1 and position == 0:  # 开多仓
                # 计算仓位大小（支持波动率调整）
                actual_position_ratio = self._get_position_ratio(data, i, position_ratio)
                position_value = cash * actual_position_ratio
                position = position_value / current_price
                cash -= position_value
                cash -= position * current_price * commission_rate
                entry_bar = i
                position_type = 'long'
                
            elif signal == -1 and position == 0:  # 开空仓
                # 计算仓位大小（支持波动率调整）
                actual_position_ratio = self._get_position_ratio(data, i, position_ratio)
                position_value = cash * actual_position_ratio
                position = -position_value / current_price
                cash -= abs(position) * current_price * commission_rate
                entry_bar = i
                position_type = 'short'
                
            elif signal == 0 and position != 0:  # 信号平仓
                cash += position * current_price
                cash -= abs(position) * current_price * commission_rate
                position = 0
                entry_bar = None
                position_type = None
            
            # 计算当前资产
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)
        
        # 计算收益率
        portfolio_df = pd.DataFrame({
            'portfolio_value': portfolio_values
        }, index=data.index[1:])
        
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        return portfolio_df


    def _check_trailing_stop(self, data, current_bar, entry_bar, position_type, swing_period):
        """
        检查是否触发移动止损
        
        参数:
            data: 市场数据
            current_bar: 当前bar索引
            entry_bar: 入场bar索引
            position_type: 持仓类型 ('long' 或 'short')
            swing_period: 摆荡周期
        
        返回:
            bool: True表示应该平仓
        """
        # 计算可用的数据范围
        bars_in_position = current_bar - entry_bar
        
        # 如果持仓时间过短（少于5个bar），不检查止损
        if bars_in_position < 5:
            return False
        
        # 确定回看范围
        lookback_start = max(0, current_bar - swing_period)
        
        if position_type == 'long':
            # 多仓：如果收盘价跌破摆荡低点，平仓
            swing_low = data['low'].iloc[lookback_start:current_bar].min()
            current_close = data['close'].iloc[current_bar]
            return current_close < swing_low
            
        elif position_type == 'short':
            # 空仓：如果收盘价突破摆荡高点，平仓
            swing_high = data['high'].iloc[lookback_start:current_bar].max()
            current_close = data['close'].iloc[current_bar]
            return current_close > swing_high
        
        return False
    
    def _get_position_ratio(self, data, current_bar, base_ratio):
        """
        获取调整后的仓位比例（支持波动率调整）

        参数:
            data: 市场数据
            current_bar: 当前bar索引
            base_ratio: 基础仓位比例

        返回:
            float: 调整后的仓位比例
        """
        # 使用策略的RiskManager进行波动率调整
        if hasattr(self.strategy, 'risk_manager'):
            # 获取从开始到当前bar的数据
            data_slice = data.iloc[:current_bar + 1]
            adjusted_ratio = self.strategy.risk_manager.calculate_volatility_adjustment(
                data_slice, base_ratio
            )
            return adjusted_ratio
        else:
            # 如果没有RiskManager，返回基础比例
            return base_ratio

    def get_metrics(self):
        """
        获取回测指标

        返回:
            性能指标字典
        """
        if self.portfolio is None:
            raise ValueError("请先运行回测")

        returns = self.portfolio['returns'].dropna()

        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * 252,
            'sharpe_ratio': self._calculate_sharpe(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'volatility': returns.std() * np.sqrt(252),
        }

        return metrics

    def _calculate_sharpe(self, returns, risk_free_rate=0.03):
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
