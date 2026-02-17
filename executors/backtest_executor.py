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

        # 运行简化版回测
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
