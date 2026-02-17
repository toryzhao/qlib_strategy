# trading/analyzers/performance_analyzer.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self, portfolio):
        """
        初始化分析器

        参数:
            portfolio: 组合收益DataFrame，必须包含'returns'列
        """
        self.portfolio = portfolio

    def generate_report(self, output_path='reports'):
        """
        生成完整分析报告

        参数:
            output_path: 报告输出目录

        返回:
            性能指标字典
        """
        # 1. 基础指标
        metrics = self._calculate_metrics()

        # 2. 可视化
        self._plot_equity_curve(metrics, output_path)
        self._plot_drawdown(metrics, output_path)

        # 3. 生成文本报告
        self._generate_text_report(metrics, output_path)

        return metrics

    def _calculate_metrics(self):
        """计算详细指标"""
        returns = self.portfolio['returns'].dropna()

        metrics = {
            # 收益指标
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * 252,

            # 风险指标
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),

            # 风险调整收益
            'sharpe_ratio': self._calculate_sharpe(returns),
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

    def _plot_equity_curve(self, metrics, output_path):
        """绘制资金曲线"""
        os.makedirs(output_path, exist_ok=True)

        plt.figure(figsize=(12, 6))
        cumulative_returns = (1 + self.portfolio['returns'].fillna(0)).cumprod()

        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.savefig(f'{output_path}/equity_curve.png')
        plt.close()

    def _plot_drawdown(self, metrics, output_path):
        """绘制回撤图"""
        os.makedirs(output_path, exist_ok=True)

        plt.figure(figsize=(12, 6))
        cumulative = (1 + self.portfolio['returns'].fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        plt.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3, color='red')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig(f'{output_path}/drawdown.png')
        plt.close()

    def _generate_text_report(self, metrics, output_path):
        """生成文本报告"""
        os.makedirs(output_path, exist_ok=True)

        with open(f'{output_path}/performance_report.txt', 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("回测性能报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("收益指标:\n")
            f.write(f"  总收益率: {metrics['total_return']:.2%}\n")
            f.write(f"  年化收益率: {metrics['annual_return']:.2%}\n\n")

            f.write("风险指标:\n")
            f.write(f"  年化波动率: {metrics['volatility']:.2%}\n")
            f.write(f"  最大回撤: {metrics['max_drawdown']:.2%}\n\n")

            f.write("风险调整收益:\n")
            f.write(f"  夏普比率: {metrics['sharpe_ratio']:.4f}\n")
