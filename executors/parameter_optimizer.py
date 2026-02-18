# trading/executors/parameter_optimizer.py
from sklearn.model_selection import ParameterGrid
import pandas as pd
from tqdm import tqdm
import json


class ParameterOptimizer:
    """参数优化器"""

    def __init__(self, strategy_class, data, base_config):
        """
        参数:
            strategy_class: 策略类
            data: 回测数据
            base_config: 基础配置
        """
        self.strategy_class = strategy_class
        self.data = data
        self.base_config = base_config

    def grid_search(self, param_grid, metric='sharpe_ratio', verbose=True):
        """
        网格搜索优化

        参数:
            param_grid: 参数网格
            metric: 优化目标指标 ('sharpe_ratio', 'total_return', 'max_drawdown')
            verbose: 是否显示进度条

        返回:
            (best_result, results_df) - 最佳参数和所有结果
        """
        from executors.backtest_executor import BacktestExecutor

        # 生成参数组合
        params_list = list(ParameterGrid(param_grid))

        results = []
        iterator = tqdm(params_list, desc="优化进度") if verbose else params_list

        for params in iterator:
            # 更新配置
            config = self.base_config.copy()
            config.update(params)

            try:
                # 创建策略
                strategy = self.strategy_class(
                    instrument=config['instrument'],
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    config=config
                )

                # 运行回测
                executor = BacktestExecutor(strategy, config)
                executor.run_backtest(self.data)

                # 获取指标
                metrics = executor.get_metrics()
                metrics['params'] = params
                results.append(metrics)

                if verbose:
                    # 显示当前结果
                    print(f"  MA({params['fast_period']}/{params['slow_period']}): "
                          f"{metric}={metrics[metric]:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  参数 {params} 失败: {str(e)}")
                continue

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            raise ValueError("没有成功的回测结果")

        # 找出最优参数
        if metric == 'max_drawdown':
            # 对于最大回撤，越小越好
            best_idx = results_df[metric].idxmin()
        else:
            # 对于其他指标，越大越好
            best_idx = results_df[metric].idxmax()

        best_result = results_df.iloc[best_idx].to_dict()

        return best_result, results_df

    def random_search(self, param_distributions, n_iter=50, metric='sharpe_ratio', random_state=42):
        """
        随机搜索优化

        参数:
            param_distributions: 参数分布字典
            n_iter: 迭代次数
            metric: 优化目标指标
            random_state: 随机种子

        返回:
            (best_result, results_df)
        """
        import numpy as np
        np.random.seed(random_state)

        results = []

        for i in tqdm(range(n_iter), desc="随机优化进度"):
            # 随机采样参数
            params = {}
            for param_name, (low, high) in param_distributions.items():
                params[param_name] = np.random.randint(low, high + 1)

            # 更新配置
            config = self.base_config.copy()
            config.update(params)

            try:
                # 创建策略并运行回测
                from executors.backtest_executor import BacktestExecutor
                strategy = self.strategy_class(
                    instrument=config['instrument'],
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    config=config
                )

                executor = BacktestExecutor(strategy, config)
                executor.run_backtest(self.data)
                metrics = executor.get_metrics()
                metrics['params'] = params
                results.append(metrics)

            except Exception as e:
                continue

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 找出最优参数
        if metric == 'max_drawdown':
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()

        best_result = results_df.iloc[best_idx].to_dict()

        return best_result, results_df

    def save_results(self, results_df, output_path='optimization_results.csv'):
        """保存优化结果"""
        # 展开params字典
        params_df = pd.DataFrame(results_df['params'].tolist())
        combined = pd.concat([params_df, results_df.drop('params', axis=1).reset_index(drop=True)], axis=1)
        combined.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")

    def print_summary(self, best_result):
        """打印优化结果摘要"""
        print("\n" + "=" * 60)
        print("参数优化结果")
        print("=" * 60)

        print("\n最佳参数:")
        for key, value in best_result['params'].items():
            print(f"  {key}: {value}")

        print("\n性能指标:")
        print(f"  总收益率: {best_result['total_return']:.2%}")
        print(f"  年化收益率: {best_result['annual_return']:.2%}")
        print(f"  夏普比率: {best_result['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {best_result['max_drawdown']:.2%}")
        print(f"  年化波动率: {best_result['volatility']:.2%}")
        print("=" * 60)
