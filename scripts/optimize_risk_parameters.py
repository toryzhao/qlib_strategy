#!/usr/bin/env python
# trading/scripts/optimize_risk_parameters.py
"""
优化带风险管理的MA策略参数

使用示例:
python scripts/optimize_risk_parameters.py --instrument TA --start 2020-01-01 --end 2023-12-31 --method grid
"""

import argparse
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import ContinuousContractProcessor
from strategies.technical.ma_strategy import MAStrategy
from executors.parameter_optimizer import ParameterOptimizer


def main():
    parser = argparse.ArgumentParser(description='优化带风险管理的MA策略参数')
    parser.add_argument('--instrument', type=str, required=True, help='品种代码 (TA, rb, m)')
    parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--method', type=str, default='grid', choices=['grid', 'random'],
                        help='优化方法: grid (网格搜索) 或 random (随机搜索)')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                        choices=['sharpe_ratio', 'total_return', 'max_drawdown'],
                        help='优化目标指标')
    parser.add_argument('--iterations', type=int, default=50,
                        help='随机搜索迭代次数 (仅用于random方法)')
    parser.add_argument('--output', type=str, default='reports',
                        help='报告输出目录')

    args = parser.parse_args()

    # 基础配置
    base_config = {
        'instrument': args.instrument,
        'start_date': args.start,
        'end_date': args.end,
        'initial_cash': 1000000,
        'position_ratio': 0.3,
        'commission_rate': 0.0001,
    }

    # MA策略参数网格
    ma_param_grid = {
        'fast_period': [3, 5, 7, 10],
        'slow_period': [15, 20, 25, 30],
    }

    # 风险管理参数网格
    risk_param_grid = {
        'trend_ma_period': [100, 150, 200, 250],
        'atr_period': [10, 14, 20],
        'atr_lookback': [50, 100, 150],
        'volatility_threshold': [70, 80, 90],
        'swing_period': [15, 20, 25],
    }

    # 合并参数网格
    param_grid = {**ma_param_grid, **risk_param_grid}

    print(f"\n参数优化配置:")
    print(f"  优化方法: {args.method}")
    print(f"  目标指标: {args.metric}")
    print(f"  参数范围:")

    if args.method == 'grid':
        # 计算参数组合数量
        from sklearn.model_selection import ParameterGrid
        num_combinations = len(list(ParameterGrid(param_grid)))
        print(f"  总组合数: {num_combinations}")
    else:
        print(f"  迭代次数: {args.iterations}")

    print(f"\nMA策略参数:")
    for key, values in ma_param_grid.items():
        print(f"  {key}: {values}")

    print(f"\n风险管理参数:")
    for key, values in risk_param_grid.items():
        print(f"  {key}: {values}")

    try:
        # 加载数据
        csv_path = f'data/raw/{args.instrument}.csv'
        if not os.path.exists(csv_path):
            print(f"\n错误: 数据文件不存在: {csv_path}")
            print(f"请确保 {args.instrument}.csv 文件在 data/raw/ 目录下")
            return 1

        print(f"\n正在加载数据: {csv_path}")
        processor = ContinuousContractProcessor(csv_path)
        data = processor.process(adjust_price=True)
        data = processor.load_data(start_date=args.start, end_date=args.end)
        print(f"数据加载完成: {len(data)} 条记录")

        # 创建优化器
        print("\n创建参数优化器...")
        optimizer = ParameterOptimizer(MAStrategy, data, base_config)

        # 运行优化
        print(f"\n开始优化 ({args.method} search)...")
        print("=" * 60)

        if args.method == 'grid':
            best_result, results_df = optimizer.grid_search(
                param_grid,
                metric=args.metric,
                verbose=True
            )
        else:  # random
            # 转换为参数分布格式
            param_distributions = {
                'fast_period': (3, 10),
                'slow_period': (15, 30),
                'trend_ma_period': (100, 250),
                'atr_period': (10, 20),
                'atr_lookback': (50, 150),
                'volatility_threshold': (70, 90),
                'swing_period': (15, 25),
            }
            best_result, results_df = optimizer.random_search(
                param_distributions,
                n_iter=args.iterations,
                metric=args.metric,
                random_state=42
            )

        # 打印结果
        optimizer.print_summary(best_result)

        # 保存结果
        output_dir = f'{args.output}/{args.instrument}_risk_optimization'
        os.makedirs(output_dir, exist_ok=True)

        results_path = f'{output_dir}/optimization_results.csv'
        optimizer.save_results(results_df, results_path)

        # 保存最佳参数到JSON
        import json
        best_params_path = f'{output_dir}/best_parameters.json'
        with open(best_params_path, 'w') as f:
            # Convert params to proper format for JSON serialization
            best_params_copy = best_result['params'].copy()
            json.dump(best_params_copy, f, indent=2)
        print(f"最佳参数已保存到: {best_params_path}")

        # 生成对比报告
        print("\n生成优化报告...")
        generate_optimization_report(results_df, best_result, output_dir, args.metric)

        print(f"\n所有结果已保存到: {output_dir}")
        return 0

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def generate_optimization_report(results_df, best_result, output_dir, metric):
    """生成优化报告"""
    report_path = f'{output_dir}/optimization_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("风险管理MA策略 - 参数优化报告\n")
        f.write("=" * 70 + "\n\n")

        # 基本信息
        f.write("优化概览:\n")
        f.write("-" * 70 + "\n")
        f.write(f"测试组合数: {len(results_df)}\n")
        f.write(f"优化目标: {metric}\n\n")

        # 最佳参数
        f.write("最佳参数:\n")
        f.write("-" * 70 + "\n")
        for key, value in best_result['params'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # 性能指标
        f.write("最佳性能指标:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  总收益率: {best_result['total_return']:.2%}\n")
        f.write(f"  年化收益率: {best_result['annual_return']:.2%}\n")
        f.write(f"  夏普比率: {best_result['sharpe_ratio']:.4f}\n")
        f.write(f"  最大回撤: {best_result['max_drawdown']:.2%}\n")
        f.write(f"  年化波动率: {best_result['volatility']:.2%}\n\n")

        # 统计摘要
        f.write("所有结果统计:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {metric}:\n")
        f.write(f"    最大值: {results_df[metric].max():.4f}\n")
        f.write(f"    最小值: {results_df[metric].min():.4f}\n")
        f.write(f"    平均值: {results_df[metric].mean():.4f}\n")
        f.write(f"    标准差: {results_df[metric].std():.4f}\n\n")

        f.write(f"  总收益率:\n")
        f.write(f"    最大值: {results_df['total_return'].max():.2%}\n")
        f.write(f"    最小值: {results_df['total_return'].min():.2%}\n")
        f.write(f"    平均值: {results_df['total_return'].mean():.2%}\n\n")

        f.write(f"  最大回撤:\n")
        f.write(f"    最大值: {results_df['max_drawdown'].max():.2%}\n")
        f.write(f"    最小值: {results_df['max_drawdown'].min():.2%}\n")
        f.write(f"    平均值: {results_df['max_drawdown'].mean():.2%}\n\n")

        # Top 10 参数组合
        f.write("Top 10 参数组合:\n")
        f.write("-" * 70 + "\n")
        if metric == 'max_drawdown':
            top_results = results_df.nsmallest(10, metric)
        else:
            top_results = results_df.nlargest(10, metric)

        for i, row in enumerate(top_results.itertuples(), 1):
            f.write(f"\n{i}. {metric}={getattr(row, metric):.4f}\n")
            for key, value in row.params.items():
                f.write(f"   {key}: {value}  ")
            f.write("\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"优化报告已保存到: {report_path}")


if __name__ == '__main__':
    sys.exit(main())
