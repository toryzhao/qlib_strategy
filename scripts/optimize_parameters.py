#!/usr/bin/env python
# trading/scripts/optimize_parameters.py
"""
参数优化脚本

使用示例:
python scripts/optimize_parameters.py --instrument TA --start 2020-01-01 --end 2023-12-31 --metric sharpe_ratio
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import ContinuousContractProcessor
from strategies.technical.ma_strategy import MAStrategy
from executors.parameter_optimizer import ParameterOptimizer


def main():
    parser = argparse.ArgumentParser(description='优化策略参数')
    parser.add_argument('--instrument', type=str, required=True, help='品种代码 (TA, rb, m)')
    parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'total_return', 'max_drawdown'],
                       help='优化目标指标')
    parser.add_argument('--method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='优化方法 (grid或random)')
    parser.add_argument('--fast-min', type=int, default=3, help='快速均线最小值')
    parser.add_argument('--fast-max', type=int, default=30, help='快速均线最大值')
    parser.add_argument('--slow-min', type=int, default=10, help='慢速均线最小值')
    parser.add_argument('--slow-max', type=int, default=120, help='慢速均线最大值')
    parser.add_argument('--n-iter', type=int, default=50, help='随机搜索迭代次数')
    parser.add_argument('--output', type=str, default='optimization_results.csv', help='结果输出文件')

    args = parser.parse_args()

    try:
        # 1. 加载数据
        csv_path = f'data/raw/{args.instrument}.csv'
        print(f"正在加载数据: {csv_path}")
        processor = ContinuousContractProcessor(csv_path)
        data = processor.process(adjust_price=True)
        data = processor.load_data(start_date=args.start, end_date=args.end)
        print(f"数据加载完成: {len(data)} 条记录\n")

        # 2. 基础配置
        base_config = {
            'instrument': args.instrument,
            'start_date': args.start,
            'end_date': args.end,
            'initial_cash': 1000000,
            'position_ratio': 0.3,
            'commission_rate': 0.0001,
        }

        # 3. 创建优化器
        optimizer = ParameterOptimizer(MAStrategy, data, base_config)

        # 4. 运行优化
        if args.method == 'grid':
            print(f"使用网格搜索优化...")
            print(f"参数范围: fast MA {args.fast_min}-{args.fast_max}, slow MA {args.slow_min}-{args.slow_max}")
            print(f"优化目标: {args.metric}\n")

            param_grid = {
                'fast_period': range(args.fast_min, args.fast_max + 1),
                'slow_period': range(args.slow_min, args.slow_max + 1)
            }

            # 过滤掉slow <= fast的组合
            param_grid['slow_period'] = [s for s in param_grid['slow_period'] if s > args.fast_min]

            best_result, results_df = optimizer.grid_search(param_grid, metric=args.metric)

        else:  # random search
            print(f"使用随机搜索优化...")
            print(f"参数范围: fast MA {args.fast_min}-{args.fast_max}, slow MA {args.slow_min}-{args.slow_max}")
            print(f"迭代次数: {args.n_iter}")
            print(f"优化目标: {args.metric}\n")

            param_distributions = {
                'fast_period': (args.fast_min, args.fast_max),
                'slow_period': (args.slow_min, args.slow_max)
            }

            best_result, results_df = optimizer.random_search(
                param_distributions,
                n_iter=args.n_iter,
                metric=args.metric
            )

        # 5. 打印结果
        optimizer.print_summary(best_result)

        # 6. 保存结果
        optimizer.save_results(results_df, args.output)

        # 7. 显示Top 10结果
        print(f"\nTop 10 参数组合 (按 {args.metric} 排序):")
        print("-" * 80)

        if args.metric == 'max_drawdown':
            top_results = results_df.nsmallest(10, args.metric)
        else:
            top_results = results_df.nlargest(10, args.metric)

        for idx, row in top_results.iterrows():
            params = row['params']
            print(f"#{len(top_results) - top_results.index.get_loc(idx)}: "
                  f"MA({params['fast_period']}/{params['slow_period']}) | "
                  f"总收益: {row['total_return']:7.2%} | "
                  f"年化: {row['annual_return']:6.2%} | "
                  f"夏普: {row['sharpe_ratio']:6.4f} | "
                  f"回撤: {row['max_drawdown']:6.2%}")

        return 0

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
