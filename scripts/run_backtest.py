#!/usr/bin/env python
# trading/scripts/run_backtest.py
"""
命令行回测脚本

使用示例:
python scripts/run_backtest.py --instrument TA --strategy ma_cross --start 2020-01-01 --end 2023-12-31
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import ContinuousContractProcessor
from strategies.strategy_factory import StrategyFactory
from executors.backtest_executor import BacktestExecutor
from analyzers.performance_analyzer import PerformanceAnalyzer


def main():
    parser = argparse.ArgumentParser(description='运行回测')
    parser.add_argument('--instrument', type=str, required=True, help='品种代码 (TA, rb, m)')
    parser.add_argument('--strategy', type=str, required=True, help='策略类型 (ma_cross, macd, boll)')
    parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--fast', type=int, help='快速周期')
    parser.add_argument('--slow', type=int, help='慢速周期')
    parser.add_argument('--output', type=str, default='reports', help='报告输出目录')

    args = parser.parse_args()

    # 基础配置
    config = {
        'instrument': args.instrument,
        'start_date': args.start,
        'end_date': args.end,
        'initial_cash': 1000000,
        'position_ratio': 0.3,
    }

    # 添加策略参数
    if args.fast:
        config['fast_period'] = args.fast
    if args.slow:
        config['slow_period'] = args.slow

    try:
        # 加载数据
        csv_path = f'data/raw/{args.instrument}.csv'
        if not os.path.exists(csv_path):
            print(f"错误: 数据文件不存在: {csv_path}")
            print(f"请确保 {args.instrument}.csv 文件在 data/raw/ 目录下")
            return 1

        print(f"正在加载数据: {csv_path}")
        processor = ContinuousContractProcessor(csv_path)
        data = processor.process(adjust_price=True)
        data = processor.load_data(start_date=args.start, end_date=args.end)
        print(f"数据加载完成: {len(data)} 条记录")

        # 创建策略
        print(f"创建策略: {args.strategy}")
        strategy = StrategyFactory.create_strategy(args.strategy, config)

        # 运行回测
        print("正在运行回测...")
        executor = BacktestExecutor(strategy, config)
        portfolio = executor.run_backtest(data)

        # 输出结果
        metrics = executor.get_metrics()
        print("\n" + "=" * 50)
        print("回测结果")
        print("=" * 50)
        print(f"总收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annual_return']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"年化波动率: {metrics['volatility']:.2%}")

        # 生成报告
        output_path = f'{args.output}/{args.instrument}_{args.strategy}'
        print(f"\n正在生成报告...")
        analyzer = PerformanceAnalyzer(executor.portfolio)
        analyzer.generate_report(output_path)
        print(f"报告已保存到: {output_path}")

        return 0

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
