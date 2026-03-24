"""
Integration Test for Enhanced MA Strategy

Tests baseline MA(20) vs Enhanced MA(20) with regime filter on TA futures data.
Multiple test periods: 2009-2011 bull market, 2019-2025 out-of-sample, full period.
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor
from strategies.technical.enhanced_ma_strategy import EnhancedMAStrategy
from strategies.regime.regime_detector import RegimeDetector
import matplotlib.pyplot as plt


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute data to daily OHLCV.

    Parameters
    ----------
    df : pd.DataFrame
        Minute-level data with datetime column

    Returns
    -------
    pd.DataFrame
        Daily OHLCV data with datetime as index
    """
    df_copy = df.copy()
    df_copy.set_index('datetime', inplace=True)

    daily = df_copy.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return daily


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns series.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns

    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def run_backtest(data: pd.DataFrame, strategy, initial_cash: float = 1000000) -> dict:
    """
    Simple backtest executor.

    Parameters
    ----------
    data : pd.DataFrame
        Market data with 'close' column
    strategy : Strategy
        Strategy instance with generate_signals() method
    initial_cash : float
        Starting capital

    Returns
    -------
    dict
        Performance metrics
    """
    signals_df = strategy.generate_signals(data)

    cash = initial_cash
    position = 0
    portfolio_values = []
    trades = []

    for i in range(1, len(data)):
        price = data['close'].iloc[i]
        signal = signals_df['signal'].iloc[i]
        position_size = signals_df['position_size'].iloc[i]

        if signal == 1 and position == 0:
            position_value = cash * position_size
            position = position_value / price
            cash -= position_value
            trades.append({
                'date': data.index[i],
                'action': 'BUY',
                'price': price,
                'size': position_size
            })

        elif signal == 0 and position > 0:
            cash += position * price
            trades[-1]['exit_price'] = price
            position = 0

        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

    portfolio_series = pd.Series(portfolio_values, index=data.index[1:])
    returns = portfolio_series.pct_change().dropna()

    years = (data.index[-1] - data.index[0]).days / 365.25

    metrics = {
        'final_value': portfolio_values[-1],
        'total_return': (portfolio_values[-1] / initial_cash - 1),
        'annual_return': returns.mean() * 252 if len(returns) > 0 else 0,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(returns),
        'num_trades': len([t for t in trades if t['action'] == 'BUY']),
    }

    return metrics


def format_percent(value: float) -> str:
    """Format float as percentage string."""
    return f"{value * 100:.2f}%"


def print_metrics(metrics: dict) -> None:
    """Print backtest metrics."""
    print(f"Total Return: {format_percent(metrics['total_return'])}")
    print(f"Annual Return: {format_percent(metrics['annual_return'])}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {format_percent(metrics['max_drawdown'])}")
    print(f"Number of Trades: {metrics['num_trades']}")


def test_period(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    detector: RegimeDetector,
    period_name: str
) -> tuple:
    """
    Test both baseline and enhanced strategies on a specific period.

    Parameters
    ----------
    data : pd.DataFrame
        Full market data
    start_date : str
        Period start date
    end_date : str
        Period end date
    detector : RegimeDetector
        Fitted regime detector
    period_name : str
        Name of the period for display

    Returns
    -------
    tuple
        (baseline_metrics, enhanced_metrics)
    """
    # Filter data for period
    period_data = data.loc[start_date:end_date].copy()

    if len(period_data) < 100:
        print(f"\n=== Testing: {period_name} ===")
        print(f"WARNING: Insufficient data ({len(period_data)} days), skipping...")
        return None, None

    print(f"\n=== Testing: {period_name} ===")
    print(f"Period: {start_date} to {end_date} ({len(period_data)} days)")

    # Baseline: Vanilla MA(20) without regime filter
    print("\n--- Baseline: Vanilla MA(20) ---")
    baseline_config = {
        'ma_period': 20,
        'use_regime_filter': False,
        'base_position': 1.0,
        'confidence_threshold': 0.60,
        'min_position': 0.30,
        'max_position': 1.50,
        'stop_loss': 0.10,
    }
    baseline_strategy = EnhancedMAStrategy(
        instrument='TA',
        start_date=start_date,
        end_date=end_date,
        config=baseline_config
    )
    baseline_metrics = run_backtest(period_data, baseline_strategy)
    print_metrics(baseline_metrics)

    # Enhanced: MA(20) + Regime Filter
    print("\n--- Enhanced: MA(20) + Regime Filter ---")
    enhanced_config = {
        'ma_period': 20,
        'use_regime_filter': True,
        'base_position': 1.0,
        'confidence_threshold': 0.60,
        'min_position': 0.30,
        'max_position': 1.50,
        'stop_loss': 0.10,
    }
    enhanced_strategy = EnhancedMAStrategy(
        instrument='TA',
        start_date=start_date,
        end_date=end_date,
        config=enhanced_config
    )
    enhanced_strategy.regime_detector = detector
    enhanced_metrics = run_backtest(period_data, enhanced_strategy)
    print_metrics(enhanced_metrics)

    # Comparison
    print("\n--- Comparison ---")
    ann_return_diff = enhanced_metrics['annual_return'] - baseline_metrics['annual_return']
    sharpe_diff = enhanced_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
    dd_diff = enhanced_metrics['max_drawdown'] - baseline_metrics['max_drawdown']

    print(f"Annual Return Improvement: {format_percent(ann_return_diff)}")
    print(f"Sharpe Ratio Improvement: {sharpe_diff:+.2f}")
    print(f"Max Drawdown Improvement: {format_percent(dd_diff)}")

    if ann_return_diff > 0:
        print("[PASS] Enhanced strategy OUTPERFORMS baseline")
    else:
        print("[FAIL] Enhanced strategy UNDERPERFORMS baseline")

    return baseline_metrics, enhanced_metrics


def main():
    """Main test execution."""
    print("=" * 80)
    print("Enhanced MA(20) Integration Test")
    print("=" * 80)

    # Load TA futures data
    print("\nLoading TA futures data...")
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    df = processor.process(adjust_price=True)

    # Resample to daily
    print("Resampling to daily data...")
    daily_data = resample_to_daily(df)
    print(f"Data shape: {daily_data.shape}")
    print(f"Date range: {daily_data.index[0]} to {daily_data.index[-1]}")

    # Fit regime detector on full dataset
    print("\nFitting regime detector...")
    detector = RegimeDetector(n_states=3, random_state=42)
    detector.fit(daily_data)
    print("Regime detector fitted successfully")

    # Define test periods
    test_periods = [
        ('2009-01-08', '2011-02-09', '2009-2011 Bull Market'),
        ('2019-01-01', '2025-03-21', '2019-2025 Out-of-Sample'),
        ('2009-01-01', '2025-03-21', 'Full Period'),
    ]

    # Store results
    results = []

    # Run tests
    for start, end, name in test_periods:
        baseline, enhanced = test_period(daily_data, start, end, detector, name)
        if baseline is not None and enhanced is not None:
            results.append({
                'period': name,
                'baseline': baseline,
                'enhanced': enhanced,
            })

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Period':<30} {'Baseline':>12} {'Enhanced':>12} {'Improvement':>12}")
    print("-" * 80)

    for result in results:
        period = result['period']
        baseline_ann = result['baseline']['annual_return']
        enhanced_ann = result['enhanced']['annual_return']
        improvement = enhanced_ann - baseline_ann

        print(f"{period:<30} {format_percent(baseline_ann):>12} {format_percent(enhanced_ann):>12} {format_percent(improvement):>12}")

    # Print primary metric
    print("\n" + "=" * 80)
    print("PRIMARY METRIC")
    print("=" * 80)
    full_period_result = [r for r in results if r['period'] == 'Full Period'][0]
    full_return = full_period_result['enhanced']['annual_return']
    print(f"Full Period Enhanced Annual Return: {format_percent(full_return)}")

    # Generate regime visualization
    print("\nGenerating regime visualization...")
    regimes, _ = detector.predict(daily_data)
    fig = detector.plot_regimes(daily_data, regimes, figsize=(16, 10))
    plt.savefig('regime_visualization.png', dpi=150, bbox_inches='tight')
    print("Regime visualization saved to: regime_visualization.png")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
