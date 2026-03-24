#!/usr/bin/env python
"""
Regimetry pipeline execution script
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from utils.data_processor import ContinuousContractProcessor
from strategies.regime.regimetry_wrapper import RegimetryWrapper


def main():
    """Run regimetry pipeline on TA futures data"""

    print("=" * 80)
    print("Regimetry Pipeline - TA Futures")
    print("=" * 80)

    # Load data
    print("\nLoading TA futures data...")
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    df = processor.process(adjust_price=True)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Resample to daily
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"Data loaded: {len(daily_df)} daily bars")

    # Run regimetry pipeline
    print("\nRunning regimetry pipeline...")
    wrapper = RegimetryWrapper()

    try:
        assignments_path = wrapper.run_full_pipeline(
            daily_df,
            instrument='TA',
            n_clusters=8,
            window_size=30
        )

        print(f"\n[OK] Pipeline complete!")
        print(f"  Cluster assignments: {assignments_path}")

        # Display sample
        assignments = pd.read_csv(assignments_path)
        print("\nSample cluster assignments:")
        print(assignments.tail(10))

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
