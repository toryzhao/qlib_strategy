# Regimetry Adaptive Range Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an adaptive trading strategy that uses Transformer-based regime detection (Regimetry) to switch between mean-reversion and trend-following logic based on market state.

**Architecture:**
- **Regime Detection**: Regimetry pipeline generates cluster labels → dynamic mapping to BULL/BEAR/RANGING
- **Signal Generation**: Market-state-specific entry logic (pullbacks in bull, rallies in bear, breakouts in ranging)
- **Risk Management**: 2% account risk per trade, 2 ATR stop loss, regime change exits
- **Pure Positions**: One position per direction, no hedging

**Tech Stack:**
- Python 3.8+, pandas, numpy
- Regimetry (Transformer embeddings + spectral clustering)
- TA-Lib or custom ATR calculation
- Existing codebase: strategies.base, utils.data_processor, executors.backtest_executor

---

## Task 1: Install and Configure Regimetry

**Files:**
- Modify: `requirements.txt`
- Create: `strategies/regime/regimetry_wrapper.py`
- Create: `scripts/regimetry_pipeline.py`

**Step 1: Add regimetry to requirements.txt**

```bash
echo "regimetry>=0.1.0" >> requirements.txt
echo "tensorflow>=2.12.0" >> requirements.txt
echo "scikit-learn>=1.2.0" >> requirements.txt
```

**Step 2: Install regimetry**

```bash
pip install regimetry tensorflow scikit-learn
```

Expected: Successfully installs regimetry and dependencies

**Step 3: Create Regimetry wrapper**

Create `strategies/regime/regimetry_wrapper.py`:

```python
"""
Regimetry integration wrapper
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


class RegimetryWrapper:
    """
    Wrapper for regimetry regime detection pipeline
    """

    def __init__(self, base_dir='data/regimetry'):
        self.base_dir = Path(base_dir)
        self.embeddings_dir = self.base_dir / 'embeddings'
        self.reports_dir = self.base_dir / 'reports'

        # Create directories
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, df, output_path):
        """
        Prepare data for regimetry ingestion

        Args:
            df: DataFrame with OHLCV data
            output_path: Path to save processed CSV
        """
        # Select required columns
        required_cols = ['close', 'high', 'low', 'open']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain {required_cols}")

        # Add basic features if not present
        if 'AHMA' not in df.columns:
            df['AHMA'] = df['close'].rolling(window=20).mean()

        if 'ATR' not in df.columns:
            df['ATR'] = self._calculate_atr(df)

        # Reset index to have Date column
        df_output = df.reset_index()
        df_output.columns = df_output.columns.str.capitalize()

        # Save to CSV
        df_output.to_csv(output_path, index=False)
        print(f"Data prepared: {output_path}")

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def run_embedding(self, input_path, output_name, window_size=30):
        """
        Run regimetry embedding pipeline

        Args:
            input_path: Path to input CSV
            output_name: Name for output file
            window_size: Rolling window size
        """
        cmd = [
            'python', '-m', 'regimetry.cli', 'embed',
            '--signal-input-path', str(input_path),
            '--output-name', output_name,
            '--window-size', str(window_size),
            '--stride', '1',
            '--encoding-method', 'sinusoidal',
            '--encoding-style', 'interleaved'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Embedding failed: {result.stderr}")

        print(f"Embedding complete: {output_name}")

    def run_clustering(self, embedding_path, data_path, n_clusters=8, window_size=30):
        """
        Run regimetry clustering pipeline

        Args:
            embedding_path: Path to embeddings .npy file
            data_path: Path to original data CSV
            n_clusters: Number of clusters
            window_size: Window size used for embedding
        """
        output_dir = self.reports_dir / 'TA'
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'python', '-m', 'regimetry.cli', 'cluster',
            '--embedding-path', str(embedding_path),
            '--regime-data-path', str(data_path),
            '--output-dir', str(output_dir),
            '--window-size', str(window_size),
            '--n-clusters', str(n_clusters)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Clustering failed: {result.stderr}")

        print(f"Clustering complete: {output_dir}")

        # Return path to cluster assignments
        return output_dir / 'cluster_assignments.csv'

    def run_full_pipeline(self, df, instrument='TA', n_clusters=8, window_size=30):
        """
        Run full regimetry pipeline: prepare → embed → cluster

        Args:
            df: DataFrame with OHLCV data
            instrument: Instrument name
            n_clusters: Number of clusters
            window_size: Window size for embeddings

        Returns:
            Path to cluster assignments CSV
        """
        # Prepare paths
        data_path = self.base_dir / f'{instrument}_processed.csv'
        embedding_name = f'{instrument}_embeddings.npy'
        embedding_path = self.embeddings_dir / embedding_name

        # Step 1: Prepare data
        self.prepare_data(df, data_path)

        # Step 2: Generate embeddings
        self.run_embedding(data_path, embedding_name, window_size)

        # Step 3: Cluster regimes
        assignments_path = self.run_clustering(
            embedding_path, data_path, n_clusters, window_size
        )

        return assignments_path
```

**Step 4: Create pipeline script**

Create `scripts/regimetry_pipeline.py`:

```python
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

        print(f"\n✓ Pipeline complete!")
        print(f"  Cluster assignments: {assignments_path}")

        # Display sample
        assignments = pd.read_csv(assignments_path)
        print("\nSample cluster assignments:")
        print(assignments.tail(10))

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
```

**Step 5: Commit**

```bash
git add requirements.txt strategies/regime/regimetry_wrapper.py scripts/regimetry_pipeline.py
git commit -m "feat: Add regimetry integration

- Install regimetry and dependencies
- Create RegimetryWrapper class for pipeline execution
- Add regimetry_pipeline.py script for data processing
- Support embedding generation and clustering"
```

---

## Task 2: Implement Dynamic Regime Mapping

**Files:**
- Create: `strategies/regime/regime_mapper.py`
- Create: `tests/regime/test_regime_mapper.py`

**Step 1: Write failing tests**

Create `tests/regime/test_regime_mapper.py`:

```python
import pytest
import pandas as pd
import numpy as np
from strategies.regime.regime_mapper import RegimeMapper


def test_regime_mapper_initialization():
    """Test RegimeMapper can be initialized"""
    mapper = RegimeMapper()
    assert mapper.lookback_days == 20
    assert mapper.threshold_multiplier == 1.5


def test_calculate_cluster_return():
    """Test calculating cluster returns"""
    mapper = RegimeMapper()

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.arange(100) * 0.5  # Upward trend
    df = pd.DataFrame({'close': prices}, index=dates)

    cluster_returns = mapper._calculate_cluster_returns(df, lookback=20)

    # Should have returns for days 20 onwards
    assert len(cluster_returns) == 81  # 100 - 20 + 1
    assert cluster_returns.iloc[-1] > 0  # Positive return


def test_map_cluster_to_state_bull():
    """Test mapping cluster to bull state"""
    mapper = RegimeMapper()

    # Create data with strong uptrend
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.arange(100) * 2  # Strong uptrend
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    state = mapper.map_cluster_to_state(df)

    # Should be BULL (strong positive returns)
    assert state == 'BULL'


def test_map_cluster_to_state_bear():
    """Test mapping cluster to bear state"""
    mapper = RegimeMapper()

    # Create data with strong downtrend
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 - np.arange(100) * 2  # Strong downtrend
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    state = mapper.map_cluster_to_state(df)

    # Should be BEAR (strong negative returns)
    assert state == 'BEAR'


def test_map_cluster_to_state_ranging():
    """Test mapping cluster to ranging state"""
    mapper = RegimeMapper()

    # Create sideways data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(100) * 2  # Sideways with noise
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    state = mapper.map_cluster_to_state(df)

    # Should be RANGING (returns near zero)
    assert state == 'RANGING'


def test_map_all_clusters():
    """Test mapping multiple clusters"""
    mapper = RegimeMapper()

    # Create cluster assignments
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    cluster_ids = np.array([0] * 30 + [1] * 40 + [2] * 30)
    assignments = pd.DataFrame({
        'Date': dates,
        'Cluster_ID': cluster_ids
    })

    # Create price data for each cluster
    prices = np.concatenate([
        100 + np.arange(30) * 2,  # Cluster 0: uptrend
        100 - np.arange(40),       # Cluster 1: downtrend
        np.ones(30) * 100 + np.random.randn(30) * 2  # Cluster 2: sideways
    ])

    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1
    }, index=dates)

    # Map all clusters
    cluster_states = mapper.map_all_clusters(assignments, df)

    # Check that all clusters are mapped
    assert 0 in cluster_states
    assert 1 in cluster_states
    assert 2 in cluster_states

    # Check cluster 0 is BULL, cluster 1 is BEAR, cluster 2 is RANGING
    assert cluster_states[0] == 'BULL'
    assert cluster_states[1] == 'BEAR'
    assert cluster_states[2] == 'RANGING'
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/regime/test_regime_mapper.py -v
```

Expected: All tests FAIL with "ModuleNotFoundError: No module named 'strategies.regime.regime_mapper'"

**Step 3: Implement RegimeMapper**

Create `strategies/regime/regime_mapper.py`:

```python
"""
Dynamic regime mapping - maps cluster IDs to trading states
"""

import pandas as pd
import numpy as np


class RegimeMapper:
    """
    Maps Regimetry cluster IDs to trading states (BULL/BEAR/RANGING)

    Uses dynamic thresholds based on volatility (ATR) to classify clusters.
    """

    def __init__(self, lookback_days=20, threshold_multiplier=1.5):
        """
        Initialize RegimeMapper

        Args:
            lookback_days: Days to look back for calculating returns
            threshold_multiplier: Multiplier for ATR-based thresholds
        """
        self.lookback_days = lookback_days
        self.threshold_multiplier = threshold_multiplier

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_cluster_returns(self, df, lookback=None):
        """
        Calculate returns for a cluster's data

        Args:
            df: DataFrame with price data for this cluster
            lookback: Lookback period (default: self.lookback_days)

        Returns:
            Series of returns
        """
        if lookback is None:
            lookback = self.lookback_days

        # Calculate lookback return
        returns = (df['close'] / df['close'].shift(lookback)) - 1

        return returns

    def map_cluster_to_state(self, df):
        """
        Map a single cluster's data to trading state

        Args:
            df: DataFrame with 'close', 'high', 'low' columns

        Returns:
            str: 'BULL', 'BEAR', or 'RANGING'
        """
        # Calculate recent return
        returns = self._calculate_cluster_returns(df)
        recent_return = returns.iloc[-1]

        # Calculate ATR for volatility-based threshold
        atr = self._calculate_atr(df)
        atr_20 = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else atr.iloc[-2]
        price = df['close'].iloc[-1]

        # Dynamic threshold based on volatility
        atr_ratio = (atr_20 * self.threshold_multiplier) / price
        threshold_bull = atr_ratio
        threshold_bear = -atr_ratio

        # Map to state
        if recent_return > threshold_bull:
            return 'BULL'
        elif recent_return < threshold_bear:
            return 'BEAR'
        else:
            return 'RANGING'

    def map_all_clusters(self, assignments, df):
        """
        Map all clusters to trading states

        Args:
            assignments: DataFrame with columns ['Date', 'Cluster_ID']
            df: DataFrame with price data (indexed by date)

        Returns:
            dict: {cluster_id: 'BULL'/'BEAR'/'RANGING'}
        """
        cluster_states = {}

        # Merge assignments with price data
        assignments['Date'] = pd.to_datetime(assignments['Date'])
        df_merged = df.reset_index().merge(
            assignments,
            left_on='datetime',
            right_on='Date',
            how='inner'
        )
        df_merged = df_merged.set_index('Date')

        # For each unique cluster
        for cluster_id in df_merged['Cluster_ID'].unique():
            # Get data for this cluster
            cluster_data = df_merged[df_merged['Cluster_ID'] == cluster_id]

            # Map to state
            state = self.map_cluster_to_state(cluster_data)
            cluster_states[cluster_id] = state

        return cluster_states

    def get_market_state(self, assignments, df, date):
        """
        Get market state for a specific date

        Args:
            assignments: DataFrame with cluster assignments
            df: DataFrame with price data
            date: Date to query

        Returns:
            str: 'BULL', 'BEAR', or 'RANGING'
        """
        # Find cluster for this date
        assignments['Date'] = pd.to_datetime(assignments['Date'])
        date = pd.to_datetime(date)

        cluster_row = assignments[assignments['Date'] == date]

        if len(cluster_row) == 0:
            # No assignment for this date, use previous day
            previous_dates = assignments[assignments['Date'] < date]['Date']
            if len(previous_dates) == 0:
                return 'RANGING'  # Default

            last_date = previous_dates.max()
            cluster_row = assignments[assignments['Date'] == last_date]

        cluster_id = cluster_row['Cluster_ID'].iloc[0]

        # Map cluster to state (cached or compute)
        if not hasattr(self, '_cluster_states'):
            self._cluster_states = self.map_all_clusters(assignments, df)

        return self._cluster_states.get(cluster_id, 'RANGING')
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/regime/test_regime_mapper.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add strategies/regime/regime_mapper.py tests/regime/test_regime_mapper.py
git commit -m "feat: Implement RegimeMapper for dynamic cluster mapping

- Map cluster IDs to BULL/BEAR/RANGING states
- Dynamic thresholds based on ATR volatility
- Support for single cluster and batch mapping
- Query market state by date"
```

---

## Task 3: Implement Core Strategy Class

**Files:**
- Create: `strategies/technical/regimetry_adaptive_range_strategy.py`
- Create: `tests/technical/test_regimetry_adaptive_range_strategy.py`

**Step 1: Write failing tests for strategy initialization**

Create `tests/technical/test_regimetry_adaptive_range_strategy.py`:

```python
import pytest
import pandas as pd
import numpy as np
from strategies.technical.regimetry_adaptive_range_strategy import RegimetryAdaptiveRangeStrategy


def test_strategy_initialization():
    """Test strategy can be initialized"""
    config = {
        'lookback_days': 20,
        'window_short': 10,
        'window_long': 30,
        'entry_buffer_atr': 1.0,
        'stop_loss_atr': 2.0,
        'risk_per_trade': 0.02,
    }

    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', config)

    assert strategy.lookback_days == 20
    assert strategy.window_short == 10
    assert strategy.window_long == 30
    assert strategy.entry_buffer_atr == 1.0


def test_calculate_atr():
    """Test ATR calculation"""
    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', {})

    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'high': np.random.rand(50) + 100,
        'low': np.random.rand(50) + 99,
        'close': np.random.rand(50) + 99.5
    }, index=dates)

    atr = strategy.calculate_atr(df)

    # Should have ATR values for days 14 onwards
    assert atr.iloc[-1] > 0


def test_calculate_dynamic_window():
    """Test dynamic window calculation"""
    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', {})

    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    atr_values = np.concatenate([
        np.ones(40) * 50,  # Low volatility
        np.ones(60) * 100  # High volatility
    ])

    atr_series = pd.Series(atr_values, index=dates)

    # Low volatility period
    window_low = strategy.calculate_dynamic_window(atr_series, 60)
    assert window_low == 30  # Long window

    # High volatility period
    window_high = strategy.calculate_dynamic_window(atr_series, 90)
    assert window_high == 10  # Short window


def test_calculate_position_size():
    """Test risk-based position sizing"""
    config = {'risk_per_trade': 0.02}
    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', config)

    account_value = 1000000
    entry_price = 3000
    atr = 50

    contracts, position_value = strategy.calculate_position_size(
        account_value, entry_price, atr
    )

    # Risk amount = 1,000,000 * 0.02 = 20,000
    # Stop distance = 2 * 50 = 100
    # Position = 20,000 / 100 = 200,000
    # Contracts = 200,000 / 3000 = 66

    assert contracts == 66
    assert position_value == pytest.approx(200000, rel=0.1)


def test_generate_bull_signal():
    """Test bull market signal generation"""
    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', {})

    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    prices = 100 + np.arange(50)

    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1,
        'atr': np.ones(50) * 2
    }, index=dates)

    # Current price near recent low
    signal = strategy.generate_bull_signal(df, long_position=0)

    # Should generate long signal
    assert signal == 'LONG'


def test_generate_bear_signal():
    """Test bear market signal generation"""
    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', {})

    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    prices = 100 - np.arange(50)

    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1,
        'atr': np.ones(50) * 2
    }, index=dates)

    # Current price near recent high
    signal = strategy.generate_bear_signal(df, short_position=0)

    # Should generate short signal
    assert signal == 'SHORT'


def test_generate_ranging_signal_breakout():
    """Test ranging market breakout signal"""
    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', {})

    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(50) * 2  # Sideways

    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1,
        'atr': np.ones(50) * 2
    }, index=dates)

    # Last price breaks out
    df.loc[df.index[-1], 'close'] = 110

    signal, pending = strategy.generate_ranging_signal(df, long_position=0, short_position=0)

    # Should generate pending long signal
    assert pending == 'LONG_PENDING'


def test_generate_ranging_signal_confirmation():
    """Test ranging market breakout confirmation"""
    strategy = RegimetryAdaptiveRangeStrategy('TA', '2020-01-01', '2021-01-01', {})

    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(50) * 2
    prices[-2] = 110  # Breakout
    prices[-1] = 111  # Confirmation

    df = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1,
        'atr': np.ones(50) * 2
    }, index=dates)

    signal, pending = strategy.generate_ranging_signal(
        df, long_position=0, short_position=0, pending_signal='LONG_PENDING'
    )

    # Should confirm and enter long
    assert signal == 'LONG'
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/technical/test_regimetry_adaptive_range_strategy.py -v
```

Expected: All tests FAIL with module not found

**Step 3: Implement RegimetryAdaptiveRangeStrategy**

Create `strategies/technical/regimetry_adaptive_range_strategy.py`:

```python
"""
Regimetry Adaptive Range Strategy (RARS)

Adapts trading logic based on market regime detected by Regimetry:
- Bull markets: Buy pullbacks to recent lows
- Bear markets: Short rallies to recent highs
- Ranging markets: Trade confirmed breakouts
"""

import pandas as pd
import numpy as np
from strategies.base.base_strategy import FuturesStrategy
from strategies.regime.regime_mapper import RegimeMapper


class RegimetryAdaptiveRangeStrategy(FuturesStrategy):
    """
    Adaptive strategy that changes behavior based on market regime

    Entry Logic:
    - Bull: Enter long when price ≤ (recent_low + entry_buffer_atr)
    - Bear: Enter short when price ≥ (recent_high - entry_buffer_atr)
    - Ranging: Enter on breakout with confirmation

    Exit Logic:
    - Regime change
    - Stop loss at 2 ATR
    """

    def __init__(self, instrument, start_date, end_date, config):
        super().__init__(instrument, start_date, end_date, config)

        # Parameters
        self.lookback_days = config.get('lookback_days', 20)
        self.window_short = config.get('window_short', 10)
        self.window_long = config.get('window_long', 30)
        self.entry_buffer_atr = config.get('entry_buffer_atr', 1.0)
        self.stop_loss_atr = config.get('stop_loss_atr', 2.0)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)

        # Regime detection
        self.regime_mapper = RegimeMapper(
            lookback_days=self.lookback_days,
            threshold_multiplier=1.5
        )
        self.assignments_path = config.get('assignments_path')

        # State tracking
        self.pending_signal = None
        self.entry_price = None
        self.entry_atr = None
        self.entry_date = None

    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_dynamic_window(self, atr_series, current_idx):
        """
        Calculate dynamic window based on volatility

        Returns 10 (high volatility) or 30 (low volatility)
        """
        # Calculate ATR baseline (60-day median)
        baseline_idx = max(0, current_idx - 60)
        atr_baseline = atr_series.iloc[baseline_idx:current_idx].median()

        current_atr = atr_series.iloc[current_idx]

        if current_atr > atr_baseline:
            return self.window_short
        else:
            return self.window_long

    def calculate_position_size(self, account_value, entry_price, atr):
        """
        Calculate position size based on risk management rules

        Args:
            account_value: Total account value
            entry_price: Entry price
            atr: Current ATR

        Returns:
            (contracts, position_value)
        """
        risk_amount = account_value * self.risk_per_trade
        stop_loss_distance = self.stop_loss_atr * atr

        position_value = risk_amount / stop_loss_distance
        contracts = int(position_value / entry_price)

        return contracts, position_value

    def generate_bull_signal(self, data, long_position):
        """
        Generate long signal for bull market

        Enters when price pulls back to recent low + buffer
        """
        atr = self.calculate_atr(data)
        current_idx = len(data) - 1

        window = self.calculate_dynamic_window(atr, current_idx)

        recent_low = data['low'].rolling(window).min().iloc[-1]
        current_atr = atr.iloc[-1]
        entry_zone = recent_low + (self.entry_buffer_atr * current_atr)
        current_price = data['close'].iloc[-1]

        if long_position == 0 and current_price <= entry_zone:
            return 'LONG'

        return None

    def generate_bear_signal(self, data, short_position):
        """
        Generate short signal for bear market

        Enters when price rallies to recent high - buffer
        """
        atr = self.calculate_atr(data)
        current_idx = len(data) - 1

        window = self.calculate_dynamic_window(atr, current_idx)

        recent_high = data['high'].rolling(window).max().iloc[-1]
        current_atr = atr.iloc[-1]
        entry_zone = recent_high - (self.entry_buffer_atr * current_atr)
        current_price = data['close'].iloc[-1]

        if short_position == 0 and current_price >= entry_zone:
            return 'SHORT'

        return None

    def generate_ranging_signal(self, data, long_position, short_position, pending_signal=None):
        """
        Generate signal for ranging market

        Trades breakouts with next-bar confirmation
        """
        atr = self.calculate_atr(data)
        current_idx = len(data) - 1

        window = self.calculate_dynamic_window(atr, current_idx)

        recent_high = data['high'].rolling(window).max().iloc[-1]
        recent_low = data['low'].rolling(window).min().iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]

        # Check for pending signal confirmation
        if pending_signal == 'LONG_PENDING':
            breakout_level = recent_high + (self.entry_buffer_atr * current_atr)
            if current_price > breakout_level:
                return 'LONG', None
            else:
                return None, None  # Failed confirmation

        if pending_signal == 'SHORT_PENDING':
            breakdown_level = recent_low - (self.entry_buffer_atr * current_atr)
            if current_price < breakdown_level:
                return 'SHORT', None
            else:
                return None, None

        # Check for new breakout signals
        if long_position == 0:
            breakout_level = recent_high + (self.entry_buffer_atr * current_atr)
            if current_price > breakout_level:
                return None, 'LONG_PENDING'

        if short_position == 0:
            breakdown_level = recent_low - (self.entry_buffer_atr * current_atr)
            if current_price < breakdown_level:
                return None, 'SHORT_PENDING'

        return None, None

    def check_stop_loss(self, position, current_price):
        """
        Check if stop loss is hit

        Returns True if stop loss hit
        """
        if self.entry_price is None or self.entry_atr is None:
            return False

        if position['side'] == 'LONG':
            stop_loss = self.entry_price - (self.stop_loss_atr * self.entry_atr)
            return current_price < stop_loss

        elif position['side'] == 'SHORT':
            stop_loss = self.entry_price + (self.stop_loss_atr * self.entry_atr)
            return current_price > stop_loss

        return False

    def generate_signals(self, data, assignments):
        """
        Generate trading signals

        Args:
            data: DataFrame with OHLCV data
            assignments: DataFrame with cluster assignments

        Returns:
            DataFrame with signals
        """
        signals = []
        position_state = {'long': 0, 'short': 0}
        current_regime = None

        for i in range(60, len(data)):  # Skip warmup period
            current_data = data.iloc[:i+1]
            current_date = data.index[i]
            current_price = data['close'].iloc[i]

            # Get market regime
            regime = self.regime_mapper.get_market_state(
                assignments, current_data, current_date
            )

            # Check for regime change (exit all positions)
            if current_regime is not None and regime != current_regime:
                position_state = {'long': 0, 'short': 0}
                self.entry_price = None
                self.entry_atr = None

            current_regime = regime

            # Generate signal based on regime
            signal = None

            if regime == 'BULL':
                signal = self.generate_bull_signal(current_data, position_state['long'])

            elif regime == 'BEAR':
                signal = self.generate_bear_signal(current_data, position_state['short'])

            elif regime == 'RANGING':
                signal, self.pending_signal = self.generate_ranging_signal(
                    current_data,
                    position_state['long'],
                    position_state['short'],
                    self.pending_signal
                )

            # Check stop loss
            if position_state['long'] > 0 or position_state['short'] > 0:
                position = {
                    'side': 'LONG' if position_state['long'] > 0 else 'SHORT',
                    'size': position_state['long'] + position_state['short']
                }

                if self.check_stop_loss(position, current_price):
                    # Close position
                    if position_state['long'] > 0:
                        position_state['long'] = 0
                    if position_state['short'] > 0:
                        position_state['short'] = 0

                    self.entry_price = None
                    self.entry_atr = None
                    signals.append({
                        'date': current_date,
                        'signal': 0,
                        'price': current_price,
                        'reason': 'STOP_LOSS'
                    })
                    continue

            # Execute new signal
            if signal == 'LONG' and position_state['long'] == 0:
                # Close short if exists
                if position_state['short'] > 0:
                    position_state['short'] = 0

                # Enter long
                atr = self.calculate_atr(current_data).iloc[-1]
                contracts, _ = self.calculate_position_size(
                    1000000, current_price, atr  # Use account value from config
                )

                position_state['long'] = contracts
                self.entry_price = current_price
                self.entry_atr = atr
                self.entry_date = current_date

                signals.append({
                    'date': current_date,
                    'signal': 1,
                    'price': current_price,
                    'size': contracts,
                    'reason': 'ENTRY'
                })

            elif signal == 'SHORT' and position_state['short'] == 0:
                # Close long if exists
                if position_state['long'] > 0:
                    position_state['long'] = 0

                # Enter short
                atr = self.calculate_atr(current_data).iloc[-1]
                contracts, _ = self.calculate_position_size(
                    1000000, current_price, atr
                )

                position_state['short'] = contracts
                self.entry_price = current_price
                self.entry_atr = atr
                self.entry_date = current_date

                signals.append({
                    'date': current_date,
                    'signal': -1,
                    'price': current_price,
                    'size': contracts,
                    'reason': 'ENTRY'
                })

        return pd.DataFrame(signals)

    def get_features(self):
        """Return required features"""
        return {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/technical/test_regimetry_adaptive_range_strategy.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add strategies/technical/regimetry_adaptive_range_strategy.py tests/technical/test_regimetry_adaptive_range_strategy.py
git commit -m "feat: Implement RegimetryAdaptiveRangeStrategy

- Dynamic regime-based signal generation
- Bull markets: buy pullbacks
- Bear markets: short rallies
- Ranging markets: trade confirmed breakouts
- Risk-based position sizing (2% per trade)
- Dual exit system (regime change + 2 ATR stop loss)"
```

---

## Task 4: Create Integration Test Script

**Files:**
- Create: `scripts/test_rars_strategy.py`

**Step 1: Write integration test script**

Create `scripts/test_rars_strategy.py`:

```python
#!/usr/bin/env python
"""
Integration test for Regimetry Adaptive Range Strategy
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor
from strategies.regime.regimetry_wrapper import RegimetryWrapper
from strategies.technical.regimetry_adaptive_range_strategy import RegimetryAdaptiveRangeStrategy


def load_data():
    """Load TA futures data"""
    print("Loading TA futures data...")
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

    print(f"Loaded {len(daily_df)} daily bars")
    return daily_df


def run_regimetry_pipeline(df):
    """Run regimetry to generate cluster assignments"""
    print("\n" + "=" * 80)
    print("Running Regimetry Pipeline")
    print("=" * 80)

    wrapper = RegimetryWrapper()

    # Use subset for faster testing
    df_test = df.loc['2019-01-01':'2025-03-21']

    try:
        assignments_path = wrapper.run_full_pipeline(
            df_test,
            instrument='TA',
            n_clusters=8,
            window_size=30
        )

        print(f"✓ Regimetry complete: {assignments_path}")

        # Load assignments
        assignments = pd.read_csv(assignments_path)
        print(f"\nCluster assignments:")
        print(assignments.tail(10))

        return assignments

    except Exception as e:
        print(f"✗ Regimetry failed: {e}")
        print("\nUsing mock assignments for testing...")

        # Generate mock assignments
        dates = df_test.index
        n_clusters = 8
        cluster_ids = np.random.choice(n_clusters, size=len(dates))

        mock_assignments = pd.DataFrame({
            'Date': dates,
            'Cluster_ID': cluster_ids
        })

        return mock_assignments


def run_backtest(data, strategy, assignments, initial_cash=1000000):
    """
    Simple backtest executor

    Returns:
        dict: Performance metrics
    """
    print("\nGenerating signals...")

    signals_df = strategy.generate_signals(data, assignments)

    if len(signals_df) == 0:
        print("No signals generated")
        return None

    cash = initial_cash
    position = 0
    portfolio_values = []
    trades = []

    for i in range(1, len(data)):
        price = data['close'].iloc[i]
        date = data.index[i]

        # Get signal for this date
        signal_row = signals_df[signals_df['date'] == date]

        if len(signal_row) > 0:
            signal = signal_row.iloc[0]

            if signal['signal'] == 1 and position == 0:
                # Enter long
                contracts = signal['size']
                position_value = contracts * price
                cash -= position_value
                position = contracts
                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'size': contracts
                })

            elif signal['signal'] == -1 and position == 0:
                # Enter short
                contracts = signal['size']
                cash += contracts * price  # Credit from short sale
                position = -contracts
                trades.append({
                    'date': date,
                    'action': 'SHORT',
                    'price': price,
                    'size': contracts
                })

            elif signal['signal'] == 0 and position != 0:
                # Exit position
                if position > 0:
                    cash += position * price
                else:
                    cash -= abs(position) * price
                position = 0

                if trades:
                    trades[-1]['exit_date'] = date
                    trades[-1]['exit_price'] = price

        # Calculate portfolio value
        if position > 0:
            portfolio_value = cash + position * price
        elif position < 0:
            portfolio_value = cash - abs(position) * price
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)

    # Calculate metrics
    portfolio_series = pd.Series(portfolio_values, index=data.index[1:])
    returns = portfolio_series.pct_change().dropna()

    years = (data.index[-1] - data.index[0]).days / 365.25

    metrics = {
        'final_value': portfolio_values[-1],
        'total_return': (portfolio_values[-1] / initial_cash - 1),
        'annual_return': returns.mean() * 252 if len(returns) > 0 else 0,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(returns),
        'num_trades': len(trades),
        'portfolio_values': portfolio_series,
        'trades': trades
    }

    return metrics


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def print_metrics(metrics, name):
    """Print performance metrics"""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")

    if metrics is None:
        print("No metrics available")
        return

    print(f"Final Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")


def main():
    """Main test function"""
    print("=" * 80)
    print("Regimetry Adaptive Range Strategy - Integration Test")
    print("=" * 80)

    # Load data
    df = load_data()

    # Test period
    test_start = '2019-01-01'
    test_end = '2025-03-21'
    df_test = df.loc[test_start:test_end]

    print(f"\nTest period: {test_start} to {test_end} ({len(df_test)} days)")

    # Run regimetry
    assignments = run_regimetry_pipeline(df_test)

    # Initialize strategy
    config = {
        'lookback_days': 20,
        'window_short': 10,
        'window_long': 30,
        'entry_buffer_atr': 1.0,
        'stop_loss_atr': 2.0,
        'risk_per_trade': 0.02,
        'assignments_path': None  # Not used, passing assignments directly
    }

    strategy = RegimetryAdaptiveRangeStrategy(
        'TA', test_start, test_end, config
    )

    # Run backtest
    metrics = run_backtest(df_test, strategy, assignments)

    # Print results
    print_metrics(metrics, "RARS Performance")

    # Compare to baseline
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Strategy: Regimetry Adaptive Range Strategy")
    print(f"Annual Return: {metrics['annual_return']:.2%}" if metrics else "N/A")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}" if metrics else "N/A")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}" if metrics else "N/A")

    return metrics


if __name__ == '__main__':
    main()
```

**Step 2: Make script executable**

```bash
chmod +x scripts/test_rars_strategy.py
```

**Step 3: Run integration test**

```bash
python scripts/test_rars_strategy.py
```

Expected: Script runs, generates signals, prints performance metrics

**Step 4: Commit**

```bash
git add scripts/test_rars_strategy.py
git commit -m "feat: Add RARS integration test script

- Load TA futures data
- Run regimetry pipeline (or use mock assignments)
- Generate signals and run backtest
- Calculate and display performance metrics
- Support for full backtest pipeline"
```

---

## Task 5: Document Results and Create Final Report

**Files:**
- Create: `docs/rars_results.md`

**Step 1: Run full integration test**

```bash
python scripts/test_rars_strategy.py > docs/rars_test_output.txt 2>&1
```

**Step 2: Create results document**

Create `docs/rars_results.md`:

```markdown
# Regimetry Adaptive Range Strategy - Results Report

**Date:** 2025-03-24
**Strategy:** RARS (Regimetry Adaptive Range Strategy)
**Test Period:** 2019-01-01 to 2025-03-21

---

## Executive Summary

[Fill in after running tests]

### Key Findings

- [Finding 1]
- [Finding 2]
- [Finding 3]

### Performance Summary

| Metric | Value |
|--------|-------|
| Annual Return | X% |
| Sharpe Ratio | X |
| Max Drawdown | X% |
| Number of Trades | X |

---

## Regime Analysis

[Describe detected regimes, their characteristics, and distribution]

---

## Conclusion

[Assessment of whether regime-adaptive approach achieved objectives]

---

## Comparison to Baselines

| Strategy | Annual Return | Sharpe | Max DD |
|----------|---------------|--------|--------|
| RARS | X% | X | X% |
| Buy & Hold | X% | X | X% |
| MA(20) | X% | X | X% |
```

**Step 3: Fill in results from test output**

[Manual step: Copy results from test_output.txt into results.md]

**Step 4: Commit**

```bash
git add docs/rars_results.md docs/rars_test_output.txt
git commit -m "docs: Add RARS test results and analysis

- Complete performance report
- Regime analysis
- Comparison with baseline strategies"
```

---

## Task 6: Final Validation and Clean Up

**Files:**
- Modify: `README.md` (if applicable)

**Step 1: Run all tests**

```bash
pytest tests/regime/ tests/technical/ -v
```

Expected: All tests pass

**Step 2: Verify integration test**

```bash
python scripts/test_rars_strategy.py
```

Expected: Results are consistent

**Step 3: Update project documentation**

Add entry to main README or appropriate documentation:

```markdown
## Regimetry Adaptive Range Strategy (RARS)

Location: `strategies/technical/regimetry_adaptive_range_strategy.py`

Adaptive trading strategy that changes behavior based on market regime:
- **Bull Markets**: Buy pullbacks to recent lows
- **Bear Markets**: Short rallies to recent highs
- **Ranging Markets**: Trade confirmed breakouts

Uses Regimetry (Transformer embeddings + spectral clustering) for regime detection.

Results: See `docs/rars_results.md`
```

**Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: Update README with RARS documentation"
```

---

## Success Criteria

The implementation is successful when:

✅ All unit tests pass (RegimetryWrapper, RegimeMapper, RARS)
✅ Integration test runs without errors
✅ Regime detection produces meaningful classifications
✅ Strategy generates signals in all three market states
✅ Risk management (2% rule, 2 ATR stop) is correctly implemented
✅ Code is documented and committed
✅ Results report is complete

**Stretch Goal:** Annual return > 10% with Sharpe > 1.0

---

## Implementation Notes

**Dependencies:**
- `regimetry` - Regime detection with Transformer embeddings
- `tensorflow>=2.12.0` - Deep learning framework
- `scikit-learn>=1.2.0` - Spectral clustering

**Performance Considerations:**
- Regimetry pipeline can take 5-10 minutes on full dataset
- Consider pre-computing cluster assignments weekly
- Strategy execution is fast once regime labels are loaded

**Known Limitations:**
- Regime detection has lag (window_size bars)
- Cluster IDs need to be mapped to trading states
- Strategy may have few signals in strongly trending/ranging periods

**Future Enhancements:**
- Multi-timeframe regime analysis
- Regime-specific parameter optimization
- Ensemble with traditional indicators
- Machine learning for regime transition prediction
