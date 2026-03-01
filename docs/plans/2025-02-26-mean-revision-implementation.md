# Mean Reversion Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Z-Score based mean reversion trading strategy with multi-layer position sizing for range-bound markets.

**Architecture:** Statistical strategy using Z-Score (standardized deviation from mean) to identify entry points, with dynamic position sizing (50%/75%/100%) based on deviation magnitude and hybrid exit mechanism (reversion + stop loss).

**Tech Stack:** Python 3.8+, pandas, numpy, pytest, existing Qlib trading framework.

---

## Task 1: Create Statistical Strategy Package

**Files:**
- Create: `strategies/statistical/__init__.py`
- Test: `tests/test_statistical_init.py`

**Step 1: Write test for package initialization**

```python
# tests/test_statistical_init.py
def test_statistical_package_exists():
    """Test that statistical strategy package can be imported"""
    import strategies.statistical
    assert strategies.statistical is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_statistical_init.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'strategies.statistical'"

**Step 3: Create __init__.py**

```python
# strategies/statistical/__init__.py
"""
Statistical trading strategies including mean reversion.
"""
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_statistical_init.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/statistical/__init__.py tests/test_statistical_init.py
git commit -m "feat: add statistical strategy package structure"
```

---

## Task 2: Implement Z-Score Calculation

**Files:**
- Create: `strategies/statistical/mean_reversion.py`
- Test: `tests/test_mean_reversion.py`

**Step 1: Write failing test for Z-Score calculation**

```python
# tests/test_mean_reversion.py
import pytest
import pandas as pd
import numpy as np
from strategies.statistical.mean_reversion import MeanReversionStrategy

def test_calculate_zscore_basic():
    """Test Z-Score calculation with simple data"""
    # Create data where mean=100, std should be calculable
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })

    config = {'lookback_period': 5}
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    zscore = strategy.calculate_zscore(data)

    # Verify output
    assert len(zscore) == len(data)
    assert zscore.dtype == np.float64
    # First 4 values should be NaN (insufficient data for 5-period window)
    assert pd.isna(zscore.iloc[0])
    assert pd.isna(zscore.iloc[3])
    # 5th value should be valid
    assert not pd.isna(zscore.iloc[4])
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mean_reversion.py::test_calculate_zscore_basic -v
```
Expected: FAIL with "MeanReversionStrategy not defined" or "calculate_zscore method not found"

**Step 3: Implement MeanReversionStrategy skeleton with calculate_zscore**

```python
# strategies/statistical/mean_reversion.py
from strategies.base.base_strategy import FuturesStrategy
import pandas as pd
import numpy as np

class MeanReversionStrategy(FuturesStrategy):
    """Z-Score based mean reversion strategy for range-bound markets"""

    def __init__(self, instrument, start_date, end_date, config):
        """
        Initialize Mean Reversion Strategy

        Parameters:
            instrument: Instrument code
            start_date: Start date
            end_date: End date
            config: Configuration dictionary containing:
                - lookback_period: Rolling window for Z-Score calculation
                - entry_threshold: Z-Score threshold for entry
                - level1_threshold: 50% position threshold
                - level2_threshold: 75% position threshold
                - level3_threshold: 100% position threshold
                - exit_threshold: Z-Score threshold for exit
                - max_hold_period: Maximum holding period in bars
                - stop_multiplier: Stop loss multiplier
        """
        super().__init__(instrument, start_date, end_date, config)
        self.lookback_period = config.get("lookback_period", 20)
        self.entry_threshold = config.get("entry_threshold", 1.5)
        self.level1_threshold = config.get("level1_threshold", 1.5)
        self.level2_threshold = config.get("level2_threshold", 2.0)
        self.level3_threshold = config.get("level3_threshold", 2.5)
        self.exit_threshold = config.get("exit_threshold", 0.5)
        self.max_hold_period = config.get("max_hold_period", 50)
        self.stop_multiplier = config.get("stop_multiplier", 1.5)

    def calculate_zscore(self, data):
        """
        Calculate Z-Score for price series

        Z-Score = (Price - Rolling Mean) / Rolling Std Dev

        Parameters:
            data: DataFrame with 'close' column

        Returns:
            Series: Z-Score values (NaN for insufficient data)
        """
        close = data['close']
        rolling_mean = close.rolling(self.lookback_period).mean()
        rolling_std = close.rolling(self.lookback_period).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (close - rolling_mean) / rolling_std

        # Clip extreme values
        zscore = zscore.clip(-5, 5)

        return zscore
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mean_reversion.py::test_calculate_zscore_basic -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/statistical/mean_reversion.py tests/test_mean_reversion.py
git commit -m "feat: add Z-Score calculation to MeanReversionStrategy"
```

---

## Task 3: Implement Multi-Layer Position Sizing

**Files:**
- Modify: `strategies/statistical/mean_reversion.py`
- Test: `tests/test_mean_reversion.py`

**Step 1: Write failing test for position sizing**

```python
# tests/test_mean_reversion.py
def test_get_position_size_multi_layer():
    """Test multi-layer position sizing based on Z-Score"""
    config = {
        'level1_threshold': 1.5,
        'level2_threshold': 2.0,
        'level3_threshold': 2.5
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Test different Z-Score levels
    assert strategy.get_position_size(0) == 0.0  # No deviation
    assert strategy.get_position_size(1.0) == 0.0  # Below threshold
    assert strategy.get_position_size(1.5) == 0.5  # Level 1
    assert strategy.get_position_size(1.8) == 0.5  # Level 1
    assert strategy.get_position_size(2.0) == 0.75  # Level 2
    assert strategy.get_position_size(2.3) == 0.75  # Level 2
    assert strategy.get_position_size(2.5) == 1.0  # Level 3
    assert strategy.get_position_size(3.0) == 1.0  # Level 3

    # Test short positions (negative Z-Score)
    assert strategy.get_position_size(-1.5) == 0.5
    assert strategy.get_position_size(-2.0) == 0.75
    assert strategy.get_position_size(-2.5) == 1.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mean_reversion.py::test_get_position_size_multi_layer -v
```
Expected: FAIL with "get_position_size method not found"

**Step 3: Implement get_position_size method**

Add to `MeanReversionStrategy` class:

```python
def get_position_size(self, zscore):
    """
    Calculate position size based on Z-Score deviation magnitude

    Multi-layer position sizing:
    - |z| < level1: 0% (no trade)
    - level1 ≤ |z| < level2: 50%
    - level2 ≤ |z| < level3: 75%
    - |z| ≥ level3: 100%

    Parameters:
        zscore: Current Z-Score value (can be float or Series)

    Returns:
        float or Series: Position size ratio (0.0 to 1.0)
    """
    abs_z = abs(zscore)

    if abs_z < self.level1_threshold:
        return 0.0
    elif abs_z < self.level2_threshold:
        return 0.5
    elif abs_z < self.level3_threshold:
        return 0.75
    else:
        return 1.0
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mean_reversion.py::test_get_position_size_multi_layer -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/statistical/mean_reversion.py tests/test_mean_reversion.py
git commit -m "feat: add multi-layer position sizing"
```

---

## Task 4: Implement Exit Logic

**Files:**
- Modify: `strategies/statistical/mean_reversion.py`
- Test: `tests/test_mean_reversion.py`

**Step 1: Write failing test for exit conditions**

```python
# tests/test_mean_reversion.py
def test_should_exit_reversion():
    """Test exit when Z-Score reverts to mean"""
    config = {
        'exit_threshold': 0.5,
        'stop_multiplier': 1.5,
        'max_hold_period': 50
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Entered at Z-Score = 2.0
    # Should exit when Z-Score returns to ±0.5
    assert strategy.should_exit(current_zscore=0.3, entry_zscore=2.0, bars_held=10) == True
    assert strategy.should_exit(current_zscore=-0.3, entry_zscore=-2.0, bars_held=10) == True

    # Should NOT exit when still far from mean
    assert strategy.should_exit(current_zscore=1.5, entry_zscore=2.0, bars_held=10) == False

def test_should_exit_stop_loss():
    """Test exit when stop loss is hit"""
    config = {'stop_multiplier': 1.5}
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Entered at Z-Score = 2.0, stop at 2.0 × 1.5 = 3.0
    assert strategy.should_exit(current_zscore=3.1, entry_zscore=2.0, bars_held=10) == True
    assert strategy.should_exit(current_zscore=2.9, entry_zscore=2.0, bars_held=10) == False

def test_should_exit_max_holding_period():
    """Test exit when max holding period exceeded"""
    config = {'max_hold_period': 50}
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Should exit after 50 bars regardless of Z-Score
    assert strategy.should_exit(current_zscore=1.0, entry_zscore=2.0, bars_held=51) == True
    assert strategy.should_exit(current_zscore=1.0, entry_zscore=2.0, bars_held=50) == False
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mean_reversion.py::test_should_exit_reversion -v
```
Expected: FAIL with "should_exit method not found"

**Step 3: Implement should_exit method**

Add to `MeanReversionStrategy` class:

```python
def should_exit(self, current_zscore, entry_zscore, bars_held):
    """
    Check if position should be closed

    Exit conditions (priority order):
    1. Stop loss hit (Z-Score moved against position)
    2. Z-Score reverted to mean (profit taking)
    3. Max holding period exceeded (time-based exit)

    Parameters:
        current_zscore: Current Z-Score value
        entry_zscore: Z-Score at position entry
        bars_held: Number of bars since entry

    Returns:
        bool: True if should exit, False otherwise
    """
    # Check max holding period first (time-based exit)
    if bars_held >= self.max_hold_period:
        return True

    # Calculate stop loss threshold
    stop_threshold = abs(entry_zscore) * self.stop_multiplier

    # Check stop loss (price moved further against us)
    if abs(current_zscore) > stop_threshold:
        return True

    # Check reversion to mean (profit taking)
    if abs(current_zscore) < self.exit_threshold:
        return True

    return False
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mean_reversion.py::test_should_exit_reversion -v
pytest tests/test_mean_reversion.py::test_should_exit_stop_loss -v
pytest tests/test_mean_reversion.py::test_should_exit_max_holding_period -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add strategies/statistical/mean_reversion.py tests/test_mean_reversion.py
git commit -m "feat: add exit logic (reversion, stop loss, time-based)"
```

---

## Task 5: Implement Signal Generation

**Files:**
- Modify: `strategies/statistical/mean_reversion.py`
- Test: `tests/test_mean_reversion.py`

**Step 1: Write failing test for signal generation**

```python
# tests/test_mean_reversion.py
def test_generate_signals_basic():
    """Test signal generation with oscillating price"""
    # Create price data that oscillates
    data = pd.DataFrame({
        'close': [100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100]
    })

    config = {
        'lookback_period': 5,
        'entry_threshold': 1.0  # Lower threshold for test data
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    signals = strategy.generate_signals(data)

    # Verify output format
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert 'target_position' in signals.columns
    assert len(signals) == len(data)

    # Verify signal values are -1, 0, or 1
    assert signals['signal'].isin([-1, 0, 1]).all()

    # Verify position sizes are 0.0, 0.5, 0.75, or 1.0
    assert signals['target_position'].isin([0.0, 0.5, 0.75, 1.0]).all()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mean_reversion.py::test_generate_signals_basic -v
```
Expected: FAIL with "generate_signals method not found" or incorrect output format

**Step 3: Implement generate_signals method**

Add to `MeanReversionStrategy` class:

```python
def generate_signals(self, data):
    """
    Generate trading signals based on Z-Score mean reversion

    Returns DataFrame with:
    - signal: 1 (long), -1 (short), 0 (no position)
    - target_position: 0.0 to 1.0 (position size ratio)

    Parameters:
        data: DataFrame with 'close' column

    Returns:
        DataFrame: Signals and target positions
    """
    # Calculate Z-Score
    zscore = self.calculate_zscore(data)

    # Initialize output DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['zscore'] = zscore
    signals['signal'] = 0
    signals['target_position'] = 0.0

    # Generate signals for valid Z-Scores
    valid_mask = ~pd.isna(zscore)

    for i in range(len(data)):
        if not valid_mask.iloc[i]:
            continue

        current_z = zscore.iloc[i]

        # Determine signal direction and position size
        if current_z > 0:
            # Price above mean → potential short
            position_size = self.get_position_size(current_z)
            if position_size > 0:
                signals.loc[signals.index[i], 'signal'] = -1
                signals.loc[signals.index[i], 'target_position'] = position_size
        else:
            # Price below mean → potential long
            position_size = self.get_position_size(current_z)
            if position_size > 0:
                signals.loc[signals.index[i], 'signal'] = 1
                signals.loc[signals.index[i], 'target_position'] = position_size

    return signals
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mean_reversion.py::test_generate_signals_basic -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/statistical/mean_reversion.py tests/test_mean_reversion.py
git commit -m "feat: add signal generation with Z-Score"
```

---

## Task 6: Implement Volatility Adjustment

**Files:**
- Modify: `strategies/statistical/mean_reversion.py`
- Test: `tests/test_mean_reversion.py`

**Step 1: Write failing test for volatility adjustment**

```python
# tests/test_mean_reversion.py
def test_volatility_adjustment():
    """Test position size adjustment for volatility"""
    from strategies.risk.risk_manager import RiskManager

    config = {
        'use_volatility_adjustment': True,
        'atr_period': 14,
        'atr_lookback': 100,
        'volatility_threshold': 80
    }
    strategy = MeanReversionStrategy('TEST', '2020-01-01', '2020-12-31', config)

    # Create data with high, low, close
    data = pd.DataFrame({
        'close': [100] * 150,
        'high': [101] * 150,
        'low': [99] * 150
    })

    # Get position with volatility adjustment
    base_position = 0.5
    adjusted_position = strategy.get_position_with_volatility(
        data, current_bar=149, base_position=base_position
    )

    # Should return a float between 0 and base_position
    assert isinstance(adjusted_position, float)
    assert 0 <= adjusted_position <= base_position
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mean_reversion.py::test_volatility_adjustment -v
```
Expected: FAIL with "get_position_with_volatility method not found"

**Step 3: Implement volatility adjustment**

Add to `__init__` method:

```python
# Check if volatility adjustment should be used
self.use_volatility_adjustment = config.get("use_volatility_adjustment", False)

# Initialize RiskManager if volatility adjustment enabled
if self.use_volatility_adjustment:
    from strategies.risk.risk_manager import RiskManager
    self.risk_manager = RiskManager(config)
else:
    self.risk_manager = None
```

Add method to class:

```python
def get_position_with_volatility(self, data, current_bar, base_position):
    """
    Apply volatility adjustment to position size

    Uses RiskManager to reduce position during high volatility periods.

    Parameters:
        data: DataFrame with OHLC data
        current_bar: Current bar index
        base_position: Base position ratio

    Returns:
        float: Adjusted position ratio
    """
    if self.risk_manager is not None:
        data_slice = data.iloc[:current_bar + 1]
        adjusted_ratio = self.risk_manager.calculate_volatility_adjustment(
            data_slice, base_position
        )
        return adjusted_ratio
    else:
        return base_position
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mean_reversion.py::test_volatility_adjustment -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/statistical/mean_reversion.py tests/test_mean_reversion.py
git commit -m "feat: add volatility adjustment integration"
```

---

## Task 7: Update Strategy Factory

**Files:**
- Modify: `strategies/strategy_factory.py`
- Test: `tests/test_strategy_factory.py`

**Step 1: Write failing test for factory registration**

```python
# tests/test_strategy_factory.py
def test_create_mean_reversion_strategy():
    """Test creating mean reversion strategy via factory"""
    config = {
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'lookback_period': 20,
        'entry_threshold': 1.5
    }

    strategy = StrategyFactory.create_strategy('mean_reversion', config)

    assert strategy is not None
    assert hasattr(strategy, 'calculate_zscore')
    assert hasattr(strategy, 'get_position_size')
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_strategy_factory.py::test_create_mean_reversion_strategy -v
```
Expected: FAIL with "Unknown strategy type: mean_reversion"

**Step 3: Register strategy in factory**

Modify `strategies/strategy_factory.py`:

Add import at top:
```python
from strategies.statistical.mean_reversion import MeanReversionStrategy
```

Add to `create_strategy` method:

```python
@staticmethod
def create_strategy(strategy_type, config):
    """Factory method to create strategy instances"""
    if strategy_type == 'ma_cross':
        return MAStrategy(
            config.get('instrument', 'TEST'),
            config.get('start_date', '2020-01-01'),
            config.get('end_date', '2020-12-31'),
            config
        )
    elif strategy_type == 'mean_reversion':
        return MeanReversionStrategy(
            config.get('instrument', 'TEST'),
            config.get('start_date', '2020-01-01'),
            config.get('end_date', '2020-12-31'),
            config
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_strategy_factory.py::test_create_mean_reversion_strategy -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/strategy_factory.py tests/test_strategy_factory.py
git commit -m "feat: register mean reversion strategy in factory"
```

---

## Task 8: Update CLI Script

**Files:**
- Modify: `scripts/run_backtest.py`

**Step 1: Verify CLI can handle new parameters**

Test manually:
```bash
cd mean-reversion
python scripts/run_backtest.py --instrument TA --strategy mean_reversion --start 2020-01-01 --end 2020-12-31 --lookback-period 20 --entry-threshold 1.5
```

Expected: Should work without errors

**Step 2: Add mean reversion specific parameters to CLI**

Modify `scripts/run_backtest.py`, add to argument parser:

```python
# Mean reversion parameters
parser.add_argument('--lookback-period', type=int, default=20,
                    help='Z-Score lookback period (default 20)')
parser.add_argument('--entry-threshold', type=float, default=1.5,
                    help='Z-Score entry threshold (default 1.5)')
parser.add_argument('--exit-threshold', type=float, default=0.5,
                    help='Z-Score exit threshold (default 0.5)')
```

Add to config dictionary:
```python
# Add mean reversion parameters
config['lookback_period'] = args.lookback_period
config['entry_threshold'] = args.entry_threshold
config['exit_threshold'] = args.exit_threshold
```

**Step 3: Commit**

```bash
git add scripts/run_backtest.py
git commit -m "feat: add mean reversion parameters to CLI"
```

---

## Task 9: Run Backtest and Validate

**Files:**
- Test: Manual backtest execution

**Step 1: Run backtest on 2020 data**

```bash
cd mean-reversion
python scripts/run_backtest.py \
  --instrument TA \
  --strategy mean_reversion \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --lookback-period 20 \
  --entry-threshold 1.5
```

**Step 2: Compare with MA strategy results**

MA Strategy (2020):
- Total return: -100%
- Sharpe ratio: -1.82
- Max drawdown: -100%

Expected improvements for Mean Reversion:
- Total return: > -50%
- Sharpe ratio: > -1.0
- Max drawdown: < -70%

**Step 3: Document results**

Create `docs/mean_reversion_backtest_results.md` with:
- Backtest configuration
- Performance metrics
- Comparison with MA strategy
- Analysis of results

**Step 4: Commit**

```bash
git add docs/mean_reversion_backtest_results.md
git commit -m "docs: add mean reversion backtest results"
```

---

## Task 10: Parameter Optimization

**Files:**
- Script: `scripts/optimize_mean_reversion.py`

**Step 1: Create optimization script**

```python
#!/usr/bin/env python
"""
Optimize mean reversion strategy parameters
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import ContinuousContractProcessor
from strategies.statistical.mean_reversion import MeanReversionStrategy
from executors.parameter_optimizer import ParameterOptimizer
import pandas as pd

# Load data
processor = ContinuousContractProcessor('data/raw/TA.csv')
data = processor.process(adjust_price=True)
data = processor.load_data(start_date='2020-01-01', end_date='2020-12-31')

# Base configuration
base_config = {
    'instrument': 'TA',
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'initial_cash': 1000000,
    'position_ratio': 0.3,
    'commission_rate': 0.0001,
}

# Parameter grid
param_grid = {
    'lookback_period': [15, 20, 25],
    'entry_threshold': [1.2, 1.5, 1.8],
    'exit_threshold': [0.3, 0.5, 0.7],
}

# Create optimizer
optimizer = ParameterOptimizer(MeanReversionStrategy, data, base_config)

# Run optimization
best_result, results_df = optimizer.grid_search(
    param_grid,
    metric='sharpe_ratio',
    verbose=True
)

# Print results
optimizer.print_summary(best_result)
optimizer.save_results(results_df, 'reports/TA_mean_reversion_optimization.csv')
```

**Step 2: Run optimization**

```bash
cd mean-reversion
python scripts/optimize_mean_reversion.py
```

**Step 3: Analyze optimal parameters**

Review `reports/TA_mean_reversion_optimization.csv` and document best parameters.

**Step 4: Commit**

```bash
git add scripts/optimize_mean_reversion.py
git commit -m "feat: add mean reversion parameter optimization script"
```

---

## Success Criteria

After completing all tasks:

- [ ] All 10 tasks implemented
- [ ] All unit tests pass
- [ ] Backtest shows improvement over MA strategy
- [ ] Sharpe ratio > -1.0 (vs MA's -1.82)
- [ ] Max drawdown < -70% (vs MA's -100%)
- [ ] No total loss of capital
- [ ] Code committed and pushed to GitHub

---

## References

- Design document: `docs/plans/2025-02-26-mean-reversion-design.md`
- Base strategy: `strategies/base/base_strategy.py`
- Risk manager: `strategies/risk/risk_manager.py`
- Backtest executor: `executors/backtest_executor.py`
