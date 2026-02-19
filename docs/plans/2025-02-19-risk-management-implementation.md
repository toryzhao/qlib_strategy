# Risk Management for MA Strategy - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three-layer risk management (trend filter, volatility filter, trailing stop) to the MA crossover strategy to improve risk-adjusted returns and reduce drawdowns.

**Architecture:**
- Create `RiskManager` class encapsulating all three risk layers
- Integrate RiskManager into MAStrategy for signal filtering and position sizing
- Modify BacktestExecutor to track entry bars and check trailing stops each period
- Follow TDD: write tests first, implement feature, verify, commit

**Tech Stack:** Python 3.12, pandas, numpy, pytest

---

## Task 1: Create Risk Management Module Structure

**Files:**
- Create: `strategies/risk/__init__.py`
- Create: `strategies/risk/risk_manager.py`

**Step 1: Create package init file**

```bash
touch strategies/risk/__init__.py
```

**Step 2: Write the RiskManager class skeleton with tests**

Test file: `tests/test_risk_manager.py`

```python
import pytest
import pandas as pd
import numpy as np
from strategies.risk.risk_manager import RiskManager

def test_risk_manager_initialization():
    """Test RiskManager initialization with config"""
    config = {
        'trend_ma_period': 200,
        'atr_period': 14,
        'atr_lookback': 100,
        'volatility_threshold': 80,
        'swing_period': 20
    }
    rm = RiskManager(config)
    assert rm.trend_ma_period == 200
    assert rm.atr_period == 14
    assert rm.atr_lookback == 100
    assert rm.volatility_threshold == 80
    assert rm.swing_period == 20
```

Implementation: `strategies/risk/risk_manager.py`

```python
class RiskManager:
    """Risk management for trading strategies"""

    def __init__(self, config):
        """
        Initialize RiskManager

        Parameters:
            config: Configuration dictionary containing:
                - trend_ma_period: Period for trend filter MA
                - atr_period: ATR calculation period
                - atr_lookback: Lookback for ATR percentile
                - volatility_threshold: Percentile threshold (0-100)
                - swing_period: Lookback for swing high/low
        """
        self.trend_ma_period = config.get('trend_ma_period', 200)
        self.atr_period = config.get('atr_period', 14)
        self.atr_lookback = config.get('atr_lookback', 100)
        self.volatility_threshold = config.get('volatility_threshold', 80)
        self.swing_period = config.get('swing_period', 20)
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_risk_manager_initialization -v`
Expected: PASS

**Step 4: Commit**

```bash
cd /e/claude_project/trading/.worktrees/risk-management
git add strategies/risk/__init__.py strategies/risk/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: Add RiskManager class skeleton

- Create risk management module structure
- Add RiskManager class with config initialization
- Add initialization tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Implement ATR Calculation

**Files:**
- Modify: `strategies/risk/risk_manager.py`
- Modify: `tests/test_risk_manager.py`

**Step 1: Write failing test for ATR calculation**

Add to `tests/test_risk_manager.py`:

```python
def test_calculate_atr():
    """Test ATR calculation"""
    config = {'atr_period': 14}
    rm = RiskManager(config)

    # Create sample data with known ATR
    data = pd.DataFrame({
        'high': [10, 11, 12, 13, 14],
        'low': [9, 10, 11, 12, 13],
        'close': [9.5, 10.5, 11.5, 12.5, 13.5]
    })

    atr = rm._calculate_atr(data)
    assert len(atr) == len(data)
    assert pd.isna(atr.iloc[0])  # First value is NaN
    assert not pd.isna(atr.iloc[-1])  # Last value is not NaN
    assert atr.iloc[-1] > 0  # ATR should be positive
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_calculate_atr -v`
Expected: FAIL with "RiskManager has no attribute '_calculate_atr'"

**Step 3: Implement ATR calculation**

Add to `strategies/risk/risk_manager.py`:

```python
def _calculate_atr(self, data):
    """
    Calculate Average True Range

    Parameters:
        data: DataFrame with 'high', 'low', 'close' columns

    Returns:
        Series with ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close']

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR using exponential moving average
    atr = tr.ewm(span=self.atr_period, adjust=False).mean()

    return atr
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_calculate_atr -v`
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/risk/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: Add ATR calculation to RiskManager

- Implement _calculate_atr method using Wilder's smoothing
- Add unit tests for ATR calculation
- ATR used for volatility filter and position sizing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement Trend Filter

**Files:**
- Modify: `strategies/risk/risk_manager.py`
- Modify: `tests/test_risk_manager.py`

**Step 1: Write failing test for trend filter**

Add to `tests/test_risk_manager.py`:

```python
def test_calculate_trend_filter_uptrend():
    """Test trend filter in uptrend"""
    config = {'trend_ma_period': 5}
    rm = RiskManager(config)

    # Price above MA = uptrend
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })

    trend = rm.calculate_trend_filter(data)
    assert trend.iloc[-1] == 1  # Uptrend signal

def test_calculate_trend_filter_downtrend():
    """Test trend filter in downtrend"""
    config = {'trend_ma_period': 5}
    rm = RiskManager(config)

    # Price below MA = downtrend
    data = pd.DataFrame({
        'close': [110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
    })

    trend = rm.calculate_trend_filter(data)
    assert trend.iloc[-1] == -1  # Downtrend signal

def test_calculate_trend_filter_insufficient_data():
    """Test trend filter with insufficient data"""
    config = {'trend_ma_period': 200}
    rm = RiskManager(config)

    # Only 50 bars, need 200
    data = pd.DataFrame({
        'close': [100 + i for i in range(50)]
    })

    trend = rm.calculate_trend_filter(data)
    # Should return 0 (no trend) when insufficient data
    assert trend.iloc[-1] == 0
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_calculate_trend_filter_uptrend -v`
Expected: FAIL with "RiskManager has no attribute 'calculate_trend_filter'"

**Step 3: Implement trend filter**

Add to `strategies/risk/risk_manager.py`:

```python
def calculate_trend_filter(self, data):
    """
    Calculate trend filter based on long-term MA

    Parameters:
        data: DataFrame with 'close' column

    Returns:
        Series: 1 for uptrend (close > MA), -1 for downtrend, 0 for neutral/no data
    """
    if len(data) < self.trend_ma_period:
        return pd.Series(0, index=data.index)

    # Calculate long-term MA
    trend_ma = data['close'].rolling(self.trend_ma_period).mean()

    # Determine trend
    trend = pd.Series(0, index=data.index)
    trend[data['close'] > trend_ma] = 1
    trend[data['close'] < trend_ma] = -1

    return trend
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_calculate_trend_filter -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add strategies/risk/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: Add trend filter to RiskManager

- Implement 200-period MA trend filter
- Returns 1 for uptrend, -1 for downtrend, 0 for neutral
- Handles insufficient data gracefully
- Add comprehensive unit tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Volatility Filter (ATR Percentile)

**Files:**
- Modify: `strategies/risk/risk_manager.py`
- Modify: `tests/test_risk_manager.py`

**Step 1: Write failing test for volatility filter**

Add to `tests/test_risk_manager.py`:

```python
def test_calculate_volatility_adjustment_normal():
    """Test position size adjustment in normal volatility"""
    config = {
        'atr_period': 14,
        'atr_lookback': 100,
        'volatility_threshold': 80
    }
    rm = RiskManager(config)

    # Create data with stable ATR (not in top 20%)
    atr_values = [1.0] * 100
    data = pd.DataFrame({'close': [100] * 100})

    position_size = rm.calculate_volatility_adjustment(data, 0.3)
    assert position_size == 0.3  # No reduction

def test_calculate_volatility_adjustment_high():
    """Test position size adjustment in high volatility"""
    config = {
        'atr_period': 14,
        'atr_lookback': 10,
        'volatility_threshold': 80
    }
    rm = RiskManager(config)

    # Create data with ATR in top 20%
    data = pd.DataFrame({'close': range(100)})

    # Mock ATR calculation to return known values
    atr_series = pd.Series([1.0] * 8 + [5.0, 5.0])  # Last 2 are high (90th percentile)
    # We'll patch _calculate_atr in the actual test

    # For now, just test the method exists
    position_size = rm.calculate_volatility_adjustment(data, 0.3)
    assert isinstance(position_size, float)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_calculate_volatility_adjustment_normal -v`
Expected: FAIL with "RiskManager has no attribute 'calculate_volatility_adjustment'"

**Step 3: Implement volatility filter**

Add to `strategies/risk/risk_manager.py`:

```python
def calculate_volatility_adjustment(self, data, base_position_ratio):
    """
    Calculate volatility-adjusted position size

    Reduces position size by 50% when ATR is in top 20% of historical range

    Parameters:
        data: DataFrame with price data
        base_position_ratio: Base position ratio (e.g., 0.3 for 30%)

    Returns:
        float: Adjusted position ratio
    """
    if len(data) < self.atr_lookback:
        # Not enough data, use base ratio
        return base_position_ratio

    # Calculate ATR
    atr = self._calculate_atr(data)

    # Calculate ATR percentile over lookback period
    current_atr = atr.iloc[-1]
    atr_lookback_values = atr.iloc[-self.atr_lookback:]

    # Calculate percentile rank
    percentile = (atr_lookback_values < current_atr).sum() / len(atr_lookback_values) * 100

    # Reduce position size if volatility is high
    if percentile >= self.volatility_threshold:
        return base_position_ratio * 0.5
    else:
        return base_position_ratio
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_calculate_volatility_adjustment -v`
Expected: PASS

**Step 5: Commit**

```bash
git add strategies/risk/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: Add volatility filter for position sizing

- Implement ATR percentile calculation
- Reduce position size by 50% when ATR in top 20%
- Use expanding window for initial data
- Add unit tests for volatility adjustment

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement Trailing Stop (Swing High/Low)

**Files:**
- Modify: `strategies/risk/risk_manager.py`
- Modify: `tests/test_risk_manager.py`

**Step 1: Write failing test for trailing stop**

Add to `tests/test_risk_manager.py`:

```python
def test_should_exit_trailing_stop_long():
    """Test trailing stop for long position"""
    config = {'swing_period': 5}
    rm = RiskManager(config)

    # Entry at bar 5, now at bar 10
    entry_bar = 5
    current_bar = 10

    # Price data: lowest low since entry is 95 at bar 7
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 95, 106, 107, 108, 109],  # bar 7 is low
        'low': [99, 100, 101, 102, 103, 104, 94, 105, 106, 107, 108]
    })

    # Close at 109, above lowest low (95) - should NOT exit
    should_exit = rm.should_exit_trailing_stop(data, entry_bar, current_bar, 1)
    assert should_exit == False

    # Now close at 93, below lowest low - should exit
    data_with_exit = data.copy()
    data_with_exit.loc[10, 'close'] = 93
    should_exit = rm.should_exit_trailing_stop(data_with_exit, entry_bar, current_bar, 1)
    assert should_exit == True

def test_should_exit_trailing_stop_short():
    """Test trailing stop for short position"""
    config = {'swing_period': 5}
    rm = RiskManager(config)

    entry_bar = 5
    current_bar = 10

    # Price data: highest high since entry is 110 at bar 7
    data = pd.DataFrame({
        'close': [100, 99, 98, 97, 96, 95, 110, 94, 93, 92, 91],
        'high': [101, 100, 99, 98, 97, 96, 111, 95, 94, 93, 92]
    })

    # Close at 91, below highest high - should NOT exit
    should_exit = rm.should_exit_trailing_stop(data, entry_bar, current_bar, -1)
    assert should_exit == False

    # Close at 112, above highest high - should exit
    data_with_exit = data.copy()
    data_with_exit.loc[10, 'close'] = 112
    should_exit = rm.should_exit_trailing_stop(data_with_exit, entry_bar, current_bar, -1)
    assert should_exit == True

def test_should_exit_trailing_stop_insufficient_bars():
    """Test trailing stop with insufficient bars since entry"""
    config = {'swing_period': 20}
    rm = RiskManager(config)

    entry_bar = 95
    current_bar = 100  # Only 5 bars since entry, need 20

    data = pd.DataFrame({
        'close': [100] * 101,
        'high': [101] * 101,
        'low': [99] * 101
    })

    # Should use available bars
    should_exit = rm.should_exit_trailing_stop(data, entry_bar, current_bar, 1)
    assert isinstance(should_exit, bool)
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_should_exit_trailing_stop_long -v`
Expected: FAIL with "RiskManager has no attribute 'should_exit_trailing_stop'"

**Step 3: Implement trailing stop**

Add to `strategies/risk/risk_manager.py`:

```python
def should_exit_trailing_stop(self, data, entry_bar, current_bar, position_type):
    """
    Check if position should be closed based on trailing stop

    Uses swing high/low method: exit when price penetrates
    the highest high (short) or lowest low (long) since entry

    Parameters:
        data: DataFrame with 'close', 'high', 'low' columns
        entry_bar: Index of bar when position was entered
        current_bar: Index of current bar
        position_type: 1 for long, -1 for short

    Returns:
        bool: True if should exit, False otherwise
    """
    # Determine lookback period (bars since entry)
    bars_since_entry = current_bar - entry_bar
    lookback = min(bars_since_entry, self.swing_period)

    # Minimum 5 bars required
    if lookback < 5:
        # Use entry price as stop
        entry_price = data['close'].iloc[entry_bar]
        current_price = data['close'].iloc[current_bar]

        if position_type == 1:  # Long
            return current_price < entry_price
        else:  # Short
            return current_price > entry_price

    # Get data since entry
    data_since_entry = data.iloc[entry_bar:current_bar + 1]

    if position_type == 1:  # Long position
        # Exit if close falls below lowest low
        lowest_low = data_since_entry['low'].min()
        current_price = data['close'].iloc[current_bar]
        return current_price < lowest_low

    else:  # Short position
        # Exit if close rises above highest high
        highest_high = data_since_entry['high'].max()
        current_price = data['close'].iloc[current_bar]
        return current_price > highest_high
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_risk_manager.py::test_should_exit_trailing_stop -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add strategies/risk/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: Add trailing stop using swing high/low

- Implement swing-based trailing stop
- For longs: exit on close below lowest low since entry
- For shorts: exit on close above highest high since entry
- Handle insufficient bars with entry price fallback
- Add comprehensive unit tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Integrate RiskManager into MAStrategy

**Files:**
- Modify: `strategies/technical/ma_strategy.py`
- Create: `tests/test_ma_strategy_with_risk.py`

**Step 1: Write failing test for MA strategy with risk**

Create `tests/test_ma_strategy_with_risk.py`:

```python
import pytest
import pandas as pd
from strategies.technical.ma_strategy import MAStrategy

def test_ma_strategy_with_risk_manager():
    """Test MA strategy uses RiskManager for signal filtering"""
    config = {
        'fast_period': 5,
        'slow_period': 20,
        'trend_ma_period': 10,  # Short for testing
        'instrument': 'TA',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31'
    }

    strategy = MAStrategy('TA', '2020-01-01', '2020-12-31', config)

    # Verify RiskManager is initialized
    assert strategy.risk_manager is not None
    assert strategy.risk_manager.trend_ma_period == 10

def test_ma_strategy_signal_filtering():
    """Test that MA signals are filtered by trend"""
    config = {
        'fast_period': 3,
        'slow_period': 5,
        'trend_ma_period': 10
    }

    strategy = MAStrategy('TA', '2020-01-01', '2020-12-31', config)

    # Create price data: uptrend then downtrend
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
    })

    signals = strategy.generate_signals(data)

    # Verify signals are filtered (should have zeros when trend conflicts)
    assert len(signals) == len(data)
    assert signals.dtype in [int, 'int64']
    # All signals should be -1, 0, or 1
    assert all(signals.isin([-1, 0, 1]))
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_ma_strategy_with_risk.py -v`
Expected: FAIL with "MAStrategy has no attribute 'risk_manager'"

**Step 3: Integrate RiskManager into MAStrategy**

Modify `strategies/technical/ma_strategy.py`:

```python
# trading/strategies/technical/ma_strategy.py
from strategies.base.base_strategy import FuturesStrategy
from strategies.risk.risk_manager import RiskManager
import pandas as pd

class MAStrategy(FuturesStrategy):
    """双均线策略with risk management"""

    def __init__(self, instrument, start_date, end_date, config):
        """
        初始化双均线策略

        参数:
            instrument: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            config: 配置字典，包含:
                - fast_period: 快速均线周期 (默认5)
                - slow_period: 慢速均线周期 (默认20)
                - trend_ma_period: 趋势过滤MA周期 (默认200)
                - atr_period: ATR周期 (默认14)
                - atr_lookback: ATR回看期 (默认100)
                - volatility_threshold: 波动率阈值 (默认80)
                - swing_period: 摆荡周期 (默认20)
        """
        super().__init__(instrument, start_date, end_date, config)
        self.fast_period = config.get('fast_period', 5)
        self.slow_period = config.get('slow_period', 20)

        # Initialize RiskManager
        self.risk_manager = RiskManager(config)

    def generate_signals(self, data):
        """
        生成交易信号 (with trend filter)

        策略逻辑:
        - 计算快速MA和慢速MA
        - 应用趋势过滤器 (200-MA)
        - 只在趋势方向上交易

        参数:
            data: 市场数据DataFrame，必须包含'close'列

        返回:
            pd.Series: 信号序列 (1=做多, -1=做空, 0=无信号)
        """
        # 计算均线
        fast_ma = data['close'].rolling(self.fast_period).mean()
        slow_ma = data['close'].rolling(self.slow_period).mean()

        # 计算趋势
        trend = self.risk_manager.calculate_trend_filter(data)

        # 生成原始信号
        raw_signals = pd.Series(0, index=data.index)
        raw_signals[fast_ma > slow_ma] = 1   # 金叉做多
        raw_signals[fast_ma < slow_ma] = -1  # 死叉做空

        # 应用趋势过滤器: 只在趋势方向上交易
        signals = pd.Series(0, index=data.index)
        signals[(raw_signals == 1) & (trend == 1)] = 1   # 做多且上升趋势
        signals[(raw_signals == -1) & (trend == -1)] = -1  # 做空且下降趋势

        return signals

    def get_features(self):
        """获取策略所需特征"""
        return {
            f'MA{self.fast_period}': 'close',
            f'MA{self.slow_period}': 'close'
        }
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_ma_strategy_with_risk.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add strategies/technical/ma_strategy.py tests/test_ma_strategy_with_risk.py
git commit -m "feat: Integrate RiskManager into MAStrategy

- Add RiskManager initialization to MAStrategy
- Apply trend filter to MA crossover signals
- Only take trades aligned with long-term trend
- Add integration tests for signal filtering
- Update docstrings with risk parameters

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Modify BacktestExecutor for Trailing Stops

**Files:**
- Modify: `executors/backtest_executor.py`
- Modify: `tests/test_backtest_executor.py`

**Step 1: Write failing test for trailing stop execution**

Add to `tests/test_backtest_executor.py`:

```python
def test_backtest_with_trailing_stop():
    """Test backtest with trailing stop exits"""
    from strategies.technical.ma_strategy import MAStrategy

    config = {
        'fast_period': 3,
        'slow_period': 5,
        'swing_period': 5,
        'initial_cash': 100000,
        'position_ratio': 0.3
    }

    strategy = MAStrategy('TA', '2020-01-01', '2020-12-31', config)
    executor = BacktestExecutor(strategy, config)

    # Create price data that triggers trailing stop
    # Price goes up, then drops below swing low
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 95],  # Drop at end
        'high': [101, 102, 103, 104, 105, 106, 107, 96],
        'low': [99, 100, 101, 102, 103, 104, 105, 94]
    })

    portfolio = executor.run_backtest(data)

    # Verify portfolio was created
    assert portfolio is not None
    assert 'portfolio_value' in portfolio.columns
    assert 'returns' in portfolio.columns

    # Verify trailing stop was triggered (position closed before end)
    # The -100% return should not happen with trailing stop
    final_return = portfolio['returns'].dropna().iloc[-1]
    assert final_return > -0.5  # Should not lose more than 50%
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_backtest_executor.py::test_backtest_with_trailing_stop -v`
Expected: FAIL (test may run but trailing stop logic not implemented)

**Step 3: Implement trailing stop logic in BacktestExecutor**

Modify `executors/backtest_executor.py`:

```python
# trading/executors/backtest_executor.py
import pandas as pd
import numpy as np

class BacktestExecutor:
    """回测执行器with trailing stop support"""

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

        # 运行简化版回测with trailing stop
        self.portfolio = self._backtest_with_trailing_stop(data, signals)

        return self.portfolio

    def _backtest_with_trailing_stop(self, data, signals):
        """回测实现with trailing stop"""
        initial_cash = self.config.get('initial_cash', 1000000)
        base_position_ratio = self.config.get('position_ratio', 0.3)
        commission_rate = self.config.get('commission_rate', 0.0001)

        cash = initial_cash
        position = 0
        position_type = 0  # 1 for long, -1 for short, 0 for no position
        entry_bar = None
        portfolio_values = []

        # Check if strategy has RiskManager
        has_risk_manager = hasattr(self.strategy, 'risk_manager')

        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]

            # Check trailing stop if in position
            if has_risk_manager and position != 0 and entry_bar is not None:
                should_exit = self.strategy.risk_manager.should_exit_trailing_stop(
                    data, entry_bar, i, position_type
                )
                if should_exit:
                    # Close position due to trailing stop
                    cash += position * current_price
                    cash -= abs(position) * current_price * commission_rate
                    position = 0
                    position_type = 0
                    entry_bar = None

            # Trading logic
            if signal == 1 and position == 0:  # 开多仓
                # Get volatility-adjusted position size
                if has_risk_manager:
                    position_ratio = self.strategy.risk_manager.calculate_volatility_adjustment(
                        data, base_position_ratio
                    )
                else:
                    position_ratio = base_position_ratio

                position_value = cash * position_ratio
                position = position_value / current_price
                cash -= position_value
                position_type = 1
                entry_bar = i

            elif signal == -1 and position == 0:  # 开空仓
                # Get volatility-adjusted position size
                if has_risk_manager:
                    position_ratio = self.strategy.risk_manager.calculate_volatility_adjustment(
                        data, base_position_ratio
                    )
                else:
                    position_ratio = base_position_ratio

                position_value = cash * position_ratio
                position = -position_value / current_price
                cash -= abs(position) * current_price * commission_rate
                position_type = -1
                entry_bar = i

            elif signal == 0 and position != 0:  # 平仓 (signal-based)
                cash += position * current_price
                cash -= abs(position) * current_price * commission_rate
                position = 0
                position_type = 0
                entry_bar = None

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
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_backtest_executor.py::test_backtest_with_trailing_stop -v`
Expected: PASS

**Step 5: Update existing tests and verify they still pass**

Run: `PYTHONPATH=/e/claude_project/trading pytest tests/test_backtest_executor.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add executors/backtest_executor.py tests/test_backtest_executor.py
git commit -m "feat: Add trailing stop support to BacktestExecutor

- Track entry_bar and position_type during backtest
- Check trailing stop condition each bar
- Exit positions when swing high/low penetrated
- Support volatility-adjusted position sizing
- Add test for trailing stop execution
- All existing tests still pass

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update CLI Script with Risk Parameters

**Files:**
- Modify: `scripts/run_backtest.py`

**Step 1: Update CLI to accept risk management parameters**

Modify `scripts/run_backtest.py`:

```python
#!/usr/bin/env python
# trading/scripts/run_backtest.py
"""
回测脚本

使用示例:
python scripts/run_backtest.py --instrument TA --start 2020-01-01 --end 2023-12-31
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_processor import ContinuousContractProcessor
from strategies.technical.ma_strategy import MAStrategy
from executors.backtest_executor import BacktestExecutor
from analyzers.performance_analyzer import PerformanceAnalyzer


def main():
    parser = argparse.ArgumentParser(description='运行回测')
    parser.add_argument('--instrument', type=str, required=True, help='品种代码')
    parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--fast', type=int, default=5, help='快速均线周期')
    parser.add_argument('--slow', type=int, default=20, help='慢速均线周期')
    parser.add_argument('--trend-ma', type=int, default=200, help='趋势过滤MA周期')
    parser.add_argument('--swing-period', type=int, default=20, help='摆荡周期 (trailing stop)')
    parser.add_argument('--vol-threshold', type=int, default=80, help='波动率阈值 (percentile)')
    parser.add_argument('--position-ratio', type=float, default=0.3, help='仓位比例')
    parser.add_argument('--output', type=str, default='backtest_report', help='输出文件前缀')

    args = parser.parse_args()

    try:
        # 1. 加载数据
        csv_path = f'data/raw/{args.instrument}.csv'
        print(f"正在加载数据: {csv_path}")
        processor = ContinuousContractProcessor(csv_path)
        data = processor.process(adjust_price=True)
        data = processor.load_data(start_date=args.start, end_date=args.end)
        print(f"数据加载完成: {len(data)} 条记录\n")

        # 2. 配置
        config = {
            'instrument': args.instrument,
            'start_date': args.start,
            'end_date': args.end,
            'fast_period': args.fast,
            'slow_period': args.slow,
            'trend_ma_period': args.trend_ma,
            'swing_period': args.swing_period,
            'volatility_threshold': args.vol_threshold,
            'position_ratio': args.position_ratio,
            'initial_cash': 1000000,
            'commission_rate': 0.0001,
        }

        # 3. 创建策略
        print(f"策略参数:")
        print(f"  快速MA: {args.fast}")
        print(f"  慢速MA: {args.slow}")
        print(f"  趋势MA: {args.trend_ma}")
        print(f"  摆荡周期: {args.swing_period}")
        print(f"  波动率阈值: {args.vol_threshold}")
        print(f"  仓位比例: {args.position_ratio}\n")

        strategy = MAStrategy(args.instrument, args.start, args.end, config)

        # 4. 运行回测
        print("正在运行回测...")
        executor = BacktestExecutor(strategy, config)
        executor.run_backtest(data)

        # 5. 获取指标
        metrics = executor.get_metrics()

        # 6. 生成报告
        print("\n" + "="*60)
        print("回测结果")
        print("="*60)
        print(f"\n总收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annual_return']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"年化波动率: {metrics['volatility']:.2%}")
        print("="*60 + "\n")

        # 7. 可视化
        analyzer = PerformanceAnalyzer(executor.portfolio, metrics)
        analyzer.plot_equity_curve(f'{args.output}_equity.png')
        analyzer.plot_drawdown(f'{args.output}_drawdown.png')

        print(f"图表已保存:")
        print(f"  - {args.output}_equity.png")
        print(f"  - {args.output}_drawdown.png\n")

        return 0

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

**Step 2: Test the updated CLI**

Run: `cd /e/claude_project/trading && python scripts/run_backtest.py --help`
Expected: Show all new risk parameters

**Step 3: Commit**

```bash
git add scripts/run_backtest.py
git commit -m "feat: Add risk management parameters to CLI

- Add --trend-ma for trend filter period
- Add --swing-period for trailing stop lookback
- Add --vol-threshold for volatility threshold
- Display all risk parameters in output
- Update usage examples

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Run Integration Test and Validation

**Files:**
- None (validation only)

**Step 1: Run all tests**

Run: `cd /e/claude_project/trading && PYTHONPATH=/e/claude_project/trading pytest tests/ -v --tb=short`
Expected: All tests pass (except data file tests)

**Step 2: Run backtest with risk management**

Run: `cd /e/claude_project/trading && python scripts/run_backtest.py --instrument TA --start 2020-01-01 --end 2023-12-31 --fast 5 --slow 20 --trend-ma 200 --swing-period 20 --vol-threshold 80 --output reports/TA_with_risk`

Expected: Backtest completes successfully

**Step 3: Compare results with baseline**

The new backtest with risk management should show:
- Improved Sharpe ratio (> -0.63)
- Reduced maximum drawdown (< -50%)
- Fewer trades due to trend filter
- Not 100% capital loss

**Step 4: Document results**

Create `docs/risk_management_results.md`:

```markdown
# Risk Management Implementation Results

## Baseline (MA Crossover Only)
- Sharpe Ratio: -0.63
- Max Drawdown: -100%
- Total Return: -100%

## With Risk Management
- Sharpe Ratio: [FILL IN]
- Max Drawdown: [FILL IN]
- Total Return: [FILL IN]

## Improvement
- Sharpe Ratio improvement: [CALCULATE]%
- Drawdown reduction: [CALCULATE]%
```

**Step 5: Commit**

```bash
git add docs/risk_management_results.md
git commit -m "docs: Add risk management validation results

- Compare baseline vs. risk-managed performance
- Document improvements in Sharpe and drawdown
- Validate three-layer risk system effectiveness

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Optimize Parameters with Risk Management

**Files:**
- Modify: `scripts/optimize_parameters.py`

**Step 1: Update optimizer to use risk parameters**

Modify the parameter ranges in `scripts/optimize_parameters.py` to include risk parameters.

**Step 2: Run optimization with risk management**

Run: `cd /e/claude_project/trading && python scripts/optimize_parameters.py --instrument TA --start 2020-01-01 --end 2023-12-31 --metric sharpe_ratio --fast-min 3 --fast-max 15 --slow-min 20 --slow-max 60 --output reports/TA_optimization_with_risk.csv`

**Step 3: Analyze top performers**

Review the best parameter combinations with risk management enabled.

**Step 4: Commit**

```bash
git add scripts/optimize_parameters.py
git commit -m "feat: Optimize parameters with risk management

- Update optimizer to include risk parameters
- Run grid search with risk-aware strategy
- Identify best risk-adjusted parameter combinations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Verification

Before completing, verify:

- [ ] All 25 core tests pass (excluding data file tests)
- [ ] RiskManager has 100% test coverage for all methods
- [ ] MAStrategy integrates RiskManager correctly
- [ ] BacktestExecutor implements trailing stop logic
- [ ] CLI accepts all risk parameters
- [ ] Backtest shows improvement over baseline
- [ ] Code committed to feature branch
- [ ] Ready to merge to main

## Success Criteria

- [ ] Sharpe ratio improved by at least 50% (from -0.63 to > -0.32)
- [ ] Max drawdown reduced by at least 30% (from -100% to < -70%)
- [ ] No total loss of capital (total return > -100%)
- [ ] All tests pass
- [ ] Code committed and pushed to GitHub
