# Risk Management for MA Strategy - Design Document

**Date:** 2025-02-19
**Status:** Approved
**Author:** Claude

## Overview

Enhance the dual Moving Average crossover strategy with three risk management layers to improve risk-adjusted returns and reduce large drawdowns.

## Problem Statement

The current MA crossover strategy (2020-2023 on TA futures) performed poorly:
- All parameter combinations: -100% total return
- Max drawdown: -100%
- No risk management mechanisms

## Solution

Add three risk management layers:

### 1. Trend Filter (200-period MA)
**Purpose:** Only trade in the direction of the long-term trend

**Logic:**
- Calculate 200-period moving average
- Long signals: Only when fast MA > slow MA AND price > 200-MA
- Short signals: Only when fast MA < slow MA AND price < 200-MA
- Conflicting signals: Skip trade (signal = 0)

**Expected Impact:**
- Reduce whipsaw trades in choppy markets
- Improve win rate by trading with trend
- Fewer trades but higher quality

### 2. Volatility Filter (ATR Percentile)
**Purpose:** Reduce position size during high volatility periods

**Logic:**
- Calculate 14-period ATR
- Calculate ATR percentile over last 100 bars (0-100)
- If ATR percentile > 80 (top 20% volatility): Reduce position size by 50%
- Otherwise: Use configured base position ratio (default 30%)

**Expected Impact:**
- Less risk during volatile periods
- Smaller losses on gap openings
- Better risk-adjusted returns

### 3. Trailing Stop (20-bar Swing)
**Purpose:** Lock in profits by exiting on swing reversals

**Logic:**
- For long positions: Exit if close falls below lowest low of last 20 bars since entry
- For short positions: Exit if close rises above highest high of last 20 bars since entry
- Update swing high/low each bar while in position
- Handle edge case: If position held < 20 bars, use entry price as stop

**Expected Impact:**
- Protect profits from reversals
- Reduce maximum drawdown
- Let winners run while cutting losses

## Architecture

### New Components

```
strategies/risk/risk_manager.py
├── RiskManager class
│   ├── calculate_trend_filter(data) -> int
│   ├── calculate_volatility_adjustment(data, base_ratio) -> float
│   └── should_exit_trailing_stop(data, entry_bar, position_type) -> bool
```

### Modified Components

```
strategies/technical/ma_strategy.py
├── Add RiskManager integration
├── generate_signals(): Apply trend filter
└── get_position_size(): Apply volatility adjustment

executors/backtest_executor.py
├── Track entry_bar and position_type
├── Check trailing stop each bar
└── Exit when stop hit
```

## Configuration

```python
config = {
    # MA Strategy parameters
    'fast_period': 5,
    'slow_period': 20,

    # Risk Management parameters
    'trend_ma_period': 200,        # Period for trend filter MA
    'atr_period': 14,              # ATR calculation period
    'atr_lookback': 100,           # Lookback for ATR percentile
    'volatility_threshold': 80,    # Percentile threshold for high vol
    'swing_period': 20,            # Lookback for swing high/low

    # Position sizing
    'position_ratio': 0.3,         # Base position ratio
    'initial_cash': 1000000,
    'commission_rate': 0.0001,
}
```

## Data Flow

```
1. Pre-calculate Indicators:
   ├── Fast MA (5)
   ├── Slow MA (20)
   ├── Trend MA (200)
   ├── ATR (14)
   └── ATR Percentile (rolling 100-bar rank)

2. Generate Entry Signals:
   ├── Calculate MA crossover signal
   ├── Calculate trend (price vs 200-MA)
   └── Filter: Only take aligned signals

3. Determine Position Size:
   ├── Calculate base position size
   ├── Check ATR percentile
   └── Reduce by 50% if high volatility

4. Monitor Exit Conditions:
   ├── Each bar: Check trailing stop
   ├── Calculate swing high/low since entry
   └── Exit if price penetrates swing level
```

## Error Handling

### Insufficient Data
- **Issue:** First 200 bars needed for 200-MA
- **Solution:** Start signals from bar 201, return 0 for earlier bars

### Incomplete Swing Data
- **Issue:** Position held < swing_period (20 bars)
- **Solution:** Use available bars (min 5), else use entry price

### ATR Percentile Initialization
- **Issue:** Need 100 bars for rolling percentile
- **Solution:** Use expanding window until 100 bars available

### Gap Openings
- **Issue:** Market opens beyond stop price
- **Solution:** Exit at opening price (realistic fill)

## Testing Strategy

### Unit Tests (`tests/test_risk_manager.py`)
- Test trend filter calculation (uptrend/downtrend/sideways)
- Test volatility adjustment (normal/high/low ATR)
- Test trailing stop logic (long/short positions)
- Test edge cases (insufficient data, gaps)

### Integration Tests (`tests/test_ma_strategy_with_risk.py`)
- Test full signal generation with all filters
- Verify position size adjustments
- Test entry/exit logic
- Compare trade count with baseline

### Backtest Validation
- Run on TA futures (2020-2023)
- Compare metrics:
  - Sharpe ratio (target: > -0.63, ideally > 0.5)
  - Max drawdown (target: < -50%)
  - Total return (target: > -50%, ideally positive)
  - Trade count (expected: 30-50% reduction)

## Implementation Plan

1. Create `strategies/risk/` directory and `__init__.py`
2. Implement `RiskManager` class with all three risk layers
3. Add ATR calculation utility
4. Modify `MAStrategy` to integrate RiskManager
5. Modify `BacktestExecutor` to track entry bars and check stops
6. Write unit tests for RiskManager
7. Write integration tests for enhanced MA strategy
8. Update CLI script with new config options
9. Run backtest and validate improvements
10. Optimize parameters with new risk features

## Success Criteria

- [ ] All risk management features implemented
- [ ] Unit tests pass (100% coverage of RiskManager)
- [ ] Integration tests pass
- [ ] Backtest shows improvement over baseline:
  - Sharpe ratio improved by at least 50%
  - Max drawdown reduced by at least 30%
  - No total loss of capital
- [ ] Code committed and pushed to GitHub
