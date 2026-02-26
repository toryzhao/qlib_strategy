# Mean Reversion Strategy - Design Document

**Date:** 2025-02-26
**Status:** Approved
**Author:** Claude

## Overview

Implement a statistical mean reversion strategy using Z-Score to identify price deviations from mean, with dynamic position sizing based on deviation magnitude. Designed for range-bound markets where prices tend to revert to historical averages.

## Problem Statement

The dual Moving Average crossover strategy performed poorly on TA futures (2020-2023):
- Total return: -100%
- Max drawdown: -100%
- Issue: Trend-following strategy unsuitable for range-bound markets

**Solution:** Mean reversion strategy excels in oscillating markets by buying low and selling high when prices deviate from statistical norms.

## Strategy Logic

### Z-Score Calculation

```
Z-Score = (Current Price - Rolling Mean) / Rolling Standard Deviation
```

- Z-Score > 0: Price above mean (potential short)
- Z-Score < 0: Price below mean (potential long)
- |Z-Score| larger = greater deviation = stronger signal

### Multi-Layer Position Sizing

| Z-Score Range | Position Size | Signal |
|---------------|---------------|--------|
| ≥ +2.5        | 100%          | Short  |
| +2.0 ~ +2.5   | 75%           | Short  |
| +1.5 ~ +2.0   | 50%           | Short  |
| -1.5 ~ +1.5   | 0%            | No Trade |
| -2.0 ~ -1.5   | 50%           | Long   |
| -2.5 ~ -2.0   | 75%           | Long   |
| ≤ -2.5        | 100%          | Long   |

**Benefits:**
- Small deviations → Small positions (test the waters)
- Large deviations → Large positions (higher confidence)
- Avoid all-in entries at extreme levels

## Exit Mechanism

### Primary Exit: Z-Score Mean Reversion

Close position when price reverts to mean (Z-Score enters ±0.5 range).

### Stop Loss Protection

Prevent unlimited losses if price continues to diverge:

| Entry Z-Score | Stop Z-Score | Stop Distance |
|---------------|--------------|---------------|
| ±1.5          | ±2.5         | 67%           |
| ±2.0          | ±3.0         | 50%           |
| ±2.5          | ±3.5         | 40%           |

### Forced Exit Conditions

- Maximum holding period exceeded (e.g., 50 bars)
- Gap opening crosses stop line

**Execution Priority:**
1. Check stop loss → Immediate exit
2. Check Z-Score reversion → Take profit
3. Check max holding period → Forced exit

## Configuration

```python
config = {
    # Z-Score calculation parameters
    'lookback_period': 20,        # Rolling window (optimize: 10-30)
    'entry_threshold': 1.5,       # Entry threshold (optimize: 1.2-2.0)

    # Multi-layer position thresholds
    'level1_threshold': 1.5,      # 50% position
    'level2_threshold': 2.0,      # 75% position
    'level3_threshold': 2.5,      # 100% position

    # Exit parameters
    'exit_threshold': 0.5,        # Reversion exit threshold
    'max_hold_period': 50,        # Maximum holding period

    # Stop loss parameters
    'stop_multiplier': 1.5,       # Stop multiplier (entry_threshold × multiplier)

    # Risk management
    'use_volatility_adjustment': True,
    'atr_period': 14,
    'atr_lookback': 100,
    'volatility_threshold': 80,

    # Base configuration
    'position_ratio': 0.3,
    'initial_cash': 1000000,
    'commission_rate': 0.0001
}
```

### Key Parameters for Optimization

- `lookback_period`: Affects mean/std calculation sensitivity
- `entry_threshold`: Balances trade frequency vs deviation magnitude
- `level1/2/3_threshold`: Controls position sizing aggressiveness

## Architecture

### New Components

```
strategies/statistical/mean_reversion.py
├── MeanReversionStrategy class
│   ├── __init__(config)
│   ├── calculate_zscore(data)              # Calculate Z-Score series
│   ├── get_position_size(zscore)           # Multi-layer position allocation
│   ├── should_exit(current_zscore, entry_zscore, bars_held)
│   └── generate_signals(data)              # Generate trading signals
```

### Key Methods

**1. calculate_zscore(data)**
```python
rolling_mean = data['close'].rolling(lookback_period).mean()
rolling_std = data['close'].rolling(lookback_period).std()
zscore = (data['close'] - rolling_mean) / rolling_std
return zscore
```

**2. get_position_size(zscore)**
Returns 0%/50%/75%/100% position based on Z-Score value
Supports volatility adjustment for actual position size

**3. generate_signals(data)**
- Calculate Z-Score for each bar
- Generate entry signals (1/-1) and target position ratio
- Return DataFrame: signals + target_position_ratio

### Integration with Existing System

- Inherits from `FuturesStrategy` base class
- Reuses `RiskManager` volatility adjustment
- Uses `BacktestExecutor` for backtesting
- Registers with `StrategyFactory` as 'mean_reversion'

## Data Flow

```
1. Raw Data Input
   └─> Contains: open, high, low, close, volume

2. Z-Score Calculation
   ├─> Calculate rolling mean (lookback_period)
   ├─> Calculate rolling standard deviation
   └─> Generate Z-Score series

3. Signal Generation
   ├─> Check Z-Score threshold → Trigger entry signal
   ├─> Calculate target position (multi-layer)
   └─> Apply volatility adjustment → Actual position

4. Position Management
   ├─> Check stop condition (entry_zscore × stop_multiplier)
   ├─> Check reversion condition (Z-Score in ±exit_threshold)
   └─> Check max holding period (max_hold_period)

5. Output
   └─> signals DataFrame + position ratio
```

## Error Handling

### 1. Insufficient Data
```python
if len(data) < lookback_period:
    return pd.Series(0, index=data.index)  # No signals
```

### 2. Zero Standard Deviation
```python
rolling_std = data['close'].rolling(lookback_period).std()
rolling_std = rolling_std.replace(0, np.nan)  # Avoid division by zero
```

### 3. Extreme Values
```python
zscore = zscore.clip(-5, 5)  # Limit Z-Score to ±5 range
```

## Risk Management

### Partial Integration

Mean reversion strategy uses **only volatility adjustment** from the existing risk management system:

**Why not trend filter?**
- Mean reversion designed for range-bound markets
- Trend filter would block most signals
- Strategy already has built-in exit logic

**Volatility adjustment benefits:**
- Reduces position size during high volatility
- Protects against gap openings
- Improves risk-adjusted returns

## Testing Strategy

### Unit Tests (`tests/test_mean_reversion.py`)

- Test Z-Score calculation accuracy (verify with known data)
- Test position allocation logic (threshold boundaries)
- Test exit condition triggers
- Test edge cases (insufficient data, zero std, extreme values)

### Integration Tests (`tests/test_mean_reversion_integration.py`)

- Test full signal generation with all features
- Verify position size adjustments
- Test entry/exit logic
- Compare trade count with baseline

### Backtest Validation

**Test Period:** 2020-2023 TA futures (same as MA strategy)

**Expected Results vs MA Strategy:**
- Better performance in range-bound periods
- Higher trade frequency (more mean reversion opportunities)
- Lower maximum drawdown (positions reverse with price)
- Improved Sharpe ratio (profits from oscillations)

**Success Criteria:**
- Total return > -50% (vs MA's -100%)
- Sharpe ratio > -1.0 (vs MA's -2.7)
- Max drawdown < -70% (vs MA's -100%)
- No total loss of capital

## Implementation Plan

1. Create `strategies/statistical/__init__.py`
2. Implement `MeanReversionStrategy` class with Z-Score logic
3. Implement multi-layer position sizing
4. Implement exit mechanism (reversion + stop loss)
5. Write unit tests for Z-Score calculation
6. Write unit tests for position sizing
7. Write integration tests
8. Update `StrategyFactory` to register mean reversion
9. Update CLI script to support mean reversion parameters
10. Run backtest and validate results
11. Optimize parameters
12. Generate performance report

## Success Criteria

- [ ] All mean reversion features implemented
- [ ] Unit tests pass (100% coverage of core logic)
- [ ] Integration tests pass
- [ ] Backtest shows improvement over MA strategy:
  - Sharpe ratio improved by at least 50%
  - Max drawdown reduced by at least 30%
  - No total loss of capital
- [ ] Code committed and pushed to GitHub

## Comparison with MA Strategy

| Feature | MA Crossover | Mean Reversion |
|---------|--------------|----------------|
| Market Type | Trending | Range-bound |
| Signal Type | Trend-following | Counter-trend |
| Entry Logic | MA crossover | Z-Score deviation |
| Position Sizing | Fixed | Dynamic (multi-layer) |
| Exit Logic | Crossover/SMA | Reversion/Stops |
| Risk Management | Full (3 layers) | Partial (volatility) |
| Expected Performance | Poor in TA | Better in range markets |

## Future Enhancements

1. **Multi-indicator confirmation** - Combine Z-Score with RSI, Bollinger Bands
2. **Adaptive lookback period** - Adjust based on market volatility regime
3. **Market regime detection** - Switch between trend and mean reversion strategies
4. **Portfolio optimization** - Combine mean reversion with other strategies
5. **Machine learning enhancement** - Use ML to predict reversion probability
