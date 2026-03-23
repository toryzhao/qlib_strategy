# Enhanced MA(20) Strategy with Statistical Regime Filtering

**Date:** 2025-03-23
**Status:** Design Complete
**Objective:** Optimize MA(20) to achieve 40%+ annual return through market regime filtering

---

## Executive Summary

**Problem:** MA(20) strategy achieves 41.37% in 2009-2011 bull market but only 6.73% over full 2009-2025 period.

**Solution:** Implement statistical regime detection using Hidden Markov Model (HMM) to identify favorable market conditions (bull trends) and only trade MA(20) when probabilities align.

**Expected Outcome:** Improve full-period performance from 6.73% to 15-25% annual return by avoiding whipsaws and drawdowns in unfavorable regimes.

---

## Architecture

### Core Components

1. **RegimeDetector** (HMM-based)
   - 3-state Hidden Markov Model: Bear Trend, Ranging, Bull Trend
   - Input features: Log returns, realized volatility, MA slope, price acceleration
   - Output: Probabilistic regime classification with confidence scores

2. **EnhancedMAStrategy**
   - Base MA(20) entry/exit logic
   - Regime filter: Only trade when regime=Bull AND confidence>60%
   - Multi-layered exits: Price < MA(20) OR regime≠Bull OR confidence drops

3. **PositionSizer**
   - Dynamic allocation based on regime confidence (0.50 to 1.00)
   - Trend strength multiplier using ADX
   - Position caps: Min 30%, Max 150%
   - Formula: `position_size = base_position × confidence × trend_multiplier`

4. **BacktestExecutor**
   - Walk-forward validation framework
   - Comprehensive performance metrics
   - Benchmark comparisons

### Data Flow

```
Raw Price Data
    ↓
Feature Engineering (returns, volatility, MA slope)
    ↓
HMM Regime Detection → Regime + Confidence
    ↓
Regime Filter (Bull + Confidence > 60%?)
    ↓ Yes
MA(20) Signal Generation
    ↓
Position Sizing (confidence × ADX multiplier)
    ↓
Execute Trade
```

---

## Regime Detection Implementation

### Hidden Markov Model Setup

**Library:** `hmmlearn` (Gaussian HMM)

**States:**
- State 0: Bear Trend (negative returns, high volatility)
- State 1: Ranging/Choppy (low volatility, mean-reverting)
- State 2: Bull Trend (positive returns, directional movement)

**Input Features:**
1. **Log Returns:** 1-day log returns (captures trend direction)
2. **Realized Volatility:** 20-day rolling standard deviation of returns
3. **MA Slope:** Derivative of 20-day moving average (trend strength)
4. **Price Acceleration:** 2nd derivative of price (momentum changes)

**Training:**
- Train on full 2009-2025 dataset
- Unsupervised learning (no labels required)
- Gaussian HMM with full covariance matrices
- Random seed for reproducibility

**Regime Filter Logic:**
```python
if regime == 'BULL' and confidence > 0.60:
    position_size = base_position * confidence * adx_multiplier
    execute_ma_strategy(position_size)
else:
    position = 0  # Sit out
```

**Confidence Calculation:**
- Confidence = max(P(Bull|data), P(Ranging|data), P(Bear|data))
- Range: 0.33 to 1.00
- Only trade when confident (typically >0.60)

---

## Enhanced MA Strategy & Position Sizing

### Entry/Exit Rules

**Entry Conditions (all must be true):**
1. Close price > MA(20)
2. Regime == Bull
3. Regime confidence > 0.60

**Exit Conditions (any trigger):**
1. Close price < MA(20) (primary exit)
2. Regime != Bull (regime change)
3. Confidence < 0.50 (uncertainty)
4. Stop loss: -10% from entry (optional risk management)

**Direction:** Long-only (no shorting)

### Dynamic Position Sizing

**Formula:**
```
position_size = base_position × regime_confidence × trend_strength_multiplier
```

**Trend Strength Multiplier (based on ADX):**
- ADX > 30 (strong trend): 1.2x
- ADX 20-30 (moderate): 1.0x
- ADX < 20 (weak): 0.7x

**Position Caps:**
- Minimum: 30% (even in weak bull conditions)
- Maximum: 150% (allow pyramiding in optimal conditions)
- Normal range: 60-120%

**Examples:**
- Confidence 0.75, ADX 28: 100% × 0.75 × 1.0 = **75%**
- Confidence 0.90, ADX 35: 100% × 0.90 × 1.2 = **108%**
- Regime ≠ Bull: **0%** (sit out)

---

## Backtesting & Validation

### Testing Framework

**Phase 1: Training (2009-2018)**
- Train HMM on first 10 years
- Walk-forward optimization
- Parameters to optimize:
  - HMM states (2, 3, 4)
  - Confidence threshold (0.50, 0.60, 0.70)
  - Base position (80%, 100%, 120%)
  - ADX thresholds (20, 25, 30)

**Phase 2: Validation (2019-2025)**
- Out-of-sample testing
- True test of regime filtering effectiveness
- Compare against vanilla MA(20) baseline (6.73%)

**Phase 3: Full Period (2009-2025)**
- Comprehensive performance analysis
- Regime attribution
- Benchmark comparisons

### Performance Metrics

**Primary Metrics:**
- Annual Return (target: 20%+, stretch goal: 40%+)
- Sharpe Ratio (target: >1.5)
- Maximum Drawdown (target: <25%)

**Secondary Metrics:**
- Win Rate
- Profit Factor
- Average Trade Duration
- Time in Market (% days with position)
- Regime Distribution

**Success Criteria:**
- ✅ Annual return >20% (3x improvement from 6.73%)
- ✅ Sharpe ratio >1.5
- ✅ Max drawdown <25%
- 🎯 Stretch goal: 40%+ annual return

---

## Implementation Files

### New Files

**`strategies/regime/regime_detector.py`**
```python
class RegimeDetector:
    """Hidden Markov Model for market regime detection"""
    - __init__(n_states=3, covariance_type='full')
    - fit(data)  # Train HMM
    - predict(data)  # Return regime probabilities
    - get_current_regime()  # Return (regime, confidence)
    - plot_regimes()  # Visualize regime history
```

**`strategies/technical/enhanced_ma_strategy.py`**
```python
class EnhancedMAStrategy:
    """MA(20) with statistical regime filtering"""
    - generate_signals(data)  # MA(20) + regime filter
    - calculate_position_size(confidence, adx)
    - should_enter_market(regime, confidence)
    - should_exit_market(price, ma, regime, confidence)
```

**`scripts/test_enhanced_ma.py`**
- Load data, train HMM, run backtest
- Compare enhanced vs vanilla MA(20)
- Generate performance report and plots

### Dependencies
- `hmmlearn` - HMM implementation
- `scipy` - Statistical functions
- `matplotlib` - Visualization
- Existing: pandas, numpy, base strategy classes

---

## Edge Cases & Risk Mitigation

### Known Limitations

**1. Regime Detection Lag**
- HMM may detect regime change after it occurs
- Mitigation: Use leading indicators (MA slope, acceleration)

**2. Overfitting Risk**
- Too many parameters optimized on historical data
- Mitigation: Walk-forward validation, small parameter space

**3. HMM Instability**
- Non-deterministic results across runs
- Mitigation: Set random seed, use multiple runs

**4. Insufficient Bull Regimes**
- If 2009-2025 has limited bull markets, strategy sits out too much
- Mitigation: Lower confidence threshold or use 4-state HMM

### Edge Case Handling

**Flat/Choppy Markets:**
```python
if volatility < historical_median * 0.5:
    return 0  # Sit out
```

**Rapid Regime Switching:**
```python
if regime_changed and days_in_current_regime < 5:
    return current_position  # Hold to avoid whipsaw
```

**Extreme Volatility Spikes:**
```python
if abs(daily_return) > 0.05:  # 5%+ move
    position_size *= 0.7  # Reduce for safety
```

**Low Confidence States:**
```python
if max(regime_probabilities) < 0.55:  # HMM unsure
    return min(current_position * 0.5, base_position * 0.5)
```

---

## Performance Expectations

### Best Case (Bull-Dominant Period)
- Annual Return: 35-50%
- Time in Market: 60-80%
- Regime adds: 10-15% over vanilla MA(20)

### Expected Case (Mixed Period like 2009-2025)
- Annual Return: 15-25% (vs 6.73% vanilla)
- Time in Market: 40-60%
- Significant improvement from avoiding bad periods

### Worst Case (Bear/Ranging Dominant)
- Annual Return: 0-10%
- Time in Market: 20-40%
- Preserves capital (small drawdowns)

---

## Exit Strategy

If after implementation the strategy fails to achieve >15% annual return:

1. **Investigate:** Visualize regime detection quality
2. **Alternative 1:** Try simple ADX threshold (no HMM)
3. **Alternative 2:** Volatility-filtered MA(20)
4. **Accept:** 40% may only be achievable in select bull markets

---

## Next Steps

1. ✅ Design complete
2. ⏭️ Create implementation plan
3. ⏭️ Implement RegimeDetector
4. ⏭️ Implement EnhancedMAStrategy
5. ⏭️ Run comprehensive backtests
6. ⏭️ Validate performance
7. ⏭️ Document results

---

**Design Status:** Complete and approved
**Ready for Implementation:** Yes
