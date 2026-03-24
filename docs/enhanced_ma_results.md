# Enhanced MA(20) with Regime Filter - Results Report

**Date:** 2025-03-24
**Strategy:** MA(20) + HMM Regime Detection
**Objective:** Improve full-period performance from 6.73% to 15-25% annual return

---

## Executive Summary

The Enhanced MA(20) strategy with Hidden Markov Model regime detection was successfully implemented and tested on TA futures data from 2009-2025. However, **the primary objective was not achieved** - the enhanced strategy underperformed the baseline MA(20) strategy.

### Key Findings

- **Regime filter too restrictive**: The HMM-based regime filter prevented trades during profitable periods
- **Performance degradation**: Enhanced strategy returned 3.08% annually vs 8.02% baseline
- **Fewer trades**: 57% trade reduction (218 → 93 trades) but at the cost of missing profitable opportunities
- **Drawdown improvement**: Slightly better max drawdown (-37.33% vs -41.29%)

### Performance Summary

| Period | Baseline MA(20) | Enhanced MA(20) | Improvement |
|--------|-----------------|-----------------|-------------|
| 2009-2011 Bull | 42.17% | 25.00% | -17.17% |
| 2019-2025 OOS | 2.00% | -2.25% | -4.25% |
| 2009-2025 Full | 8.02% | 3.08% | -4.94% |

---

## Regime Analysis

The HMM model classifies markets into 3 regimes:
- **State 0 (Bear)**: Negative returns, high volatility
- **State 1 (Ranging)**: Low volatility, mean-reverting
- **State 2 (Bull)**: Positive returns, directional movement

The regime visualization shows distinct regime transitions over the 2006-2025 period. However, the regime filter may be:
1. Misclassifying bull market periods
2. Using confidence threshold too high (0.60)
3. Suffering from look-ahead bias (trained on full dataset)

---

## Conclusion

**Status: ❌ TARGET NOT ACHIEVED**

The Enhanced MA(20) strategy failed to improve upon the baseline MA(20) performance. The regime detection approach, while theoretically sound, proved too restrictive in practice.

### Root Causes

1. **Regime classification issues**: HMM may not capture true bull/bear regimes effectively
2. **Over-filtering**: Confidence threshold of 0.60 excludes too many trades
3. **Look-ahead bias**: Training on 2006-2025 data contaminates backtest
4. **Feature limitations**: Current features (returns, volatility, MA slope, acceleration) may not predict regimes well

### Recommendations

1. **Lower confidence threshold**: Test with 0.40-0.50
2. **Train on pre-test data**: Use only data before 2009 for training
3. **Add features**: Include volume, ATR, multi-timeframe analysis
4. **Consider 2-state model**: Bull vs Bear (no ranging state)
5. **Alternative regime detection**: Try trend-following indicators (ADX, MACD)

---

## Primary Metric

**Full Period Annual Return:** 3.08%
**Target:** 15-25%
**Gap:** -11.92% to -21.92%
**Status:** ❌ NOT ACHIEVED

---

## Implementation Details

**Files Created:**
- `strategies/regime/regime_detector.py` - HMM-based regime detection
- `strategies/technical/enhanced_ma_strategy.py` - Enhanced MA strategy
- `scripts/test_enhanced_ma.py` - Integration test script
- `tests/regime/test_regime_detector.py` - Unit tests
- `tests/technical/test_enhanced_ma_strategy.py` - Unit tests

**Commits:**
- db3d430: Add regime detection module structure
- 66ca3bb: Implement RegimeDetector with HMM
- f4cffc3: Implement EnhancedMAStrategy
- 41a8d00: Add integration test

**Test Results:**
- All unit tests pass (13 tests)
- Integration test runs successfully
- Regime visualization generated
