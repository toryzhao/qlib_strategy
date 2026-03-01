# Mean Reversion Strategy - Backtest Results

**Date:** 2025-03-01
**Instrument:** TA futures
**Test Period:** 2020-01-01 to 2020-12-31

## Configuration

```python
{
    'lookback_period': 20,
    'entry_threshold': 1.5,
    'exit_threshold': 0.5,
    'max_hold_period': 50,
    'stop_multiplier': 1.5,
    'position_ratio': 0.3
}
```

## Results

### Mean Reversion Strategy
- **Total Return:** -100.00%
- **Annual Return:** -212.75%
- **Sharpe Ratio:** -2.7373
- **Max Drawdown:** -100.00%
- **Volatility:** 78.82%

### MA Strategy (for comparison)
- **Total Return:** -100.00%
- **Annual Return:** -196.02%
- **Sharpe Ratio:** -2.6948
- **Max Drawdown:** -100.00%
- **Volatility:** 73.85%

## Analysis

Both strategies showed similar poor performance on 2020 TA futures data:

1. **Total capital loss** in both strategies
2. **Mean Reversion Sharpe ratio** (-2.74) is slightly worse than MA (-2.69)
3. **Higher volatility** in mean reversion (78.82% vs 73.85%)

### Potential Issues

1. **Default parameters** may not be optimal for TA futures
2. **Exit conditions** may need refinement
3. **Multi-layer position sizing** might be too aggressive without proper limits
4. **No volatility adjustment** was enabled in this test run

## Next Steps

1. **Parameter optimization** (Task 10) - Grid search for optimal parameters
2. **Enable volatility adjustment** to reduce risk
3. **Test on different time periods** (2021, 2022, 2023)
4. **Analyze individual trades** to understand failure modes
5. **Consider alternative exit strategies** or position sizing limits

## Conclusion

The implementation is complete and functional, but the default parameters are not suitable for TA futures in 2020. Further optimization is required (Task 10).
