# Regimetry Adaptive Range Strategy (RARS) - Design Document

**Date:** 2025-03-24
**Status:** Design Phase
**Objective:** Use Transformer-based regime detection to adapt trading logic based on market state

---

## Executive Summary

The Regimetry Adaptive Range Strategy (RARS) uses unsupervised machine learning (Regimetry) to detect market regimes and adapts its trading logic accordingly:

- **Bull Markets:** Buy on pullbacks to recent lows (mean-reversion entries)
- **Bear Markets:** Sell short on rallies to recent highs (mean-reversion entries)
- **Ranging Markets:** Trade breakouts from recent ranges (trend-following entries)

This hybrid approach combines the best of mean-reversion and trend-following by matching the strategy to the market environment.

---

## Core Philosophy

Traditional strategies fail because they assume market behavior is constant. RARS recognizes that markets operate in distinct **regimes** that require different approaches:

| Regime | Market Characteristic | Optimal Strategy | RARS Implementation |
|--------|----------------------|------------------|---------------------|
| **Bull** | Uptrend with pullbacks | Buy dips | Enter long near recent lows |
| **Bear** | Downtrend with rallies | Short rallies | Enter short near recent highs |
| **Ranging** | Sideways, bound by ranges | Fade breakouts or trade ranges | Trade confirmed breakouts |

The key insight is **matching the strategy to the regime** rather than using a one-size-fits-all approach.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Regimetry Engine                         │
│  (runs weekly to update regime labels)                      │
│                                                             │
│  Price Data → Transformer Embeddings → Spectral Clustering  │
│                                                             │
│  Output: cluster_assignments.csv (Date, Cluster_ID)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Regime Mapping Layer                      │
│  (maps cluster IDs to BULL/BEAR/RANGING)                    │
│                                                             │
│  For each cluster:                                          │
│    - Calculate recent 20-day return                         │
│    - Compare to dynamic threshold (±1.5 × ATR ratio)        │
│    - Assign market state                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Signal Generation Layer                   │
│  (generates entry signals based on market state)            │
│                                                             │
│  Dynamic Window → Recent Extremes → Entry Zones → Signals  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Risk Management Layer                     │
│  (position sizing and stop loss)                            │
│                                                             │
│  Risk-based sizing: 2% account value per 2 ATR move         │
└─────────────────────────────────────────────────────────────┘
```

---

## Regime Detection

### Regimetry Integration

**Pipeline (runs weekly):**

```bash
# 1. Generate embeddings
python launch_host.py embed \
  --signal-input-path data/TA_features.csv \
  --window-size 30 \
  --stride 1 \
  --encoding-method sinusoidal \
  --output-name TA_embeddings.npy

# 2. Cluster regimes
python launch_host.py cluster \
  --embedding-path embeddings/TA_embeddings.npy \
  --regime-data-path data/processed/regime_input.csv \
  --n-clusters 8 \
  --window-size 30 \
  --output-dir reports/TA
```

**Output Files:**
- `embeddings/TA_embeddings.npy` - Transformer embeddings
- `reports/TA/cluster_assignments.csv` - Daily regime labels

### Dynamic Regime Mapping

Each cluster ID is dynamically mapped to a trading state based on recent performance:

```python
def map_cluster_to_state(cluster_id, lookback_days=20):
    """
    Map a cluster to BULL/BEAR/RANGING based on recent returns
    """
    # Get recent data for this cluster
    cluster_data = get_data_for_cluster(cluster_id)

    # Calculate 20-day return
    returns_20d = (cluster_data['close'] / cluster_data['close'].shift(20)) - 1
    recent_return = returns_20d.iloc[-1]

    # Calculate ATR ratio
    atr_20 = calculate_atr(cluster_data, period=20).iloc[-1]
    price = cluster_data['close'].iloc[-1]
    atr_ratio = (atr_20 * 1.5) / price

    # Dynamic thresholds
    threshold_bull = atr_ratio
    threshold_bear = -atr_ratio

    # Map to state
    if recent_return > threshold_bull:
        return 'BULL'
    elif recent_return < threshold_bear:
        return 'BEAR'
    else:
        return 'RANGING'
```

**Key Features:**
- **Adaptive thresholds**: Higher volatility requires larger returns to qualify as bull/bear
- **Dynamic mapping**: Clusters can change state as market behavior evolves
- **Lookback window**: 20 days balances responsiveness and stability

---

## Signal Generation

### Dynamic Window Calculation

The window for calculating recent extremes adapts to market volatility:

```python
def calculate_dynamic_window(atr_series, current_date):
    """
    Calculate window size based on current vs baseline volatility
    """
    # Calculate ATR baseline (60-day median)
    atr_baseline = atr_series.rolling(60).median().iloc[-1]
    current_atr = atr_series.iloc[-1]

    # Adaptive window
    if current_atr > atr_baseline:
        return 10  # High volatility: shorter window
    else:
        return 30  # Low volatility: longer window
```

**Rationale:**
- High volatility: Short window captures significant moves quickly
- Low volatility: Long window filters out minor fluctuations

### Entry Signal Logic

#### Bull Market - Buy on Pullbacks

**Trigger Conditions:**
1. Market state = BULL
2. No existing long position
3. Price ≤ (recent_low + 1 × ATR)

**Implementation:**
```python
def generate_bull_signal(data, position_state):
    window = calculate_dynamic_window(data['atr'])

    recent_low = data['low'].rolling(window).min().iloc[-1]
    current_atr = data['atr'].iloc[-1]
    entry_zone = recent_low + current_atr

    if data['close'].iloc[-1] <= entry_zone and position_state['long'] == 0:
        return 'LONG'
    return None
```

**Example:**
- Recent low (30-day): 2800
- Current ATR: 50
- Entry zone: 2800 + 50 = 2850
- Current price: 2830 → **ENTER LONG**

#### Bear Market - Short on Rallies

**Trigger Conditions:**
1. Market state = BEAR
2. No existing short position
3. Price ≥ (recent_high - 1 × ATR)

**Implementation:**
```python
def generate_bear_signal(data, position_state):
    window = calculate_dynamic_window(data['atr'])

    recent_high = data['high'].rolling(window).max().iloc[-1]
    current_atr = data['atr'].iloc[-1]
    entry_zone = recent_high - current_atr

    if data['close'].iloc[-1] >= entry_zone and position_state['short'] == 0:
        return 'SHORT'
    return None
```

**Example:**
- Recent high (30-day): 3200
- Current ATR: 50
- Entry zone: 3200 - 50 = 3150
- Current price: 3160 → **ENTER SHORT**

#### Ranging Market - Trade Confirmed Breakouts

**Trigger Conditions:**

**Long Entry:**
1. Market state = RANGING
2. No existing long position
3. Price > (recent_high + 1 × ATR)
4. **Confirmation**: Next bar's close remains above breakout level

**Short Entry:**
1. Market state = RANGING
2. No existing short position
3. Price < (recent_low - 1 × ATR)
4. **Confirmation**: Next bar's close remains below breakdown level

**Implementation:**
```python
def generate_ranging_signal(data, position_state, pending_signal=None):
    window = calculate_dynamic_window(data['atr'])

    recent_high = data['high'].rolling(window).max().iloc[-1]
    recent_low = data['low'].rolling(window).min().iloc[-1]
    current_atr = data['atr'].iloc[-1]

    # Check for pending signal confirmation
    if pending_signal == 'LONG_PENDING':
        if data['close'].iloc[-1] > (recent_high + current_atr):
            return 'LONG', None
        else:
            return None, None

    if pending_signal == 'SHORT_PENDING':
        if data['close'].iloc[-1] < (recent_low - current_atr):
            return 'SHORT', None
        else:
            return None, None

    # Check for new breakout signals
    if position_state['long'] == 0:
        breakout_level = recent_high + current_atr
        if data['close'].iloc[-1] > breakout_level:
            return None, 'LONG_PENDING'  # Wait for confirmation

    if position_state['short'] == 0:
        breakdown_level = recent_low - current_atr
        if data['close'].iloc[-1] < breakdown_level:
            return None, 'SHORT_PENDING'  # Wait for confirmation

    return None, None
```

**Why Confirmation?**
Ranging markets often have false breakouts. Waiting for the next bar to close outside the range filters out fakeouts.

---

## Position Sizing

### Risk-Based Position Calculation

**Core Principle:** Each trade risks exactly 2% of account value

**Formula:**
```
Risk Amount = Account Value × 2%
Stop Loss Distance = 2 × ATR
Position Size = Risk Amount / Stop Loss Distance

Number of Contracts = (Position Size × Account Value) / Contract Price
```

**Implementation:**
```python
def calculate_position_size(account_value, entry_price, atr):
    """
    Calculate position size based on 2% risk rule
    """
    risk_amount = account_value * 0.02
    stop_loss_distance = 2 * atr

    position_value = risk_amount / stop_loss_distance
    contracts = int(position_value / entry_price)

    return contracts, position_value
```

**Example:**
- Account value: 1,000,000
- Entry price: 3000
- ATR: 50

```
Risk Amount = 1,000,000 × 0.02 = 20,000
Stop Loss Distance = 2 × 50 = 100
Position Value = 20,000 / 100 = 200,000
Contracts = 200,000 / 3,000 = 66 contracts
```

**Key Features:**
- **Adaptive Risk**: Higher volatility (larger ATR) → smaller positions
- **Consistent Risk**: Every trade has the same risk exposure
- **No Overtrading**: Position size naturally limited in volatile markets

---

## Exit Management

### Dual Exit System

#### Primary Exit: Regime Change

When the market state changes, close all positions:

```python
def check_regime_change(current_regime, previous_regime):
    if current_regime != previous_regime:
        close_all_positions()
        log(f"Regime changed from {previous_regime} to {current_regime}")
```

**Rationale:**
- Bull market entries are only valid in bull markets
- When regime changes, the edge for the trade disappears
- Prevents holding positions in unfavorable environments

#### Protective Exit: Stop Loss

Hard stop loss at 2 ATR from entry:

```python
def check_stop_loss(position, current_price, entry_price, entry_atr):
    if position['side'] == 'LONG':
        stop_loss = entry_price - 2 * entry_atr
        if current_price < stop_loss:
            close_position(position, reason='STOP_LOSS')

    elif position['side'] == 'SHORT':
        stop_loss = entry_price + 2 * entry_atr
        if current_price > stop_loss:
            close_position(position, reason='STOP_LOSS')
```

**Rationale:**
- Unlimited risk is unacceptable
- 2 ATR allows for normal market noise
- Stops are hit before significant damage occurs

### Exit Priority

1. **Stop Loss** - Checked daily, executes immediately if triggered
2. **Regime Change** - Checked weekly (when regimetry updates)

**Special Case: Gradual Stopdown**

If price moves adversely but hasn't hit stop loss:
- After 3 consecutive days with adverse move > 1.5 ATR
- Reduce position by 50%
- This provides early risk reduction for losing trades

---

## Position Management Rules

### Pure Position Approach

Only one position per direction at any time:

```python
position_state = {
    'long': 0,      # Number of long contracts
    'short': 0,     # Number of short contracts
    'entry_price': None,
    'entry_date': None
}
```

**Rules:**
1. New long signal → Close existing long, open new long
2. New short signal → Close existing short, open new short
3. Never hold both long and short simultaneously
4. If signal contradicts existing position → Close position first

**Example:**
```
Currently: Long 50 contracts @ 3000
New Signal: Short

Action:
1. Close long 50 @ market
2. Open short (calculated size) @ market
```

---

## Parameters Summary

### Regime Detection

| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_size` | 30 | Regimetry embedding window |
| `n_clusters` | 8 | Number of regimes to detect |
| `mapping_lookback` | 20 days | For calculating cluster returns |
| `threshold_multiplier` | 1.5 | For dynamic regime mapping |
| `update_frequency` | Weekly | Regimetry re-run frequency |

### Signal Generation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `volatility_window_short` | 10 days | Window in high volatility |
| `volatility_window_long` | 30 days | Window in low volatility |
| `atr_baseline_period` | 60 days | For volatility comparison |
| `entry_buffer` | 1 ATR | Distance from extreme for entry |
| `breakout_confirmation` | Next bar | For ranging market entries |

### Risk Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| `risk_per_trade` | 2% | Account value at risk |
| `stop_loss_atr` | 2 ATR | Stop loss distance |
| `gradual_stopdown_days` | 3 days | For 50% position reduction |
| `gradual_stopdown_threshold` | 1.5 ATR | Adverse move threshold |

---

## Expected Behavior Examples

### Example 1: Bull Market Pullback

**Market Conditions:**
- Regime: BULL (cluster 5, 20-day return +2.1%)
- Recent 30-day low: 2800
- Current ATR: 45
- Current price: 2830

**Entry Logic:**
```
Entry zone = 2800 + (1 × 45) = 2845
Current price (2830) ≤ Entry zone (2845) → ✓
No existing long position → ✓
```

**Action:** Enter long
- Account: 1,000,000
- Risk: 20,000 (2%)
- Stop: 2 × 45 = 90 points
- Position value: 20,000 / 90 = 222,222
- Contracts: 222,222 / 2830 = 78 contracts

**Exit:**
- Regime changes to RANGING or BEAR → Close position
- Price drops 2 ATR below entry (2740) → Stop loss

### Example 2: Bear Market Rally

**Market Conditions:**
- Regime: BEAR (cluster 2, 20-day return -1.8%)
- Recent 30-day high: 3100
- Current ATR: 55
- Current price: 3070

**Entry Logic:**
```
Entry zone = 3100 - (1 × 55) = 3045
Current price (3070) ≥ Entry zone (3045) → ✓
No existing short position → ✓
```

**Action:** Enter short
- Position calculated similarly to long example
- Profit if price declines

### Example 3: Ranging Market Breakout

**Market Conditions:**
- Regime: RANGING (cluster 7, 20-day return +0.3%)
- Recent 10-day high: 2950 (high volatility)
- Current ATR: 60
- Current price: 3020 (just broke out)

**Day 1 - Initial Breakout:**
```
Breakout level = 2950 + (1 × 60) = 3010
Price (3020) > Breakout level (3010) → ✓
Generate pending signal: LONG_PENDING
```

**Day 2 - Confirmation:**
```
Price closes at 3035 > Breakout level (3010) → ✓
Confirmed: Enter long
```

**Why Confirmation Works:**
- Without confirmation: Would have entered on fake breakout at 3020
- If price closed back at 2990: No trade taken, avoided loss

---

## Edge Cases and Special Handling

### Regime Detection Lag

**Problem:** Regimetry has natural lag (first/last window_size bars have no labels)

**Solution:**
- Use previous day's regime for current day
- If no regime available (start of dataset), default to RANGING

### New Regime Clusters

**Problem:** Regimetry may discover new clusters not seen before

**Solution:**
- Assign initial state as RANGING
- After 20 days of data, map based on performance
- Log new cluster discovery for analysis

### Gap Risk (Overnight/Weekend)

**Problem:** Price may gap past stop loss between sessions

**Solution:**
- Use limit orders at stop loss price
- If gap occurs, exit at market open on next bar
- Accept gap risk as part of trading (cannot eliminate entirely)

### Low Volatility Periods

**Problem:** Very low ATR leads to large position sizes

**Solution:**
- Set minimum position size (e.g., 10% of account)
- Set maximum position size (e.g., 100% of account)
- Clip calculated position to [min, max] range

### Multiple Signals Same Day

**Problem:** Both long and short signals trigger on same day

**Solution:**
- Priority: Regime change > Stop loss > New entry
- If stop loss and new entry same day: Close first, then evaluate entry
- Cannot hold both directions simultaneously

---

## Performance Metrics

### Key Metrics to Track

1. **Return Metrics**
   - Annual return (by regime)
   - Average win/loss per regime
   - Win rate by regime

2. **Risk Metrics**
   - Maximum drawdown
   - Average trade drawdown
   - Stop loss hit rate

3. **Regime Metrics**
   - Time in each regime (%)
   - Regime transition frequency
   - Average trade duration by regime

4. **Execution Metrics**
   - Slippage vs expected entry zones
   - Fill rate on limit orders
   - Confirmation rate for breakout signals

### Benchmark Comparison

Compare against:
- Buy and hold
- Fixed MA(20) strategy
- Static breakout strategy

The goal is **adaptive outperformance**: better risk-adjusted returns across all market conditions.

---

## Implementation Plan

See separate document: `2025-03-24-rars-implementation-plan.md`

---

## Future Enhancements

### Phase 2 Improvements

1. **Multi-Timeframe Regimes**
   - Daily regimes for entry timing
   - Weekly regimes for position sizing
   - Monthly regimes for strategy selection

2. **Regime-Specific Parameters**
   - Optimize entry buffer per regime (not always 1 ATR)
   - Optimize stop loss per regime
   - Regime-specific profit targets

3. **Ensemble Regime Detection**
   - Combine Regimetry with traditional indicators
   - Weighted voting system
   - Confidence-weighted position sizing

4. **Machine Learning Optimization**
   - RL for optimal entry timing
   - Predict regime transitions
   - Adaptive threshold optimization

---

## Risks and Limitations

### Known Risks

1. **Regime Detection Error**
   - Regimetry may misclassify market state
   - Mitigation: Diversification across multiple parameter sets

2. **Lag in Regime Changes**
   - Transformer windows introduce delay
   - Mitigation: Weekly updates, not real-time

3. **Overfitting Risk**
   - Many parameters to optimize
   - Mitigation: Out-of-sample testing, walk-forward validation

4. **Black Swan Events**
   - Sudden regime shifts not captured in training
   - Mitigation: Hard stop losses, position limits

### Limitations

1. **Not Suitable For:**
   - Illiquid markets (slippage too high)
   - Highly correlated portfolios (regimes affect all similarly)
   - Short timeframes (noise dominates signal)

2. **Assumptions:**
   - Regimes are relatively stable (weekly updates sufficient)
   - ATR is a good volatility measure
   - 2% risk per trade is appropriate

---

## Conclusion

The Regimetry Adaptive Range Strategy represents a new paradigm in systematic trading: **match the strategy to the market, not the market to the strategy.**

By detecting latent market regimes and adapting the trading logic accordingly, RARS aims to:
- Capture profits in all market conditions
- Reduce drawdowns through regime-aware exits
- Avoid the worst pitfalls of one-size-fits-all strategies

The key innovation is not a better indicator or smarter entry, but **intelligence about when to use which approach.**

This design document provides the complete blueprint for implementation. The next step is detailed planning and execution.
