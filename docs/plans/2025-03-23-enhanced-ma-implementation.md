# Enhanced MA(20) with Regime Filter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an enhanced MA(20) trading strategy that uses Hidden Markov Models to detect market regimes and only trade in favorable bull markets, aiming to improve full-period (2009-2025) performance from 6.73% to 15-25% annual return.

**Architecture:**
- RegimeDetector: 3-state Hidden Markov Model (Bear/Ranging/Bull) trained on price features
- EnhancedMAStrategy: MA(20) signals filtered by regime confidence
- Dynamic position sizing: Scale exposure based on regime probability and ADX trend strength
- Backtest framework: Walk-forward validation with comprehensive performance metrics

**Tech Stack:**
- Python 3.8+, pandas, numpy
- hmmlearn (Hidden Markov Models)
- scipy (statistical functions)
- matplotlib (visualization)
- Existing codebase: strategies.base, utils.data_processor, executors.backtest_executor

---

## Task 1: Install Dependencies and Create Directory Structure

**Files:**
- Create: `strategies/regime/__init__.py`
- Create: `strategies/regime/README.md`
- Modify: `requirements.txt`

**Step 1: Add hmmlearn to requirements.txt**

```bash
echo "hmmlearn==0.3.0" >> requirements.txt
```

**Step 2: Install the dependency**

```bash
pip install hmmlearn==0.3.0
```

Expected: Successfully installs hmmlearn

**Step 3: Create regime directory structure**

```bash
mkdir -p strategies/regime
touch strategies/regime/__init__.py
```

**Step 4: Write README.md**

```markdown
# Market Regime Detection

Statistical models for detecting market regimes (bull/bear/ranging).

## Components

- `regime_detector.py` - Hidden Markov Model based regime detection
```

**Step 5: Commit**

```bash
git add requirements.txt strategies/regime/
git commit -m "feat: Add regime detection module structure

- Add hmmlearn dependency
- Create strategies/regime directory
- Prepare for HMM implementation"
```

---

## Task 2: Implement RegimeDetector Core Class

**Files:**
- Create: `strategies/regime/regime_detector.py`
- Create: `tests/regime/test_regime_detector.py`

**Step 1: Write failing tests for RegimeDetector initialization**

Create `tests/regime/test_regime_detector.py`:

```python
import pytest
import numpy as np
import pandas as pd
from strategies.regime.regime_detector import RegimeDetector

def test_regime_detector_initialization():
    """Test RegimeDetector can be initialized with default parameters"""
    detector = RegimeDetector()
    assert detector.n_states == 3
    assert detector.covariance_type == 'full'
    assert detector.model is None

def test_regime_detector_custom_params():
    """Test RegimeDetector with custom parameters"""
    detector = RegimeDetector(n_states=4, covariance_type='diag')
    assert detector.n_states == 4
    assert detector.covariance_type == 'diag'

def test_feature_calculation():
    """Test feature engineering for HMM"""
    detector = RegimeDetector()

    # Create sample price data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.random.randn(100).cumsum()
    df = pd.DataFrame({'close': prices}, index=dates)

    features = detector._calculate_features(df)

    # Check all features are present
    assert 'log_returns' in features.columns
    assert 'volatility' in features.columns
    assert 'ma_slope' in features.columns
    assert 'acceleration' in features.columns

    # Check no NaN values (after warmup)
    assert features['log_returns'].notna().sum() > 90

def test_fit_raises_error_without_data():
    """Test that fit requires sufficient data"""
    detector = RegimeDetector()

    # Too little data
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({'close': np.arange(10)}, index=dates)

    with pytest.raises(ValueError, match="Insufficient data"):
        detector.fit(df)

def test_fit_success():
    """Test successful model fitting"""
    detector = RegimeDetector(n_states=3, random_state=42)

    # Create sample data
    dates = pd.date_range('2009-01-01', periods=1000, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(1000).cumsum()
    df = pd.DataFrame({'close': prices}, index=dates)

    detector.fit(df)

    assert detector.model is not None
    assert detector.model.monitor_.converged

def test_predict_returns_regimes():
    """Test prediction returns regime labels and probabilities"""
    detector = RegimeDetector(n_states=3, random_state=42)

    # Train on sample data
    dates = pd.date_range('2009-01-01', periods=1000, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(1000).cumsum()
    df = pd.DataFrame({'close': prices}, index=dates)

    detector.fit(df)

    # Predict on same data
    regimes, probs = detector.predict(df)

    assert len(regimes) == len(df)
    assert regimes.dtype == int
    assert len(probs) == len(df)
    assert probs.shape[1] == 3  # 3 states
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1

def test_get_current_regime():
    """Test getting current regime with confidence"""
    detector = RegimeDetector(n_states=3, random_state=42)

    dates = pd.date_range('2009-01-01', periods=1000, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(1000).cumsum()
    df = pd.DataFrame({'close': prices}, index=dates)

    detector.fit(df)

    regime, confidence = detector.get_current_regime(df)

    assert regime in [0, 1, 2]
    assert 0.0 <= confidence <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/regime/test_regime_detector.py -v
```

Expected: All tests FAIL with "ModuleNotFoundError: No module named 'strategies.regime.regime_detector'"

**Step 3: Implement RegimeDetector class**

Create `strategies/regime/regime_detector.py`:

```python
"""
Market Regime Detection using Hidden Markov Models

Detects market regimes (bear/ranging/bull) using unsupervised learning.
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    Hidden Markov Model for market regime detection

    Uses price features to classify market into 3 regimes:
    - State 0: Bear Trend (negative returns, high volatility)
    - State 1: Ranging/Choppy (low volatility, mean-reverting)
    - State 2: Bull Trend (positive returns, directional movement)

    Attributes:
        n_states: Number of regimes to detect
        covariance_type: HMM covariance type ('full', 'diag', 'spherical')
        random_state: Random seed for reproducibility
        model: Trained HMM model
        scaler: Feature scaler
    """

    def __init__(self, n_states=3, covariance_type='full', random_state=42):
        """
        Initialize RegimeDetector

        Args:
            n_states: Number of regimes (default: 3)
            covariance_type: Covariance matrix type (default: 'full')
            random_state: Random seed (default: 42)
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def _calculate_features(self, df):
        """
        Calculate features for HMM training

        Features:
        - Log returns: Trend direction and magnitude
        - Realized volatility: 20-day rolling std of returns
        - MA slope: Derivative of 20-day MA
        - Price acceleration: 2nd derivative

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with features
        """
        close = df['close']

        # Log returns
        log_returns = np.log(close / close.shift(1))

        # Realized volatility (20-day)
        volatility = log_returns.rolling(window=20).std()

        # Moving average and slope
        ma = close.rolling(window=20).mean()
        ma_slope = ma.diff()

        # Price acceleration
        acceleration = close.diff().diff()

        features = pd.DataFrame({
            'log_returns': log_returns,
            'volatility': volatility,
            'ma_slope': ma_slope,
            'acceleration': acceleration
        }).dropna()

        return features

    def fit(self, df):
        """
        Train HMM on historical data

        Args:
            df: DataFrame with 'close' column and datetime index

        Raises:
            ValueError: If insufficient data (<500 rows after feature engineering)
        """
        # Calculate features
        features = self._calculate_features(df)

        if len(features) < 500:
            raise ValueError(f"Insufficient data: {len(features)} rows, need at least 500")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Initialize and train HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_iter=100,
            tol=1e-4
        )

        self.model.fit(features_scaled)

        return self

    def predict(self, df):
        """
        Predict regime probabilities for each time point

        Args:
            df: DataFrame with 'close' column

        Returns:
            regimes: Array of regime labels (0, 1, 2)
            probs: Array of regime probabilities (n_samples, n_states)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Calculate features
        features = self._calculate_features(df)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get posterior probabilities
        _, posteriors = self.model.decode(features_scaled)

        # Get most likely regime for each time point
        regimes = posteriors.argmax(axis=1)

        return regimes, posteriors

    def get_current_regime(self, df):
        """
        Get the most recent regime and confidence

        Args:
            df: DataFrame with 'close' column

        Returns:
            regime: Most recent regime label (0, 1, 2)
            confidence: Probability of the predicted regime (0-1)
        """
        regimes, probs = self.predict(df)

        # Get most recent prediction
        regime = regimes[-1]
        confidence = probs[-1, regime]

        return regime, confidence

    def get_regime_name(self, regime):
        """
        Convert regime number to human-readable name

        Args:
            regime: Regime label (0, 1, 2)

        Returns:
            str: Regime name
        """
        regime_names = {
            0: "BEAR",
            1: "RANGING",
            2: "BULL"
        }
        return regime_names.get(regime, "UNKNOWN")

    def plot_regimes(self, df, save_path=None):
        """
        Visualize regime history

        Args:
            df: DataFrame with 'close' column
            save_path: Optional path to save plot

        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        regimes, _ = self.predict(df)

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        # Plot price with regime colors
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Price', color='black', alpha=0.6)

        # Color code regimes
        colors = ['red', 'gray', 'green']
        for regime in range(self.n_states):
            mask = regimes == regime
            if mask.any():
                ax1.scatter(df.index[mask], df['close'].iloc[mask],
                           c=colors[regime], label=f'Regime {regime}',
                           alpha=0.5, s=10)

        ax1.set_title('Price with Regime Classification')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Plot regime probabilities
        ax2 = axes[1]
        _, probs = self.predict(df)

        for regime in range(self.n_states):
            ax2.plot(df.index, probs[:, regime],
                    label=f'Regime {regime}', alpha=0.7)

        ax2.set_title('Regime Probabilities')
        ax2.set_ylabel('Probability')
        ax2.set_xlabel('Date')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/regime/test_regime_detector.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add strategies/regime/regime_detector.py tests/regime/test_regime_detector.py
git commit -m "feat: Implement RegimeDetector with HMM

- Implement 3-state Hidden Markov Model for regime detection
- Feature engineering: returns, volatility, MA slope, acceleration
- Unit tests for initialization, fitting, prediction
- Visualization support for regime history"
```

---

## Task 3: Implement EnhancedMAStrategy

**Files:**
- Create: `strategies/technical/enhanced_ma_strategy.py`
- Create: `tests/technical/test_enhanced_ma_strategy.py`

**Step 1: Write failing tests for EnhancedMAStrategy**

Create `tests/technical/test_enhanced_ma_strategy.py`:

```python
import pytest
import pandas as pd
import numpy as np
from strategies.technical.enhanced_ma_strategy import EnhancedMAStrategy
from strategies.regime.regime_detector import RegimeDetector

def test_enhanced_ma_initialization():
    """Test strategy initialization with default config"""
    config = {
        'ma_period': 20,
        'base_position': 1.00,
        'confidence_threshold': 0.60,
        'min_position': 0.30,
        'max_position': 1.50,
    }

    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    assert strategy.ma_period == 20
    assert strategy.base_position == 1.00
    assert strategy.confidence_threshold == 0.60

def test_generate_signals_no_filter():
    """Test signal generation without regime filter (baseline MA)"""
    config = {'ma_period': 20, 'use_regime_filter': False}
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # Create sample data with clear trend
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.arange(100) * 0.5  # Upward trend
    df = pd.DataFrame({
        'close': prices,
        'open': prices - 0.5,
        'high': prices + 0.5,
        'low': prices - 1.0
    }, index=dates)

    signals = strategy.generate_signals(df)

    # Should have long signals in uptrend
    assert (signals == 1).sum() > 50
    assert (signals == 0).sum() > 0

def test_calculate_position_size():
    """Test dynamic position sizing based on confidence and ADX"""
    config = {
        'ma_period': 20,
        'base_position': 1.00,
        'min_position': 0.30,
        'max_position': 1.50,
    }
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # High confidence, strong trend
    size1 = strategy.calculate_position_size(confidence=0.90, adx=35)
    assert 1.0 <= size1 <= 1.5

    # Medium confidence, medium trend
    size2 = strategy.calculate_position_size(confidence=0.70, adx=25)
    assert 0.5 <= size2 <= 1.0

    # Low confidence, weak trend
    size3 = strategy.calculate_position_size(confidence=0.65, adx=15)
    assert 0.3 <= size3 <= 0.7

    # All positions should be within bounds
    assert strategy.min_position <= size1 <= strategy.max_position
    assert strategy.min_position <= size2 <= strategy.max_position
    assert strategy.min_position <= size3 <= strategy.max_position

def test_should_enter_market():
    """Test entry conditions with regime filter"""
    config = {
        'ma_period': 20,
        'use_regime_filter': True,
        'confidence_threshold': 0.60,
    }
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # Bull regime, high confidence - should enter
    assert strategy.should_enter_market(regime=2, confidence=0.80) == True

    # Bull regime, low confidence - should not enter
    assert strategy.should_enter_market(regime=2, confidence=0.50) == False

    # Ranging regime - should not enter
    assert strategy.should_enter_market(regime=1, confidence=0.80) == False

    # Bear regime - should not enter
    assert strategy.should_enter_market(regime=0, confidence=0.80) == False

def test_should_exit_market():
    """Test exit conditions"""
    config = {
        'ma_period': 20,
        'use_regime_filter': True,
        'confidence_threshold': 0.60,
        'stop_loss': 0.10,
    }
    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)
    strategy.entry_price = 100.0

    # Price below MA - should exit
    assert strategy.should_exit_market(price=99, ma=100, regime=2, confidence=0.80) == True

    # Regime changed to bear - should exit
    assert strategy.should_exit_market(price=105, ma=100, regime=0, confidence=0.80) == True

    # Confidence dropped - should exit
    assert strategy.should_exit_market(price=105, ma=100, regime=2, confidence=0.40) == True

    # Stop loss hit - should exit
    assert strategy.should_exit_market(price=89, ma=100, regime=2, confidence=0.80) == True

    # All good - should not exit
    assert strategy.should_exit_market(price=105, ma=100, regime=2, confidence=0.80) == False

def test_enhanced_signals_with_regime_filter():
    """Test full signal generation with regime filter"""
    config = {
        'ma_period': 20,
        'use_regime_filter': True,
        'confidence_threshold': 0.60,
    }

    strategy = EnhancedMAStrategy('TA', '2020-01-01', '2021-01-01', config)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(200).cumsum()
    df = pd.DataFrame({
        'close': prices,
        'open': prices - 0.5,
        'high': prices + 0.5,
        'low': prices - 1.0
    }, index=dates)

    # Train regime detector
    regime_detector = RegimeDetector(n_states=3, random_state=42)
    regime_detector.fit(df)
    strategy.regime_detector = regime_detector

    # Generate signals
    signals = strategy.generate_signals(df)

    # Signals should be 0, 1 only (long-only)
    assert set(signals.unique()).issubset({0, 1})

    # Should have fewer signals than vanilla MA (due to regime filter)
    vanilla_signals = (df['close'] > df['close'].rolling(20).mean()).astype(int)
    assert signals.sum() <= vanilla_signals.sum()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/technical/test_enhanced_ma_strategy.py -v
```

Expected: All tests FAIL with module not found errors

**Step 3: Implement EnhancedMAStrategy**

Create `strategies/technical/enhanced_ma_strategy.py`:

```python
"""
Enhanced MA(20) Strategy with Market Regime Filtering

Combines classic moving average trend following with statistical regime detection
to only trade in favorable market conditions.
"""

import pandas as pd
import numpy as np
from strategies.base.base_strategy import FuturesStrategy
from strategies.regime.regime_detector import RegimeDetector


class EnhancedMAStrategy(FuturesStrategy):
    """
    Enhanced MA(20) with regime filtering

    Entry: Price > MA(20) AND regime=Bull AND confidence > threshold
    Exit: Price < MA(20) OR regime≠Bull OR confidence drops

    Dynamic position sizing based on regime confidence and ADX trend strength.
    """

    def __init__(self, instrument, start_date, end_date, config):
        super().__init__(instrument, start_date, end_date, config)

        # MA parameters
        self.ma_period = config.get('ma_period', 20)

        # Regime filtering
        self.use_regime_filter = config.get('use_regime_filter', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.60)

        # Position sizing
        self.base_position = config.get('base_position', 1.00)
        self.min_position = config.get('min_position', 0.30)
        self.max_position = config.get('max_position', 1.50)

        # ADX thresholds for trend strength
        self.adx_strong_trend = config.get('adx_strong_trend', 30)
        self.adx_weak_trend = config.get('adx_weak_trend', 20)

        # Risk management
        self.stop_loss = config.get('stop_loss', 0.10)  # 10% stop loss

        # Regime detector (will be set during backtest)
        self.regime_detector = None

        # State tracking
        self.entry_price = None
        self.current_position_size = 0.0

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

    def calculate_adx(self, data, period=14):
        """
        Calculate Average Directional Index

        ADX > 30: Strong trend
        ADX 20-30: Moderate trend
        ADX < 20: Weak trend/ranging
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def calculate_position_size(self, confidence, adx):
        """
        Calculate dynamic position size

        Formula: base_position × confidence × trend_multiplier

        Args:
            confidence: Regime confidence (0.0 to 1.0)
            adx: Average Directional Index

        Returns:
            Position size ratio (e.g., 0.75 = 75%)
        """
        # Trend strength multiplier
        if adx >= self.adx_strong_trend:
            trend_multiplier = 1.2
        elif adx >= self.adx_weak_trend:
            trend_multiplier = 1.0
        else:
            trend_multiplier = 0.7

        # Calculate position
        position = self.base_position * confidence * trend_multiplier

        # Apply bounds
        position = max(self.min_position, min(position, self.max_position))

        return position

    def should_enter_market(self, regime, confidence):
        """
        Check if conditions are favorable for entry

        Args:
            regime: Regime label (0=bear, 1=ranging, 2=bull)
            confidence: Regime probability

        Returns:
            bool: True if should enter
        """
        if not self.use_regime_filter:
            return True

        # Only enter in bull regime with sufficient confidence
        return regime == 2 and confidence >= self.confidence_threshold

    def should_exit_market(self, price, ma, regime, confidence):
        """
        Check if should exit position

        Exit conditions:
        1. Price crosses below MA
        2. Regime changes from bull
        3. Confidence drops below threshold
        4. Stop loss hit

        Args:
            price: Current price
            ma: Moving average value
            regime: Current regime
            confidence: Regime confidence

        Returns:
            bool: True if should exit
        """
        # Primary exit: price below MA
        if price < ma:
            return True

        # Regime changed
        if self.use_regime_filter and regime != 2:
            return True

        # Confidence dropped
        if self.use_regime_filter and confidence < (self.confidence_threshold - 0.10):
            return True

        # Stop loss
        if self.entry_price is not None:
            loss_pct = (self.entry_price - price) / self.entry_price
            if loss_pct >= self.stop_loss:
                return True

        return False

    def generate_signals(self, data):
        """
        Generate trading signals

        Returns:
            pd.Series: 1 (long), 0 (flat), with position size in separate column
        """
        close = data['close']

        # Calculate MA
        ma = close.rolling(window=self.ma_period).mean()

        # Calculate ADX
        adx = self.calculate_adx(data)

        # Initialize signals
        signals = pd.Series(0, index=data.index)
        position_sizes = pd.Series(0.0, index=data.index)

        # Get regime predictions if using filter
        if self.use_regime_filter and self.regime_detector is not None:
            regimes, regime_probs = self.regime_detector.predict(data)
            regime_series = pd.Series(regimes, index=data.index)
            confidence_series = pd.Series([probs[i, r] for i, r in enumerate(regimes)],
                                          index=data.index)
        else:
            # No filtering - always "bull" regime with max confidence
            regime_series = pd.Series(2, index=data.index)  # Always bull
            confidence_series = pd.Series(1.0, index=data.index)

        # Generate signals
        in_position = False

        for i in range(self.ma_period, len(data)):
            current_price = close.iloc[i]
            current_ma = ma.iloc[i]
            current_regime = regime_series.iloc[i]
            current_confidence = confidence_series.iloc[i]
            current_adx = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 25

            if not in_position:
                # Check entry conditions
                if (current_price > current_ma and
                    self.should_enter_market(current_regime, current_confidence)):

                    signals.iloc[i] = 1
                    position_size = self.calculate_position_size(
                        current_confidence, current_adx
                    )
                    position_sizes.iloc[i] = position_size
                    self.entry_price = current_price
                    in_position = True
            else:
                # Check exit conditions
                if self.should_exit_market(current_price, current_ma,
                                          current_regime, current_confidence):
                    signals.iloc[i] = 0  # Exit signal
                    position_sizes.iloc[i] = 0.0
                    self.entry_price = None
                    in_position = False
                else:
                    # Maintain position
                    signals.iloc[i] = 1
                    position_size = self.calculate_position_size(
                        current_confidence, current_adx
                    )
                    position_sizes.iloc[i] = position_size

        # Combine signals and position sizes
        result = pd.DataFrame({
            'signal': signals,
            'position_size': position_sizes
        }, index=data.index)

        return result

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
pytest tests/technical/test_enhanced_ma_strategy.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add strategies/technical/enhanced_ma_strategy.py tests/technical/test_enhanced_ma_strategy.py
git commit -m "feat: Implement EnhancedMAStrategy with regime filtering

- MA(20) signals filtered by HMM regime detection
- Dynamic position sizing based on confidence and ADX
- Multi-layered exit conditions (MA, regime, confidence, stop-loss)
- Comprehensive unit tests for all components"
```

---

## Task 4: Create Integration Test Script

**Files:**
- Create: `scripts/test_enhanced_ma.py`

**Step 1: Write integration test script**

Create `scripts/test_enhanced_ma.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MA(20) Integration Test

Tests the complete enhanced strategy with regime filtering against baseline MA(20).
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_processor import ContinuousContractProcessor
from strategies.technical.enhanced_ma_strategy import EnhancedMAStrategy
from strategies.regime.regime_detector import RegimeDetector
import matplotlib.pyplot as plt


def run_backtest(data, strategy, initial_cash=1000000):
    """
    Simple backtest executor

    Returns:
        dict: Performance metrics
    """
    signals_df = strategy.generate_signals(data)

    cash = initial_cash
    position = 0
    portfolio_values = []
    trades = []

    for i in range(1, len(data)):
        price = data['close'].iloc[i]
        signal = signals_df['signal'].iloc[i]
        position_size = signals_df['position_size'].iloc[i]

        # Execute trades
        if signal == 1 and position == 0:
            # Enter long
            position_value = cash * position_size
            position = position_value / price
            cash -= position_value
            trades.append({'date': data.index[i], 'action': 'BUY',
                          'price': price, 'size': position_size})

        elif signal == 0 and position > 0:
            # Exit long
            cash += position * price
            pnl = (price - trades[-1]['price']) / trades[-1]['price'] if trades else 0
            trades[-1]['exit_price'] = price
            trades[-1]['pnl'] = pnl
            position = 0

        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

    # Calculate metrics
    portfolio_series = pd.Series(portfolio_values, index=data.index[1:])
    returns = portfolio_series.pct_change().dropna()

    years = (data.index[-1] - data.index[0]).days / 365.25

    metrics = {
        'final_value': portfolio_values[-1],
        'total_return': (portfolio_values[-1] / initial_cash - 1),
        'annual_return': returns.mean() * 252,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(returns),
        'num_trades': len([t for t in trades if t['action'] == 'BUY']),
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


def main():
    """Main test function"""
    print("=" * 80)
    print("Enhanced MA(20) Integration Test")
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

    # Test periods
    test_periods = [
        ('2009-01-08', '2011-02-09', '2009-2011 Bull Market'),
        ('2019-01-01', '2025-03-21', '2019-2025 Out-of-Sample'),
        ('2009-01-01', '2025-03-21', '2009-2025 Full Period'),
    ]

    results = []

    for start, end, period_name in test_periods:
        print(f"\n{'=' * 80}")
        print(f"Testing: {period_name}")
        print(f"Period: {start} to {end}")
        print(f"{'=' * 80}")

        data = daily_df.loc[start:end].copy()

        if len(data) < 100:
            print(f"Skipping {period_name} - insufficient data")
            continue

        # Test 1: Vanilla MA(20) (baseline)
        print("\n--- Baseline: Vanilla MA(20) ---")
        config_baseline = {
            'ma_period': 20,
            'use_regime_filter': False,
            'base_position': 1.00,
        }

        strategy_baseline = EnhancedMAStrategy('TA', start, end, config_baseline)
        metrics_baseline = run_backtest(data, strategy_baseline)

        print(f"Annual Return: {metrics_baseline['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics_baseline['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics_baseline['max_drawdown']:.2%}")
        print(f"Total Return: {metrics_baseline['total_return']:.2%}")
        print(f"Number of Trades: {metrics_baseline['num_trades']}")

        # Test 2: Enhanced MA(20) with regime filter
        print("\n--- Enhanced: MA(20) + Regime Filter ---")

        # Train regime detector on data up to test period
        train_data = daily_df.loc[:start].copy()
        if len(train_data) < 500:
            # Use test data for training if insufficient pre-data
            train_data = data.copy()

        print("Training regime detector...")
        regime_detector = RegimeDetector(n_states=3, random_state=42)
        regime_detector.fit(train_data)

        config_enhanced = {
            'ma_period': 20,
            'use_regime_filter': True,
            'confidence_threshold': 0.60,
            'base_position': 1.00,
            'min_position': 0.30,
            'max_position': 1.50,
        }

        strategy_enhanced = EnhancedMAStrategy('TA', start, end, config_enhanced)
        strategy_enhanced.regime_detector = regime_detector

        metrics_enhanced = run_backtest(data, strategy_enhanced)

        print(f"Annual Return: {metrics_enhanced['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics_enhanced['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics_enhanced['max_drawdown']:.2%}")
        print(f"Total Return: {metrics_enhanced['total_return']:.2%}")
        print(f"Number of Trades: {metrics_enhanced['num_trades']}")

        # Comparison
        print("\n--- Comparison ---")
        improvement = (metrics_enhanced['annual_return'] -
                      metrics_baseline['annual_return'])
        print(f"Annual Return Improvement: {improvement:+.2%}")

        if metrics_enhanced['annual_return'] > metrics_baseline['annual_return']:
            print("✓ Enhanced strategy OUTPERFORMS baseline")
        else:
            print("✗ Enhanced strategy underperforms baseline")

        results.append({
            'period': period_name,
            'baseline_return': metrics_baseline['annual_return'],
            'enhanced_return': metrics_enhanced['annual_return'],
            'improvement': improvement
        })

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Period':<30} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<12}")
    print("-" * 80)

    for result in results:
        print(f"{result['period']:<30} "
              f"{result['baseline_return']:>10.2%}   "
              f"{result['enhanced_return']:>10.2%}   "
              f"{result['improvement']:>+10.2%}")

    print("\n" + "=" * 80)
    print(f"PRIMARY METRIC: {results[-1]['enhanced_return']:.6f}")
    print("=" * 80)

    # Check target achievement
    target = 0.20  # 20% target
    full_period_return = results[-1]['enhanced_return']

    if full_period_return >= target:
        print(f"\n✓ SUCCESS: Achieved {full_period_return:.2%} vs {target:.0%} target")
    else:
        gap = target - full_period_return
        print(f"\n✗ Target not met: {full_period_return:.2%} vs {target:.0%}, gap: {gap:.2%}")

    # Plot regime visualization
    print("\nGenerating regime visualization...")
    fig = regime_detector.plot_regimes(daily_df.loc['2009-01-01':'2025-03-21'])
    plt.savefig('docs/enhanced_ma_regimes.png', dpi=150, bbox_inches='tight')
    print("Regime plot saved to: docs/enhanced_ma_regimes.png")

    return results


if __name__ == '__main__':
    results = main()
```

**Step 2: Make script executable**

```bash
chmod +x scripts/test_enhanced_ma.py
```

**Step 3: Run integration test**

```bash
python scripts/test_enhanced_ma.py
```

Expected: Script runs, tests both baseline and enhanced strategies, generates comparison output

**Step 4: Commit**

```bash
git add scripts/test_enhanced_ma.py
git commit -m "feat: Add integration test for Enhanced MA strategy

- Tests baseline MA(20) vs Enhanced MA(20) with regime filter
- Multiple test periods: 2009-2011 bull, 2019-2025 OOS, full period
- Performance comparison and metrics calculation
- Regime visualization generation"
```

---

## Task 5: Document Results and Create Final Report

**Files:**
- Create: `docs/enhanced_ma_results.md`

**Step 1: Run full integration test**

```bash
python scripts/test_enhanced_ma.py > docs/enhanced_ma_test_output.txt 2>&1
```

**Step 2: Create results document**

Create `docs/enhanced_ma_results.md`:

```markdown
# Enhanced MA(20) with Regime Filter - Results Report

**Date:** 2025-03-23
**Strategy:** MA(20) + HMM Regime Detection
**Objective:** Improve full-period performance from 6.73% to 15-25% annual return

---

## Executive Summary

[Fill in after running tests]

### Key Findings

- [Finding 1]
- [Finding 2]
- [Finding 3]

### Performance Summary

| Period | Baseline MA(20) | Enhanced MA(20) | Improvement |
|--------|-----------------|-----------------|-------------|
| 2009-2011 | X% | Y% | +Z% |
| 2019-2025 | X% | Y% | +Z% |
| 2009-2025 | 6.73% | Y% | +Z% |

---

## Regime Analysis

[Describe detected regimes, their characteristics, and distribution]

---

## Conclusion

[Assessment of whether regime filtering achieved objectives]

---

## Primary Metric

**Full Period Annual Return:** [VALUE] (TARGET: 20%)
**Status:** [SUCCESS/FAILURE]
```

**Step 3: Fill in results from test output**

[Manual step: Copy results from test_output.txt into results.md]

**Step 4: Commit**

```bash
git add docs/enhanced_ma_results.md docs/enhanced_ma_test_output.txt docs/enhanced_ma_regimes.png
git commit -m "docs: Add Enhanced MA test results and analysis

- Complete performance report
- Regime visualization
- Comparison with baseline MA(20)"
```

---

## Task 6: Final Validation and Clean Up

**Files:**
- Modify: `README.md` (if applicable)

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: Verify no regressions**

```bash
python scripts/test_enhanced_ma.py
```

Expected: Results are consistent

**Step 3: Update project documentation**

Add entry to main README or appropriate documentation:

```markdown
## Enhanced MA(20) Strategy

Location: `strategies/technical/enhanced_ma_strategy.py`

Combines classic MA(20) trend following with Hidden Markov Model regime detection.

Usage:
```python
from strategies.technical.enhanced_ma_strategy import EnhancedMAStrategy
from strategies.regime.regime_detector import RegimeDetector

# Train regime detector
detector = RegimeDetector(n_states=3)
detector.fit(training_data)

# Configure strategy
config = {
    'ma_period': 20,
    'use_regime_filter': True,
    'confidence_threshold': 0.60,
    'base_position': 1.00,
}

strategy = EnhancedMAStrategy('TA', start, end, config)
strategy.regime_detector = detector

# Generate signals
signals = strategy.generate_signals(data)
```

Results: See `docs/enhanced_ma_results.md`
```

**Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: Update README with Enhanced MA strategy documentation"
```

---

## Success Criteria

The implementation is successful when:

✅ All unit tests pass (RegimeDetector, EnhancedMAStrategy)
✅ Integration test runs without errors
✅ Full period (2009-2025) annual return > 15% (baseline 6.73%)
✅ Regime visualization shows distinct bull/bear/ranging periods
✅ Code is documented and committed
✅ Results report is complete

**Stretch Goal:** Full period return > 20%

---

## Implementation Notes

**Dependencies:**
- `hmmlearn==0.3.0` - Hidden Markov Models
- Requires compilation - may need Microsoft C++ Build Tools on Windows

**Performance Considerations:**
- HMM training on 16 years of data may take 1-2 minutes
- Regime prediction is fast once model is trained
- Consider caching trained model for repeated backtests

**Known Limitations:**
- Regime detection has lag - may exit after trend reversal
- HMM requires sufficient historical data (>500 days)
- Random seed affects results - use fixed seed for reproducibility
