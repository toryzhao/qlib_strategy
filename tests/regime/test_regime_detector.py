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
    # Check probabilities sum to 1 for valid predictions only
    valid_mask = regimes != -1
    assert np.allclose(probs[valid_mask].sum(axis=1), 1.0)  # Probabilities sum to 1

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
