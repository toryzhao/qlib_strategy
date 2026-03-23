"""
Regime Detection using Hidden Markov Models

This module implements a regime detector that classifies market conditions
into Bull, Ranging, and Bear states using Hidden Markov Models (HMM).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class RegimeDetector:
    """
    Market regime detector using Hidden Markov Models.

    Classifies market conditions into discrete regimes (e.g., Bull, Ranging, Bear)
    based on price and volatility features using Gaussian HMM.

    Parameters
    ----------
    n_states : int, default=3
        Number of hidden states (regimes) to detect.
    covariance_type : str, default='full'
        Type of covariance parameters for HMM ('full', 'diag', 'spherical', 'tied').
    random_state : int, default=None
        Random seed for reproducibility.
    min_obs : int, default=100
        Minimum number of observations required for fitting.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = 'full',
        random_state: Optional[int] = None,
        min_obs: int = 100
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.min_obs = min_obs
        self.model: Optional[hmm.GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols = ['log_returns', 'volatility', 'ma_slope', 'acceleration']

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for HMM training/prediction.

        Features:
        - log_returns: Log price changes
        - volatility: Rolling standard deviation of returns
        - ma_slope: Moving average slope (trend indicator)
        - acceleration: Rate of change of returns

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close' column and datetime index.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated features.
        """
        features = pd.DataFrame(index=df.index)

        # Log returns
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility (20-day rolling std)
        features['volatility'] = features['log_returns'].rolling(window=20).std()

        # Moving average slope (20-day MA)
        ma20 = df['close'].rolling(window=20).mean()
        features['ma_slope'] = ma20.diff()

        # Acceleration (rate of change of returns)
        features['acceleration'] = features['log_returns'].diff()

        return features

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the HMM model to price data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close' column and datetime index.
            Must have at least min_obs observations.

        Raises
        ------
        ValueError
            If insufficient data or missing 'close' column.
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        if len(df) < self.min_obs:
            raise ValueError(
                f"Insufficient data: {len(df)} observations. "
                f"Need at least {self.min_obs}"
            )

        # Calculate features
        features = self._calculate_features(df)

        # Remove NaN values
        features_clean = features[self.feature_cols].dropna()

        if len(features_clean) < self.min_obs:
            raise ValueError(
                f"Insufficient clean data after feature calculation: "
                f"{len(features_clean)} observations. Need at least {self.min_obs}"
            )

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features_clean)

        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_iter=1000,
            tol=1e-6
        )

        self.model.fit(X)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime labels and probabilities for price data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close' column and datetime index.

        Returns
        -------
        regimes : np.ndarray
            Array of regime labels (0, 1, 2, ...).
        probabilities : np.ndarray
            Array of shape (n_samples, n_states) with regime probabilities.

        Raises
        ------
        ValueError
            If model hasn't been fitted.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Calculate features
        features = self._calculate_features(df)

        # Get feature subset and drop NaN values
        features_subset = features[self.feature_cols].dropna()

        if len(features_subset) == 0:
            raise ValueError("No valid data for prediction after feature calculation")

        # Scale features
        X = self.scaler.transform(features_subset)

        # Predict regimes
        regimes = self.model.predict(X)

        # Get posterior probabilities
        probabilities = self.model.predict_proba(X)

        # Align with original index
        regimes_full = np.full(len(df), -1)
        probs_full = np.zeros((len(df), self.n_states))

        valid_idx = features_subset.index
        for i, idx in enumerate(valid_idx):
            pos = df.index.get_loc(idx)
            regimes_full[pos] = regimes[i]
            probs_full[pos] = probabilities[i]

        return regimes_full, probs_full

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Get the current regime and confidence level.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close' column and datetime index.

        Returns
        -------
        regime : int
            Current regime label.
        confidence : float
            Confidence level (probability of current regime).

        Raises
        ------
        ValueError
            If model hasn't been fitted or no valid prediction available.
        """
        regimes, probabilities = self.predict(df)

        # Get last valid prediction
        valid_mask = regimes != -1
        if not valid_mask.any():
            raise ValueError("No valid predictions available")

        last_valid_idx = np.where(valid_mask)[0][-1]
        regime = regimes[last_valid_idx]
        confidence = probabilities[last_valid_idx, regime]

        return int(regime), float(confidence)

    def get_regime_name(self, state: int) -> str:
        """
        Convert regime state number to human-readable name.

        Parameters
        ----------
        state : int
            Regime state number.

        Returns
        -------
        str
            Regime name (Bear, Ranging, or Bull for 3-state model).

        Raises
        ------
        ValueError
            If state is out of valid range.
        """
        if state < 0 or state >= self.n_states:
            raise ValueError(f"State must be between 0 and {self.n_states - 1}")

        if self.n_states == 3:
            names = ['Bear', 'Ranging', 'Bull']
            return names[state]
        else:
            return f"Regime_{state}"

    def plot_regimes(
        self,
        df: pd.DataFrame,
        regimes: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Visualize price data with regime classifications.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close' column and datetime index.
        regimes : np.ndarray, optional
            Pre-computed regime labels. If None, will predict.
        figsize : Tuple[int, int], default=(14, 8)
            Figure size.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        if regimes is None:
            regimes, _ = self.predict(df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot price with regime colors
        ax1.plot(df.index, df['close'], label='Price', color='black', alpha=0.6)

        # Color code regimes
        colors = ['red', 'yellow', 'green'][:self.n_states]
        for i in range(self.n_states):
            mask = regimes == i
            if mask.any():
                ax1.scatter(
                    df.index[mask],
                    df['close'].values[mask],
                    c=colors[i],
                    label=f'{self.get_regime_name(i)} (State {i})',
                    alpha=0.3,
                    s=20
                )

        ax1.set_ylabel('Price')
        ax1.set_title('Price with Regime Classifications')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot regime over time
        ax2.plot(df.index, regimes, drawstyle='steps-mid', color='blue')
        ax2.set_ylabel('Regime State')
        ax2.set_xlabel('Date')
        ax2.set_title('Regime Transitions Over Time')
        ax2.set_yticks(range(self.n_states))
        ax2.set_yticklabels(
            [self.get_regime_name(i) for i in range(self.n_states)]
        )
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
