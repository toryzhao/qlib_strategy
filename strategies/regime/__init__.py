"""
Market Regime Detection Module

This module provides statistical models for detecting market regimes
(bull/bear/ranging) using Hidden Markov Models.
"""

__version__ = "0.1.0"

from .market_regime_detector import MarketRegimeDetector

__all__ = ['MarketRegimeDetector']
