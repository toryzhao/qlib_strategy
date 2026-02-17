# trading/tests/test_data_processor.py
import pytest
import pandas as pd
from utils.data_processor import ContinuousContractProcessor

def test_load_csv_data():
    """Test loading CSV data"""
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    assert processor.df is not None
    assert len(processor.df) > 0
    assert 'datetime' in processor.df.columns
    assert 'close' in processor.df.columns

def test_data_columns():
    """Test that all required columns exist"""
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount', 'position', 'symbol']
    for col in required_cols:
        assert col in processor.df.columns

def test_datetime_parsing():
    """Test that datetime is correctly parsed"""
    processor = ContinuousContractProcessor('data/raw/TA.csv')
    assert pd.api.types.is_datetime64_any_dtype(processor.df['datetime'])
