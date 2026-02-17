# trading/tests/test_config_manager.py
import pytest
import yaml
import os
from config.config_manager import ConfigManager

def test_config_manager_loads_config():
    """Test that ConfigManager can load the config file"""
    manager = ConfigManager()
    assert manager.config is not None
    assert 'instruments' in manager.config
    assert 'TA' in manager.config['instruments']

def test_get_instrument_config():
    """Test getting instrument-specific config"""
    manager = ConfigManager()
    ta_config = manager.get_instrument_config('TA')
    assert ta_config['name'] == 'PTA'
    assert ta_config['contract_size'] == 5
    assert ta_config['commission_rate'] == 0.0001

def test_get_data_path():
    """Test getting data paths"""
    manager = ConfigManager()
    assert 'raw_path' in manager.get_data_config()
