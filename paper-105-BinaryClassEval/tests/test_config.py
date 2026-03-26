"""Tests for the configuration system."""
import os
import pytest
import tempfile
from pathlib import Path

from reproducibility.config import Configuration, DEFAULT_CONFIG
from reproducibility.seed_manager import get_default_seed


def test_config_default_values():
    """Test that the configuration system has the expected default values."""
    config = Configuration()
    assert config['seed'] == get_default_seed()
    assert config['output_dir'] == './output'
    assert config['log_level'] == 'INFO'


def test_config_override():
    """Test that configuration values can be overridden."""
    config = Configuration({'seed': 123, 'custom_value': 'test'})
    assert config['seed'] == 123
    assert config['custom_value'] == 'test'
    assert config['output_dir'] == './output'  # Default not overridden


def test_config_item_access():
    """Test item access for configuration values."""
    config = Configuration()
    config['new_key'] = 'value'
    assert config['new_key'] == 'value'


def test_seed_environment_override():
    """Test that the seed can be overridden by an environment variable."""
    # Save original environment variable if it exists
    original_env = os.environ.get('EPAMNB_SEED')
    
    try:
        # Set environment variable
        os.environ['EPAMNB_SEED'] = '789'
        config = Configuration()
        assert config.seed == 789
        
        # Invalid environment variable should fall back to default
        os.environ['EPAMNB_SEED'] = 'not_a_number'
        config = Configuration()
        assert config.seed == get_default_seed()
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ['EPAMNB_SEED'] = original_env
        else:
            os.environ.pop('EPAMNB_SEED', None)


def test_analysis_logging():
    """Test that analysis parameters are properly logged."""
    config = Configuration({'seed': 42})
    config.log_analysis('test_analysis', {'param1': 'value1'})
    
    # Check that the analysis was logged with the seed
    assert len(config._analysis_log) == 1
    log_entry = config._analysis_log[0]
    assert log_entry['analysis'] == 'test_analysis'
    assert log_entry['parameters']['param1'] == 'value1'
    assert log_entry['parameters']['seed'] == 42
    
    # If seed is explicitly provided in parameters, it shouldn't be overridden
    config.log_analysis('another_test', {'param2': 'value2', 'seed': 123})
    log_entry = config._analysis_log[1]
    assert log_entry['parameters']['seed'] == 123


def test_save_analysis_log():
    """Test saving the analysis log to a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Configuration({'output_dir': tmpdir})
        config.log_analysis('test_analysis', {'param1': 'value1'})
        
        log_file = 'test_log.json'
        config.save_analysis_log(log_file)
        
        # Check that the file was created
        log_path = Path(tmpdir) / log_file
        assert log_path.exists()
