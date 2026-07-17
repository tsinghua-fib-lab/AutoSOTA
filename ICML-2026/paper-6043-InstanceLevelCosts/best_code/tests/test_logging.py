"""
Tests for src/core/logging.py wandb wrapper.

These tests cover the disabled mode behavior (WANDB_MODE=disabled) which is
the critical path for ensuring the wrapper doesn't break code when wandb
is disabled. Enabled mode is validated through actual experiment runs.
"""

import os
import pytest
from pathlib import Path

# Set WANDB_MODE=disabled before importing the module
os.environ['WANDB_MODE'] = 'disabled'

from core import logging as log_module


# =============================================================================
# Disabled Mode Tests
# =============================================================================

class TestDisabledMode:
    """Tests for wandb disabled mode (no-op behavior)."""

    def setup_method(self):
        """Ensure disabled mode for each test."""
        os.environ['WANDB_MODE'] = 'disabled'
        # Reset module state
        log_module._wandb_enabled = False
        log_module._current_run = None

    def test_is_enabled_returns_false_when_disabled(self):
        """Test _is_enabled returns False when WANDB_MODE=disabled."""
        assert log_module._is_enabled() is False

    def test_init_run_returns_none_when_disabled(self):
        """Test init_run returns None when disabled."""
        result = log_module.init_run(project='test', name='test-run')
        assert result is None

    def test_is_enabled_false_after_init_when_disabled(self):
        """Test is_enabled() returns False after init when disabled."""
        log_module.init_run(project='test')
        assert log_module.is_enabled() is False

    def test_get_run_returns_none_when_disabled(self):
        """Test get_run returns None when disabled."""
        log_module.init_run(project='test')
        assert log_module.get_run() is None

    def test_log_metrics_noop_when_disabled(self):
        """Test log_metrics is a no-op when disabled."""
        log_module.init_run(project='test')
        # Should not raise
        log_module.log_metrics({'accuracy': 0.95, 'loss': 0.1}, step=1)

    def test_log_summary_noop_when_disabled(self):
        """Test log_summary is a no-op when disabled."""
        log_module.init_run(project='test')
        # Should not raise
        log_module.log_summary({'final_accuracy': 0.95})

    def test_log_artifact_returns_none_when_disabled(self, tmp_path):
        """Test log_artifact returns None when disabled."""
        log_module.init_run(project='test')
        test_file = tmp_path / 'test.txt'
        test_file.write_text('test content')

        result = log_module.log_artifact(test_file, name='test-artifact', type='file')
        assert result is None

    def test_log_table_noop_when_disabled(self):
        """Test log_table is a no-op when disabled."""
        log_module.init_run(project='test')
        # Should not raise
        log_module.log_table(
            key='results',
            columns=['model', 'accuracy'],
            data=[['model1', 0.9], ['model2', 0.85]],
        )

    def test_finish_run_noop_when_disabled(self):
        """Test finish_run is a no-op when disabled."""
        log_module.init_run(project='test')
        # Should not raise
        log_module.finish_run()

    def test_update_config_noop_when_disabled(self):
        """Test update_config is a no-op when disabled."""
        log_module.init_run(project='test')
        # Should not raise
        log_module.update_config({'learning_rate': 0.01})


# =============================================================================
# Module State Tests
# =============================================================================

class TestModuleState:
    """Tests for module state management."""

    def setup_method(self):
        """Reset state for each test."""
        os.environ['WANDB_MODE'] = 'disabled'
        log_module._wandb_enabled = False
        log_module._current_run = None

    def test_wandb_available_is_boolean(self):
        """Test _wandb_available is a boolean."""
        assert isinstance(log_module._wandb_available, bool)

    def test_initial_state_not_enabled(self):
        """Test initial state has logging disabled."""
        assert log_module._wandb_enabled is False
        assert log_module._current_run is None

    def test_multiple_init_finish_cycles(self):
        """Test multiple init/finish cycles work correctly."""
        # First cycle
        log_module.init_run(project='test1')
        assert log_module.is_enabled() is False  # Disabled mode
        log_module.finish_run()
        assert log_module.get_run() is None

        # Second cycle
        log_module.init_run(project='test2')
        assert log_module.is_enabled() is False
        log_module.finish_run()
        assert log_module.get_run() is None

    def test_finish_without_init_is_safe(self):
        """Test calling finish_run without init doesn't raise."""
        # Should not raise
        log_module.finish_run()
        log_module.finish_run()

    def test_log_without_init_is_safe(self):
        """Test logging without init doesn't raise."""
        # Should not raise even without init
        log_module.log_metrics({'test': 1.0})
        log_module.log_summary({'test': 1.0})
        log_module.log_table('test', ['col'], [['val']])
        log_module.update_config({'test': 1})
