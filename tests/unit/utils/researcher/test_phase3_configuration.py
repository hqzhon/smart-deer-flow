#!/usr/bin/env python3
"""
Unit tests for Phase 3 configuration parameters.

This module tests the Phase 3 configuration system for researcher context isolation,
including new parameters and their default values.
"""

import pytest
from unittest.mock import Mock, patch

from src.config.configuration import Configuration


class TestPhase3Configuration:
    """Test suite for Phase 3 configuration parameters."""

    def test_phase3_configuration_parameters_exist(self):
        """Test that all Phase 3 configuration parameters exist."""
        config = Configuration()
        
        phase3_params = [
            'enable_researcher_isolation',
            'researcher_isolation_level', 
            'researcher_max_local_context',
            'researcher_isolation_threshold',
            'researcher_auto_isolation',
            'researcher_isolation_metrics',
            'max_context_steps_researcher'
        ]
        
        for param in phase3_params:
            assert hasattr(config, param), f"Configuration missing parameter: {param}"
    
    def test_enable_researcher_isolation_default(self):
        """Test default value for enable_researcher_isolation."""
        config = Configuration()
        value = getattr(config, 'enable_researcher_isolation', None)
        assert value is not None, "enable_researcher_isolation should have a default value"
    
    def test_researcher_isolation_level_default(self):
        """Test default value for researcher_isolation_level."""
        config = Configuration()
        value = getattr(config, 'researcher_isolation_level', None)
        assert value is not None, "researcher_isolation_level should have a default value"
        assert isinstance(value, str), "researcher_isolation_level should be a string"
    
    def test_researcher_max_local_context_default(self):
        """Test default value for researcher_max_local_context."""
        config = Configuration()
        value = getattr(config, 'researcher_max_local_context', None)
        assert value is not None, "researcher_max_local_context should have a default value"
        assert isinstance(value, int), "researcher_max_local_context should be an integer"
        assert value > 0, "researcher_max_local_context should be positive"
    
    def test_researcher_isolation_threshold_default(self):
        """Test default value for researcher_isolation_threshold."""
        config = Configuration()
        value = getattr(config, 'researcher_isolation_threshold', None)
        assert value is not None, "researcher_isolation_threshold should have a default value"
        assert isinstance(value, (int, float)), "researcher_isolation_threshold should be numeric"
        assert 0 <= value <= 1, "researcher_isolation_threshold should be between 0 and 1"
    
    def test_researcher_auto_isolation_default(self):
        """Test default value for researcher_auto_isolation."""
        config = Configuration()
        value = getattr(config, 'researcher_auto_isolation', None)
        assert value is not None, "researcher_auto_isolation should have a default value"
        assert isinstance(value, bool), "researcher_auto_isolation should be a boolean"
    
    def test_researcher_isolation_metrics_default(self):
        """Test default value for researcher_isolation_metrics."""
        config = Configuration()
        value = getattr(config, 'researcher_isolation_metrics', None)
        assert value is not None, "researcher_isolation_metrics should have a default value"
        assert isinstance(value, bool), "researcher_isolation_metrics should be a boolean"
    
    def test_max_context_steps_researcher_default(self):
        """Test default value for max_context_steps_researcher."""
        config = Configuration()
        value = getattr(config, 'max_context_steps_researcher', None)
        assert value is not None, "max_context_steps_researcher should have a default value"
        assert isinstance(value, int), "max_context_steps_researcher should be an integer"
        assert value > 0, "max_context_steps_researcher should be positive"
    
    def test_configuration_consistency(self):
        """Test that configuration values are consistent and valid."""
        config = Configuration()
        
        # Test that threshold is reasonable
        threshold = getattr(config, 'researcher_isolation_threshold', 0.5)
        assert 0 <= threshold <= 1, "Isolation threshold should be between 0 and 1"
        
        # Test that max context values are reasonable
        max_local = getattr(config, 'researcher_max_local_context', 1000)
        max_steps = getattr(config, 'max_context_steps_researcher', 1)
        
        assert max_local > 0, "Max local context should be positive"
        assert max_steps > 0, "Max context steps should be positive"
        
        # Test that isolation level is valid
        isolation_level = getattr(config, 'researcher_isolation_level', 'moderate')
        valid_levels = ['low', 'moderate', 'high', 'strict']
        assert isolation_level in valid_levels, f"Isolation level should be one of {valid_levels}"
    
    @patch('src.config.configuration.Configuration')
    def test_configuration_with_custom_values(self, mock_config_class):
        """Test configuration with custom values."""
        # Create a mock configuration with custom values
        mock_config = Mock()
        mock_config.enable_researcher_isolation = True
        mock_config.researcher_isolation_level = 'high'
        mock_config.researcher_max_local_context = 5000
        mock_config.researcher_isolation_threshold = 0.8
        mock_config.researcher_auto_isolation = False
        mock_config.researcher_isolation_metrics = True
        mock_config.max_context_steps_researcher = 3
        
        mock_config_class.return_value = mock_config
        
        config = Configuration()
        
        assert config.enable_researcher_isolation is True
        assert config.researcher_isolation_level == 'high'
        assert config.researcher_max_local_context == 5000
        assert config.researcher_isolation_threshold == 0.8
        assert config.researcher_auto_isolation is False
        assert config.researcher_isolation_metrics is True
        assert config.max_context_steps_researcher == 3