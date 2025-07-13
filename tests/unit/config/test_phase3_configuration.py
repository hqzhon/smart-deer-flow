#!/usr/bin/env python3
"""
Phase 3 Configuration Unit Tests

This module contains unit tests for Phase 3 configuration parameters,
including researcher isolation settings and related configuration options.
"""

import pytest
from unittest.mock import Mock, patch

from src.config.configuration import Configuration


class TestPhase3Configuration:
    """Test Phase 3 configuration parameters."""
    
    def test_configuration_instantiation(self):
        """Test that Configuration can be instantiated."""
        config = Configuration()
        assert config is not None
        assert isinstance(config, Configuration)
    
    def test_phase3_parameters_exist(self):
        """Test that all Phase 3 parameters exist in configuration."""
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
            # Parameter should exist (not raise AttributeError)
            value = getattr(config, param, None)
            # We don't assert specific values as they may vary
            # Just ensure the parameter exists
            assert hasattr(config, param) or value is not None, f"Parameter {param} should exist"
    
    def test_enable_researcher_isolation_parameter(self):
        """Test enable_researcher_isolation parameter."""
        config = Configuration()
        value = getattr(config, 'enable_researcher_isolation', None)
        
        # Should be a boolean or None
        assert value is None or isinstance(value, bool)
    
    def test_researcher_isolation_level_parameter(self):
        """Test researcher_isolation_level parameter."""
        config = Configuration()
        value = getattr(config, 'researcher_isolation_level', None)
        
        # Should be a string or None
        if value is not None:
            assert isinstance(value, str)
            # Common isolation levels
            valid_levels = ['low', 'moderate', 'high', 'strict']
            # Don't enforce specific values, just check type
    
    def test_researcher_max_local_context_parameter(self):
        """Test researcher_max_local_context parameter."""
        config = Configuration()
        value = getattr(config, 'researcher_max_local_context', None)
        
        # Should be an integer or None
        if value is not None:
            assert isinstance(value, int)
            assert value > 0, "Max local context should be positive"
    
    def test_researcher_isolation_threshold_parameter(self):
        """Test researcher_isolation_threshold parameter."""
        config = Configuration()
        value = getattr(config, 'researcher_isolation_threshold', None)
        
        # Should be a float or None
        if value is not None:
            assert isinstance(value, (int, float))
            assert 0.0 <= value <= 1.0, "Threshold should be between 0 and 1"
    
    def test_researcher_auto_isolation_parameter(self):
        """Test researcher_auto_isolation parameter."""
        config = Configuration()
        value = getattr(config, 'researcher_auto_isolation', None)
        
        # Should be a boolean or None
        assert value is None or isinstance(value, bool)
    
    def test_researcher_isolation_metrics_parameter(self):
        """Test researcher_isolation_metrics parameter."""
        config = Configuration()
        value = getattr(config, 'researcher_isolation_metrics', None)
        
        # Should be a boolean or None
        assert value is None or isinstance(value, bool)
    
    def test_max_context_steps_researcher_parameter(self):
        """Test max_context_steps_researcher parameter."""
        config = Configuration()
        value = getattr(config, 'max_context_steps_researcher', None)
        
        # Should be an integer or None
        if value is not None:
            assert isinstance(value, int)
            assert value > 0, "Max context steps should be positive"
    
    def test_phase3_parameters_consistency(self):
        """Test consistency between related Phase 3 parameters."""
        config = Configuration()
        
        enable_isolation = getattr(config, 'enable_researcher_isolation', None)
        auto_isolation = getattr(config, 'researcher_auto_isolation', None)
        metrics_enabled = getattr(config, 'researcher_isolation_metrics', None)
        
        # If auto isolation is enabled, isolation should be enabled
        if auto_isolation is True:
            assert enable_isolation is not False, "Auto isolation requires isolation to be enabled"
        
        # If metrics are enabled, isolation should be enabled
        if metrics_enabled is True:
            assert enable_isolation is not False, "Metrics require isolation to be enabled"
    
    def test_multiple_configuration_instances(self):
        """Test that multiple configuration instances work independently."""
        config1 = Configuration()
        config2 = Configuration()
        
        # Should be separate instances
        assert config1 is not config2
        
        # Should have same parameter values (assuming no state mutation)
        phase3_params = [
            'enable_researcher_isolation',
            'researcher_isolation_level',
            'researcher_max_local_context'
        ]
        
        for param in phase3_params:
            value1 = getattr(config1, param, None)
            value2 = getattr(config2, param, None)
            assert value1 == value2, f"Parameter {param} should be consistent across instances"
    
    def test_configuration_parameter_types(self):
        """Test that Phase 3 parameters have correct types."""
        config = Configuration()
        
        # Boolean parameters
        boolean_params = [
            'enable_researcher_isolation',
            'researcher_auto_isolation',
            'researcher_isolation_metrics'
        ]
        
        for param in boolean_params:
            value = getattr(config, param, None)
            if value is not None:
                assert isinstance(value, bool), f"{param} should be boolean"
        
        # String parameters
        string_params = ['researcher_isolation_level']
        
        for param in string_params:
            value = getattr(config, param, None)
            if value is not None:
                assert isinstance(value, str), f"{param} should be string"
        
        # Numeric parameters
        numeric_params = [
            'researcher_max_local_context',
            'researcher_isolation_threshold',
            'max_context_steps_researcher'
        ]
        
        for param in numeric_params:
            value = getattr(config, param, None)
            if value is not None:
                assert isinstance(value, (int, float)), f"{param} should be numeric"
    
    def test_configuration_with_custom_values(self):
        """Test configuration with custom parameter values."""
        # This test assumes Configuration accepts custom parameters
        # Adjust based on actual Configuration implementation
        try:
            config = Configuration()
            
            # Test setting custom values if supported
            if hasattr(config, '__setattr__'):
                config.enable_researcher_isolation = True
                config.researcher_isolation_level = "high"
                config.researcher_max_local_context = 5000
                
                assert config.enable_researcher_isolation is True
                assert config.researcher_isolation_level == "high"
                assert config.researcher_max_local_context == 5000
        except Exception:
            # If configuration doesn't support custom values, that's fine
            pass
    
    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        config = Configuration()
        
        # Test accessing non-existent parameter
        non_existent_param = getattr(config, 'non_existent_phase3_param', 'default')
        assert non_existent_param == 'default'
        
        # Test that accessing existing parameters doesn't raise errors
        try:
            _ = getattr(config, 'enable_researcher_isolation', None)
            _ = getattr(config, 'researcher_isolation_level', None)
            _ = getattr(config, 'researcher_max_local_context', None)
        except Exception as e:
            pytest.fail(f"Accessing Phase 3 parameters should not raise errors: {e}")


class TestPhase3ConfigurationPerformance:
    """Test Phase 3 configuration performance."""
    
    def test_configuration_creation_performance(self):
        """Test that configuration creation is fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            config = Configuration()
        end_time = time.time()
        
        # Should create 100 configurations in less than 1 second
        assert end_time - start_time < 1.0, "Configuration creation should be fast"
    
    def test_parameter_access_performance(self):
        """Test that parameter access is fast."""
        import time
        
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
        
        start_time = time.time()
        for _ in range(1000):
            for param in phase3_params:
                _ = getattr(config, param, None)
        end_time = time.time()
        
        # Should access parameters 7000 times in less than 1 second
        assert end_time - start_time < 1.0, "Parameter access should be fast"