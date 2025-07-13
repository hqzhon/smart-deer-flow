"""Unit tests for configuration loading."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from src.config.config_loader import ConfigLoader


class TestConfigLoading:
    """Test cases for configuration loading."""
    
    def setup_method(self):
        """Test setup."""
        self.config_loader = ConfigLoader()
    
    def test_config_loader_instantiation(self):
        """Test ConfigLoader instantiation."""
        assert self.config_loader is not None
        assert isinstance(self.config_loader, ConfigLoader)
    
    def test_basic_config_loading(self):
        """Test basic configuration loading."""
        config_data = self.config_loader.load_config()
        assert config_data is not None
        assert isinstance(config_data, dict)
    
    def test_basic_model_config(self):
        """Test basic model configuration loading."""
        config_data = self.config_loader.load_config()
        basic_model_config = config_data.get('BASIC_MODEL', {})
        
        assert isinstance(basic_model_config, dict)
        # Check if model key exists (may be empty but should be present)
        assert 'model' in basic_model_config or len(basic_model_config) == 0
    
    def test_search_config(self):
        """Test search configuration loading."""
        config_data = self.config_loader.load_config()
        max_search_results = config_data.get('max_search_results', 3)
        
        assert isinstance(max_search_results, int)
        assert max_search_results > 0
    
    def test_config_data_structure(self):
        """Test configuration data structure."""
        config_data = self.config_loader.load_config()
        
        # Verify it's a dictionary
        assert isinstance(config_data, dict)
        
        # Test common configuration keys (if they exist)
        expected_keys = ['BASIC_MODEL', 'max_search_results']
        for key in expected_keys:
            if key in config_data:
                assert config_data[key] is not None
    
    def test_config_loading_consistency(self):
        """Test that config loading is consistent across multiple calls."""
        config1 = self.config_loader.load_config()
        config2 = self.config_loader.load_config()
        
        # Should return the same configuration
        assert config1 == config2
    
    def test_config_loading_with_different_instances(self):
        """Test config loading with different ConfigLoader instances."""
        loader1 = ConfigLoader()
        loader2 = ConfigLoader()
        
        config1 = loader1.load_config()
        config2 = loader2.load_config()
        
        # Should return the same configuration
        assert config1 == config2
    
    def test_config_values_types(self):
        """Test that configuration values have expected types."""
        config_data = self.config_loader.load_config()
        
        # Test max_search_results type
        if 'max_search_results' in config_data:
            assert isinstance(config_data['max_search_results'], int)
        
        # Test BASIC_MODEL type
        if 'BASIC_MODEL' in config_data:
            assert isinstance(config_data['BASIC_MODEL'], dict)
    
    def test_config_default_values(self):
        """Test configuration default values."""
        config_data = self.config_loader.load_config()
        
        # Test default search results
        max_search_results = config_data.get('max_search_results', 3)
        assert max_search_results >= 1  # Should be at least 1
        
        # Test basic model config structure
        basic_model_config = config_data.get('BASIC_MODEL', {})
        assert isinstance(basic_model_config, dict)
    
    def test_config_error_handling(self):
        """Test configuration error handling."""
        # This test verifies that config loading doesn't crash
        # even if there are issues with the configuration
        try:
            config_data = self.config_loader.load_config()
            assert config_data is not None
        except Exception as e:
            # If an exception occurs, it should be a known type
            assert isinstance(e, (FileNotFoundError, ValueError, KeyError))
    
    @pytest.mark.parametrize("expected_key", [
        "BASIC_MODEL",
        "max_search_results"
    ])
    def test_specific_config_keys(self, expected_key):
        """Test specific configuration keys."""
        config_data = self.config_loader.load_config()
        
        # Key may or may not exist, but if it does, it should have valid value
        if expected_key in config_data:
            value = config_data[expected_key]
            assert value is not None
            
            if expected_key == "max_search_results":
                assert isinstance(value, int) and value > 0
            elif expected_key == "BASIC_MODEL":
                assert isinstance(value, dict)