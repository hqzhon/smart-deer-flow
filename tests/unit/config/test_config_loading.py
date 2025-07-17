"""Unit tests for configuration loading."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

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
        settings = self.config_loader.load_configuration()
        assert settings is not None
        # Convert to dict for compatibility with existing tests
        config_data = settings.model_dump()
        assert isinstance(config_data, dict)

    def test_basic_model_config(self):
        """Test basic model configuration loading."""
        settings = self.config_loader.load_configuration()
        config_data = settings.model_dump()
        basic_model_config = config_data.get("llm", {})

        assert isinstance(basic_model_config, dict)
        # Check if basic_model key exists (may be empty but should be present)
        assert "basic_model" in basic_model_config or len(basic_model_config) == 0

    def test_search_config(self):
        """Test search configuration loading."""
        settings = self.config_loader.load_configuration()
        config_data = settings.model_dump()
        agents_config = config_data.get("agents", {})
        max_search_results = agents_config.get("max_search_results", 3)

        assert isinstance(max_search_results, int)
        assert max_search_results > 0

    def test_config_data_structure(self):
        """Test configuration data structure."""
        settings = self.config_loader.load_configuration()
        config_data = settings.model_dump()

        # Verify it's a dictionary
        assert isinstance(config_data, dict)

        # Test common configuration keys (if they exist)
        expected_keys = ["llm", "agents"]
        for key in expected_keys:
            if key in config_data:
                assert config_data[key] is not None

    def test_config_loading_consistency(self):
        """Test that config loading is consistent across multiple calls."""
        settings1 = self.config_loader.load_configuration()
        settings2 = (
            self.config_loader.get_settings()
        )  # Use get_settings for cached access

        # Should return the same configuration
        assert settings1.model_dump() == settings2.model_dump()

    def test_config_loading_with_different_instances(self):
        """Test config loading with different ConfigLoader instances."""
        loader1 = ConfigLoader()
        loader2 = ConfigLoader()

        settings1 = loader1.load_configuration()
        settings2 = loader2.load_configuration()

        # Should return the same configuration
        assert settings1.model_dump() == settings2.model_dump()

    def test_config_values_types(self):
        """Test that configuration values have expected types."""
        settings = self.config_loader.load_configuration()
        config_data = settings.model_dump()

        # Test max_search_results type
        agents_config = config_data.get("agents", {})
        if "max_search_results" in agents_config:
            assert isinstance(agents_config["max_search_results"], int)

        # Test llm type
        if "llm" in config_data:
            assert isinstance(config_data["llm"], dict)

    def test_config_default_values(self):
        """Test configuration default values."""
        settings = self.config_loader.load_configuration()
        config_data = settings.model_dump()

        # Test default search results
        agents_config = config_data.get("agents", {})
        max_search_results = agents_config.get("max_search_results", 3)
        assert max_search_results >= 1  # Should be at least 1

        # Test llm config structure
        llm_config = config_data.get("llm", {})
        assert isinstance(llm_config, dict)

    def test_config_error_handling(self):
        """Test configuration error handling."""
        # This test verifies that config loading doesn't crash
        # even if there are issues with the configuration
        try:
            settings = self.config_loader.load_configuration()
            assert settings is not None
        except Exception as e:
            # If an exception occurs, it should be a known type
            assert isinstance(e, (FileNotFoundError, ValueError, KeyError))

    @pytest.mark.parametrize("expected_key", ["llm", "agents"])
    def test_specific_config_keys(self, expected_key):
        """Test specific configuration keys."""
        settings = self.config_loader.load_configuration()
        config_data = settings.model_dump()

        # Key may or may not exist, but if it does, it should have valid value
        if expected_key in config_data:
            value = config_data[expected_key]
            assert value is not None

            if expected_key == "agents":
                assert isinstance(value, dict)
                # Check max_search_results if it exists
                if "max_search_results" in value:
                    assert (
                        isinstance(value["max_search_results"], int)
                        and value["max_search_results"] > 0
                    )
            elif expected_key == "llm":
                assert isinstance(value, dict)
