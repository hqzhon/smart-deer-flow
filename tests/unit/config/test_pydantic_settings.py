"""Tests for the new pydantic-settings based configuration system."""

import os
import pytest
from unittest.mock import patch

from src.config.models import AppSettings
from src.config.config_loader import get_settings, get_config_loader


class TestPydanticSettingsConfig:
    """Test the new pydantic-settings based configuration system."""

    @pytest.fixture(autouse=True)
    def clear_config_cache(self):
        """Clear the global configuration cache before each test."""
        loader = get_config_loader()
        loader._settings = None
        yield
        # Clean up after test
        loader._settings = None

    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        settings = AppSettings()

        assert settings.report_style == "academic"
        assert settings.llm.temperature == 0.7
        assert settings.agents.max_plan_iterations == 1
        assert settings.research.enable_researcher_isolation is True
        assert settings.reflection.enable_enhanced_reflection is True

    def test_environment_variable_loading(self):
        """Test that environment variables are automatically loaded."""
        # Test without environment variables first to see defaults
        settings = AppSettings()
        assert settings.report_style == "academic"  # default value

        env_vars = {
            "DEER_REPORT_STYLE": "business",
            "DEER_LLM__TEMPERATURE": "0.9",
            "DEER_LLM__MAX_TOKENS": "2000",
            "DEER_AGENTS__MAX_PLAN_ITERATIONS": "5",
            "DEER_AGENTS__ENABLE_DEEP_THINKING": "true",
            "DEER_RESEARCH__ENABLE_RESEARCHER_ISOLATION": "false",
            "DEER_RESEARCH__RESEARCHER_ISOLATION_LEVEL": "aggressive",
            "DEER_REFLECTION__MAX_REFLECTION_LOOPS": "5",
            "DEER_MCP__ENABLED": "true",
            "DEER_MCP__TIMEOUT": "60",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = AppSettings()

            # Test core settings
            assert settings.report_style == "business"

            # Test LLM settings
            assert settings.llm.temperature == 0.9
            assert settings.llm.max_tokens == 2000

            # Test agent settings
            assert settings.agents.max_plan_iterations == 5
            assert settings.agents.enable_deep_thinking is True

            # Test research settings
            assert settings.research.enable_researcher_isolation is False
            assert settings.research.researcher_isolation_level == "aggressive"

            # Test reflection settings
            assert settings.reflection.max_reflection_loops == 5

            # Test MCP settings
            assert settings.mcp.enabled is True
            assert settings.mcp.timeout == 60

    def test_nested_environment_variables(self):
        """Test that nested environment variables are loaded correctly."""
        # Set nested environment variables
        os.environ["DEER_AGENTS__MAX_PLAN_ITERATIONS"] = "8"
        os.environ["DEER_RESEARCH__RESEARCHER_ISOLATION_THRESHOLD"] = "0.8"

        try:
            settings = get_settings()

            # Check nested values
            assert settings.agents.max_plan_iterations == 8
            assert settings.research.researcher_isolation_threshold == 0.8

        finally:
            # Clean up
            os.environ.pop("DEER_AGENTS__MAX_PLAN_ITERATIONS", None)
            os.environ.pop("DEER_RESEARCH__RESEARCHER_ISOLATION_THRESHOLD", None)

    def test_type_conversion(self):
        """Test that environment variables are properly converted to correct types."""
        env_vars = {
            "DEER_LLM__TEMPERATURE": "0.5",  # float
            "DEER_AGENTS__MAX_PLAN_ITERATIONS": "10",  # int
            "DEER_AGENTS__ENABLE_DEEP_THINKING": "true",  # bool
            "DEER_AGENTS__ENABLE_PARALLEL_EXECUTION": "false",  # bool
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = AppSettings()

            assert isinstance(settings.llm.temperature, float)
            assert settings.llm.temperature == 0.5

            assert isinstance(settings.agents.max_plan_iterations, int)
            assert settings.agents.max_plan_iterations == 10

            assert isinstance(settings.agents.enable_deep_thinking, bool)
            assert settings.agents.enable_deep_thinking is True

            assert isinstance(settings.agents.enable_parallel_execution, bool)
            assert settings.agents.enable_parallel_execution is False

    def test_get_settings_function(self):
        """Test the get_settings function."""
        # Test loading with environment variables
        env_vars = {
            "DEER_REPORT_STYLE": "technical",
            "DEER_LLM__TEMPERATURE": "0.3",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = get_settings()

            assert settings.report_style == "technical"
            assert settings.llm.temperature == 0.3

    def test_yaml_override_with_env_vars(self, tmp_path):
        """Test that environment variables override default values."""
        # Set environment variables that should override defaults
        env_vars = {
            "DEER_REPORT_STYLE": "business",
            "DEER_LLM__TEMPERATURE": "0.9",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = get_settings()

            # Environment variables should override defaults
            assert settings.report_style == "business"  # from env
            assert settings.llm.temperature == 0.9  # from env

    def test_convenience_functions(self):
        """Test the get_settings convenience function."""
        env_vars = {
            "DEER_REPORT_STYLE": "casual",
            "DEER_LLM__TIMEOUT": "45",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = get_settings()

            assert settings.report_style == "casual"
            assert settings.llm.timeout == 45

    def test_individual_settings_classes(self):
        """Test that nested settings work through the main AppSettings."""
        env_vars = {
            "DEER_LLM__TEMPERATURE": "0.6",
            "DEER_LLM__MAX_TOKENS": "1500",
            "DEER_LLM__TIMEOUT": "25",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = AppSettings()

            assert settings.llm.temperature == 0.6
            assert settings.llm.max_tokens == 1500
            assert settings.llm.timeout == 25

    def test_validation_still_works(self):
        """Test that Pydantic validation still works with the new system."""
        env_vars = {
            "DEER_LLM__TEMPERATURE": "3.0",  # Invalid: > 2.0
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError):
                AppSettings()

    def test_case_insensitive_env_vars(self):
        """Test that environment variables are case insensitive."""
        env_vars = {
            "deer_report_style": "technical",  # lowercase
            "DEER_LLM__TEMPERATURE": "0.4",  # uppercase
            "Deer_Agents__Max_Plan_Iterations": "7",  # mixed case
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = AppSettings()

            assert settings.report_style == "technical"
            assert settings.llm.temperature == 0.4
            assert settings.agents.max_plan_iterations == 7
