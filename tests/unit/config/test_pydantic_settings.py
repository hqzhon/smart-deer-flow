"""Tests for the new pydantic-settings based configuration system."""

import os
import pytest
from unittest.mock import patch

from src.config.settings import AppSettings, LLMSettings, AgentSettings
from src.config.config_loader_v2 import ConfigLoaderV2, load_configuration_v2


class TestPydanticSettingsConfig:
    """Test the new pydantic-settings based configuration system."""
    
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
        env_vars = {
            'DEER_REPORT_STYLE': 'business',
            'DEER_LLM_TEMPERATURE': '0.9',
            'DEER_LLM_MAX_TOKENS': '2000',
            'DEER_MAX_PLAN_ITERATIONS': '5',
            'DEER_ENABLE_DEEP_THINKING': 'true',
            'DEER_ENABLE_RESEARCHER_ISOLATION': 'false',
            'DEER_RESEARCHER_ISOLATION_LEVEL': 'aggressive',
            'DEER_MAX_REFLECTION_LOOPS': '5',
            'DEER_MCP_ENABLED': 'true',
            'DEER_MCP_TIMEOUT': '60'
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
        """Test that nested environment variables work with double underscore delimiter."""
        env_vars = {
            'DEER_LLM__TEMPERATURE': '0.8',
            'DEER_AGENTS__MAX_PLAN_ITERATIONS': '3',
            'DEER_RESEARCH__RESEARCHER_ISOLATION_LEVEL': 'minimal',
            'DEER_ADVANCED_CONTEXT__MAX_CONTEXT_RATIO': '0.8',
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = AppSettings()
            
            assert settings.llm.temperature == 0.8
            assert settings.agents.max_plan_iterations == 3
            assert settings.research.researcher_isolation_level == "minimal"
            assert settings.advanced_context.max_context_ratio == 0.8
    
    def test_type_conversion(self):
        """Test that environment variables are properly converted to correct types."""
        env_vars = {
            'DEER_LLM_TEMPERATURE': '0.5',  # float
            'DEER_MAX_PLAN_ITERATIONS': '10',  # int
            'DEER_ENABLE_DEEP_THINKING': 'true',  # bool
            'DEER_ENABLE_PARALLEL_EXECUTION': 'false',  # bool
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
    
    def test_config_loader_v2(self):
        """Test the new ConfigLoaderV2 class."""
        loader = ConfigLoaderV2()
        
        # Test loading with environment variables
        env_vars = {
            'DEER_REPORT_STYLE': 'technical',
            'DEER_LLM_TEMPERATURE': '0.3',
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = loader.load_configuration()
            
            assert settings.report_style == "technical"
            assert settings.llm.temperature == 0.3
    
    def test_yaml_override_with_env_vars(self, tmp_path):
        """Test that YAML config works with environment variable overrides."""
        # Create a temporary YAML config file
        yaml_content = """
report_style: academic
llm:
  temperature: 0.7
  max_tokens: 1000
agents:
  max_plan_iterations: 2
  enable_deep_thinking: false
"""
        
        yaml_file = tmp_path / "test_conf.yaml"
        yaml_file.write_text(yaml_content)
        
        # Set environment variables that should override YAML
        env_vars = {
            'DEER_REPORT_STYLE': 'business',
            'DEER_LLM_TEMPERATURE': '0.9',
        }
        
        loader = ConfigLoaderV2(config_dir=str(tmp_path))
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = loader.load_configuration(yaml_path="test_conf.yaml")
            
            # Environment variables should override YAML
            assert settings.report_style == "business"  # from env
            assert settings.llm.temperature == 0.9  # from env
            
            # YAML values should be used where no env override exists
            assert settings.llm.max_tokens == 1000  # from YAML
            assert settings.agents.max_plan_iterations == 2  # from YAML
            assert settings.agents.enable_deep_thinking is False  # from YAML
    
    def test_convenience_functions(self):
        """Test the convenience functions for loading configuration."""
        env_vars = {
            'DEER_REPORT_STYLE': 'casual',
            'DEER_LLM_TIMEOUT': '45',
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = load_configuration_v2()
            
            assert settings.report_style == "casual"
            assert settings.llm.timeout == 45
    
    def test_individual_settings_classes(self):
        """Test that individual settings classes work with their own env prefixes."""
        env_vars = {
            'DEER_LLM_TEMPERATURE': '0.6',
            'DEER_LLM_MAX_TOKENS': '1500',
            'DEER_LLM_TIMEOUT': '25',
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            llm_settings = LLMSettings()
            
            assert llm_settings.temperature == 0.6
            assert llm_settings.max_tokens == 1500
            assert llm_settings.timeout == 25
    
    def test_validation_still_works(self):
        """Test that Pydantic validation still works with the new system."""
        env_vars = {
            'DEER_LLM_TEMPERATURE': '3.0',  # Invalid: > 2.0
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError):
                AppSettings()
    
    def test_case_insensitive_env_vars(self):
        """Test that environment variables are case insensitive."""
        env_vars = {
            'deer_report_style': 'technical',  # lowercase
            'DEER_LLM_TEMPERATURE': '0.4',  # uppercase
            'Deer_Max_Plan_Iterations': '7',  # mixed case
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = AppSettings()
            
            assert settings.report_style == "technical"
            assert settings.llm.temperature == 0.4
            assert settings.agents.max_plan_iterations == 7