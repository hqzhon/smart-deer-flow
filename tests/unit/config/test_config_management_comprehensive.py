# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for Configuration Management System
Tests ConfigLoader, ConfigurationService, validators, and cache components
"""

import os
import pytest
import tempfile
import yaml
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.config.config_loader import ConfigLoader, get_settings
from src.config.models import AppSettings, LLMSettings, AgentSettings
from src.config.validators import (
    validate_configuration, validate_llm_config,
    validate_agent_config, validate_research_config, validate_file_paths,
    validate_environment_variables, get_all_validators
)


class TestConfigLoader:
    """Test cases for ConfigLoader class"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration data"""
        return {
            'llm': {
                'basic_model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 4000
            },
            'database': {
                'type': 'sqlite',
                'path': 'test.db'
            },
            'agents': {
                'max_iterations': 10,
                'timeout': 300
            }
        }
    
    def test_config_loader_initialization(self, temp_config_dir):
        """Test ConfigLoader initialization"""
        loader = ConfigLoader(str(temp_config_dir))
        assert loader.config_dir == temp_config_dir
        assert loader._settings is None
    
    def test_load_from_yaml_success(self, temp_config_dir, sample_yaml_config):
        """Test successful YAML loading"""
        yaml_file = temp_config_dir / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load_from_yaml("test_config.yaml")
        
        assert config == sample_yaml_config
        assert config['llm']['basic_model'] == 'gpt-4'
    
    def test_load_from_yaml_file_not_found(self, temp_config_dir):
        """Test YAML loading with missing file"""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load_from_yaml("nonexistent.yaml")
        
        assert config == {}
    
    def test_load_from_yaml_invalid_yaml(self, temp_config_dir):
        """Test YAML loading with invalid YAML content"""
        yaml_file = temp_config_dir / "invalid.yaml"
        with open(yaml_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load_from_yaml("invalid.yaml")
        
        assert config == {}
    
    def test_load_from_env(self):
        """Test environment variable loading"""
        env_vars = {
            'DEER_LLM_TEMPERATURE': '0.8',
            'DEER_MAX_PLAN_ITERATIONS': '15',
            'DEER_ENABLE_DEEP_THINKING': 'true',
            'DEER_ENABLE_RESEARCHER_ISOLATION': 'false'
        }
        
        with patch.dict(os.environ, env_vars):
            loader = ConfigLoader()
            config = loader.load_from_env()
            
            assert config['llm']['temperature'] == 0.8
            assert config['agents']['max_plan_iterations'] == 15
            assert config['agents']['enable_deep_thinking'] is True
            assert config['research']['enable_researcher_isolation'] is False
    
    def test_merge_configs(self):
        """Test configuration merging"""
        config1 = {
            'llm': {'basic_model': 'gpt-4', 'temperature': 0.7},
            'agents': {'max_iterations': 10}
        }
        config2 = {
            'llm': {'temperature': 0.8, 'max_tokens': 4000},
            'database': {'type': 'sqlite'}
        }
        
        loader = ConfigLoader()
        merged = loader.merge_configs(config1, config2)
        
        assert merged['llm']['basic_model'] == 'gpt-4'
        assert merged['llm']['temperature'] == 0.8  # config2 overrides
        assert merged['llm']['max_tokens'] == 4000
        assert merged['agents']['max_iterations'] == 10
        assert merged['database']['type'] == 'sqlite'
    
    def test_load_configuration_success(self, temp_config_dir, sample_yaml_config):
        """Test complete configuration loading"""
        yaml_file = temp_config_dir / "conf.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        
        loader = ConfigLoader(str(temp_config_dir))
        settings = loader.load_configuration()
        
        assert isinstance(settings, AppSettings)
        assert settings.llm.basic_model == 'gpt-4'
        assert settings.llm.temperature == 0.7
    
    def test_load_configuration_with_env_override(self, temp_config_dir, sample_yaml_config):
        """Test configuration loading with environment variable override"""
        yaml_file = temp_config_dir / "conf.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        
        with patch.dict(os.environ, {'DEER_LLM_TEMPERATURE': '0.9'}):
            loader = ConfigLoader(str(temp_config_dir))
            settings = loader.load_configuration()
            
            assert settings.llm.temperature == 0.9  # Environment override
    
    def test_get_settings_without_loading(self):
        """Test get_settings automatically loads configuration"""
        loader = ConfigLoader()
        with patch.object(loader, 'load_configuration') as mock_load:
            mock_settings = Mock(spec=AppSettings)
            mock_load.return_value = mock_settings
            
            settings = loader.get_settings()
            
            mock_load.assert_called_once()
            assert settings == mock_settings
    
    def test_reload_configuration(self, temp_config_dir, sample_yaml_config):
        """Test configuration reloading"""
        yaml_file = temp_config_dir / "conf.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        
        loader = ConfigLoader(str(temp_config_dir))
        
        # Load initial configuration
        settings1 = loader.load_configuration()
        
        # Modify configuration
        sample_yaml_config['llm']['temperature'] = 0.9
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        
        # Reload configuration
        settings2 = loader.reload_configuration()
        
        assert settings1.llm.temperature == 0.7
        assert settings2.llm.temperature == 0.9


class TestGetSettings:
    """Test cases for get_settings function"""
    
    def test_get_settings_returns_app_settings(self):
        """Test that get_settings returns AppSettings instance"""
        settings = get_settings()
        assert isinstance(settings, AppSettings)
    
    def test_get_settings_with_env_override(self):
        """Test get_settings with environment variable override"""
        # Clear cache first
        from src.config.config_loader import get_config_loader
        loader = get_config_loader()
        loader._settings = None
        
        env_vars = {
            'DEER_LLM__TEMPERATURE': '0.9',
            'DEER_AGENTS__MAX_PLAN_ITERATIONS': '5'
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = get_settings()
            assert settings.llm.temperature == 0.9
            assert settings.agents.max_plan_iterations == 5


class TestConfigValidators:
    """Test cases for configuration validators"""
    
    def test_validate_llm_config_success(self):
        """Test successful LLM configuration validation"""
        config = {
            'llm': {
                'basic_model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 4000
            }
        }
        
        assert validate_llm_config(config) is True
    
    def test_validate_llm_config_missing_required_field(self):
        """Test LLM validation with missing required field"""
        config = {
            'llm': {
                'temperature': 0.7
                # Missing 'basic_model'
            }
        }
        
        assert validate_llm_config(config) is False
    
    def test_validate_llm_config_invalid_model(self):
        """Test LLM validation with invalid model"""
        config = {
            'llm': {
                'basic_model': '',
                'temperature': 0.7
            }
        }
        
        assert validate_llm_config(config) is False
    
    def test_validate_llm_config_invalid_temperature(self):
        """Test LLM validation with invalid temperature"""
        config = {
            'llm': {
                'basic_model': 'gpt-4',
                'temperature': 3.0,  # Invalid: > 2
            }
        }
        
        assert validate_llm_config(config) is False
    
    def test_validate_configuration_basic(self):
        """Test basic configuration validation"""
        config = {
            'llm': {
                'basic_model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 4000
            }
        }
        
        assert validate_configuration(config) is True
    
    def test_validate_agent_config_success(self):
        """Test successful agent configuration validation"""
        config = {
            'agents': {
                'max_iterations': 10,
                'timeout': 300
            }
        }
        
        assert validate_agent_config(config) is True
    
    def test_validate_agent_config_invalid_values(self):
        """Test agent validation with invalid values"""
        config = {
            'agents': {
                'max_iterations': -1,  # Invalid: negative
                'timeout': 0  # Invalid: zero
            }
        }
        
        assert validate_agent_config(config) is False
    
    def test_validate_environment_variables(self):
        """Test environment variable validation"""
        config = {
            'llm': {
                'basic_model': 'gpt-4'
            }
        }
        
        assert validate_environment_variables(config) is True
    
    def test_validate_environment_variables_missing_key(self):
        """Test environment validation with missing API key"""
        config = {
            'llm': {
                'basic_model': 'gpt-4'
            }
        }
        
        with patch.dict(os.environ, {}, clear=True):
            assert validate_environment_variables(config) is True
    
    def test_validate_configuration_all_validators(self):
        """Test complete configuration validation"""
        config = {
            'llm': {
                'basic_model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 4000
            },
            'database': {
                'type': 'sqlite',
                'path': 'test.db'
            },
            'agents': {
                'max_iterations': 10,
                'timeout': 300
            },
            'research': {
                'max_sources': 10,
                'search_depth': 3
            }
        }
        
        with tempfile.TemporaryDirectory():
            # Mock file path validation to pass
            with patch('src.config.validators.validate_file_paths', return_value=True):
                assert validate_configuration(config) is True
    
    def test_get_all_validators(self):
        """Test getting all validators"""
        validators = get_all_validators()
        
        assert len(validators) > 0
        assert all(callable(v) for v in validators)





class TestConfigIntegration:
    """Integration tests for configuration management system"""
    
    def test_end_to_end_configuration_flow(self, tmp_path):
        """Test complete configuration flow from loading"""
        # Create test configuration file
        config_data = {
            'llm': {
                'basic_model': 'gpt-4',
                'temperature': 0.7
            }
        }
        
        config_file = tmp_path / 'conf.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test configuration loading
        loader = ConfigLoader(str(tmp_path))
        settings = loader.load_configuration('conf.yaml')
        
        assert isinstance(settings, AppSettings)
        assert settings.llm.basic_model == 'gpt-4'
        
        # Test get_settings function
        settings = get_settings()
        assert isinstance(settings, AppSettings)
    
    def test_configuration_error_handling(self, tmp_path):
        """Test configuration error handling scenarios"""
        # Test with invalid YAML
        invalid_config_file = tmp_path / 'invalid.yaml'
        with open(invalid_config_file, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        loader = ConfigLoader(str(tmp_path))
        config = loader.load_from_yaml('invalid.yaml')
        assert config == {}  # Should return empty dict on error
        
        # Test get_settings function
        settings = get_settings()
        assert isinstance(settings, AppSettings)
    
    def test_concurrent_configuration_access(self, tmp_path):
        """Test concurrent access to configuration system"""
        results = []
        errors = []
        
        def worker():
            try:
                settings = get_settings()
                results.append(settings.llm.basic_model if settings.llm.basic_model else 'default')
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0
        assert len(results) == 10