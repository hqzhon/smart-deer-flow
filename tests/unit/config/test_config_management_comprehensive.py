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

from src.config.config_loader import ConfigLoader, get_config_loader, load_configuration
from src.config.config_service import ConfigurationService, get_configuration_service
from src.config.models import AppSettings, LLMSettings, DatabaseSettings, AgentSettings
from src.config.validators import (
    validate_configuration, validate_llm_config, validate_database_config,
    validate_agent_config, validate_research_config, validate_file_paths,
    validate_environment_variables, get_all_validators
)
from src.config.cache import ConfigCache, get_config_cache, cached_config, invalidate_config_cache


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


class TestConfigurationService:
    """Test cases for ConfigurationService class"""
    
    @pytest.fixture
    def mock_config_loader(self):
        """Mock ConfigLoader for testing"""
        loader = Mock(spec=ConfigLoader)
        mock_settings = AppSettings(
            llm=LLMSettings(basic_model='gpt-4'),
            database=DatabaseSettings(type='sqlite'),
            agents=AgentSettings()
        )
        loader.load_configuration.return_value = mock_settings
        return loader
    
    def test_service_initialization(self, mock_config_loader):
        """Test ConfigurationService initialization"""
        service = ConfigurationService(mock_config_loader)
        assert service._config_loader == mock_config_loader
        assert service._app_settings is None
    
    def test_get_app_settings_success(self, mock_config_loader):
        """Test successful app settings retrieval"""
        service = ConfigurationService(mock_config_loader)
        settings = service.get_app_settings()
        
        assert isinstance(settings, AppSettings)
        assert settings.llm.basic_model == 'gpt-4'
        mock_config_loader.load_configuration.assert_called_once()
    
    def test_get_app_settings_with_error(self, mock_config_loader):
        """Test app settings retrieval with loader error"""
        mock_config_loader.load_configuration.side_effect = Exception("Load error")
        
        service = ConfigurationService(mock_config_loader)
        settings = service.get_app_settings()
        
        # Should return default settings on error
        assert isinstance(settings, AppSettings)
    
    def test_get_section_success(self, mock_config_loader):
        """Test successful configuration section retrieval"""
        service = ConfigurationService(mock_config_loader)
        llm_settings = service.get_section('llm', LLMSettings)
        
        assert isinstance(llm_settings, LLMSettings)
        assert llm_settings.basic_model == 'gpt-4'
    
    def test_get_section_not_found(self, mock_config_loader):
        """Test configuration section not found"""
        service = ConfigurationService(mock_config_loader)
        
        with pytest.raises(ValueError, match="Configuration section 'nonexistent' not found"):
            service.get_section('nonexistent', LLMSettings)
    
    def test_get_section_wrong_type(self, mock_config_loader):
        """Test configuration section with wrong type"""
        service = ConfigurationService(mock_config_loader)
        
        with pytest.raises(TypeError, match="Configuration section 'llm' is not of type DatabaseSettings"):
            service.get_section('llm', DatabaseSettings)
    
    def test_reload_configuration(self, mock_config_loader):
        """Test configuration reloading"""
        service = ConfigurationService(mock_config_loader)
        
        # Load initial settings
        service.get_app_settings()
        
        # Reload
        reloaded_settings = service.reload_configuration()
        
        assert isinstance(reloaded_settings, AppSettings)
        assert mock_config_loader.load_configuration.call_count == 2
    
    def test_get_nested_value(self, mock_config_loader):
        """Test nested value retrieval"""
        service = ConfigurationService(mock_config_loader)
        
        # Test existing nested value
        value = service.get('llm.basic_model')
        assert value == 'gpt-4'
        
        # Test non-existing value with default
        value = service.get('nonexistent.key', 'default')
        assert value == 'default'
    
    def test_has_key(self, mock_config_loader):
        """Test key existence check"""
        service = ConfigurationService(mock_config_loader)
        
        assert service.has('llm.basic_model') is True
        assert service.has('nonexistent.key') is False
    
    def test_get_all(self, mock_config_loader):
        """Test getting all configuration as dictionary"""
        service = ConfigurationService(mock_config_loader)
        all_config = service.get_all()
        
        assert isinstance(all_config, dict)
        assert 'llm' in all_config
        assert 'database' in all_config


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
    
    def test_validate_database_config_sqlite(self):
        """Test database validation for SQLite"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'database': {
                    'type': 'sqlite',
                    'path': os.path.join(temp_dir, 'test.db')
                }
            }
            
            assert validate_database_config(config) is True
    
    def test_validate_database_config_postgresql(self):
        """Test database validation for PostgreSQL"""
        config = {
            'database': {
                'type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'database': 'testdb',
                'username': 'testuser'
            }
        }
        
        assert validate_database_config(config) is True
    
    def test_validate_database_config_missing_fields(self):
        """Test database validation with missing fields"""
        config = {
            'database': {
                'type': 'postgresql',
                'host': 'localhost'
                # Missing port, database, username
            }
        }
        
        assert validate_database_config(config) is False
    
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


class TestConfigCache:
    """Test cases for configuration cache"""
    
    @pytest.fixture
    def cache(self):
        """Create fresh cache instance for testing"""
        return ConfigCache(default_ttl=1.0)  # 1 second TTL for testing
    
    def test_cache_set_and_get(self, cache):
        """Test basic cache set and get operations"""
        cache.set('test_key', 'test_value')
        value = cache.get('test_key')
        
        assert value == 'test_value'
    
    def test_cache_get_nonexistent(self, cache):
        """Test getting non-existent cache entry"""
        value = cache.get('nonexistent_key')
        assert value is None
    
    def test_cache_ttl_expiration(self, cache):
        """Test cache TTL expiration"""
        cache.set('test_key', 'test_value', ttl=0.1)  # 0.1 second TTL
        
        # Should be available immediately
        assert cache.get('test_key') == 'test_value'
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get('test_key') is None
    
    def test_cache_delete(self, cache):
        """Test cache entry deletion"""
        cache.set('test_key', 'test_value')
        
        assert cache.delete('test_key') is True
        assert cache.get('test_key') is None
        assert cache.delete('test_key') is False  # Already deleted
    
    def test_cache_clear(self, cache):
        """Test cache clearing"""
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        cache.clear()
        
        assert cache.get('key1') is None
        assert cache.get('key2') is None
    
    def test_cache_invalidate_pattern(self, cache):
        """Test pattern-based cache invalidation"""
        cache.set('config.llm.provider', 'openai')
        cache.set('config.llm.basic_model', 'gpt-4')
        cache.set('config.database.type', 'sqlite')
        cache.set('other.key', 'value')
        
        count = cache.invalidate_pattern('config.llm')
        
        assert count == 2
        assert cache.get('config.llm.provider') is None
        assert cache.get('config.llm.basic_model') is None
        assert cache.get('config.database.type') == 'sqlite'  # Not invalidated
        assert cache.get('other.key') == 'value'  # Not invalidated
    
    def test_cache_dependencies(self, cache):
        """Test cache dependency management"""
        cache.set('parent_key', 'parent_value')
        cache.set('child_key', 'child_value')
        cache.add_dependency('parent_key', 'child_key')
        
        count = cache.invalidate_dependents('parent_key')
        
        assert count == 1
        assert cache.get('parent_key') == 'parent_value'  # Parent still exists
        assert cache.get('child_key') is None  # Child invalidated
    
    def test_cache_cleanup_expired(self, cache):
        """Test cleanup of expired entries"""
        cache.set('key1', 'value1', ttl=0.1)
        cache.set('key2', 'value2', ttl=10.0)
        
        time.sleep(0.2)  # Wait for first entry to expire
        
        count = cache.cleanup_expired()
        
        assert count == 1
        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
    
    def test_cache_stats(self, cache):
        """Test cache statistics"""
        cache.set('key1', 'value1', ttl=0.1)
        cache.set('key2', 'value2', ttl=10.0)
        cache.add_dependency('key1', 'key2')
        
        time.sleep(0.2)  # Let first entry expire
        
        stats = cache.get_stats()
        
        assert stats['total_entries'] == 2
        assert stats['expired_entries'] == 1
        assert stats['active_entries'] == 1
        assert stats['dependencies'] == 1
    
    def test_cache_thread_safety(self, cache):
        """Test cache thread safety"""
        def worker(thread_id):
            for i in range(100):
                key = f'thread_{thread_id}_key_{i}'
                cache.set(key, f'value_{i}')
                value = cache.get(key)
                assert value == f'value_{i}'
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify cache integrity
        stats = cache.get_stats()
        assert stats['total_entries'] == 500  # 5 threads * 100 entries each
    
    def test_cached_config_decorator(self):
        """Test cached_config decorator"""
        call_count = 0
        
        @cached_config('test_function', ttl=1.0)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again
        
        # Wait for cache to expire
        time.sleep(1.1)
        
        # Third call should execute function again
        result3 = expensive_function(5)
        assert result3 == 10
        assert call_count == 2
    
    def test_invalidate_config_cache_functions(self):
        """Test global cache invalidation functions"""
        cache = get_config_cache()
        # Clear cache first to ensure clean state
        cache.clear()
        
        cache.set('test.key1', 'value1')
        cache.set('test.key2', 'value2')
        cache.set('other.key', 'value3')
        
        # Test pattern invalidation
        count = invalidate_config_cache(pattern='test')
        assert count == 2
        assert cache.get('test.key1') is None
        assert cache.get('test.key2') is None
        assert cache.get('other.key') == 'value3'
        
        # Test key invalidation
        count = invalidate_config_cache(key='other.key')
        assert count == 1
        assert cache.get('other.key') is None


class TestConfigIntegration:
    """Integration tests for configuration management system"""
    
    def test_end_to_end_configuration_flow(self, tmp_path):
        """Test complete configuration flow from loading to caching"""
        # Create test configuration file
        config_data = {
            'llm': {
                'basic_model': 'gpt-4',
                'temperature': 0.7
            },
            'database': {
                'type': 'sqlite',
                'path': str(tmp_path / 'test.db')
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
        
        # Test configuration service
        service = ConfigurationService(loader)
        llm_settings = service.get_section('llm', LLMSettings)
        
        assert llm_settings.basic_model == 'gpt-4'
        assert llm_settings.temperature == 0.7
        
        # Test configuration validation
        assert service.validate_current_configuration() is True
        
        # Test configuration caching
        cache = get_config_cache()
        cache.set('config.loaded', True)
        
        assert cache.get('config.loaded') is True
    
    def test_configuration_error_handling(self, tmp_path):
        """Test configuration error handling scenarios"""
        # Test with invalid YAML
        invalid_config_file = tmp_path / 'invalid.yaml'
        with open(invalid_config_file, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        loader = ConfigLoader(str(tmp_path))
        config = loader.load_from_yaml('invalid.yaml')
        assert config == {}  # Should return empty dict on error
        
        # Test service with loader error
        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_configuration.side_effect = Exception("Load error")
        
        service = ConfigurationService(mock_loader)
        settings = service.get_app_settings()
        
        # Should return default settings on error
        assert isinstance(settings, AppSettings)
    
    def test_concurrent_configuration_access(self, tmp_path):
        """Test concurrent access to configuration system"""
        config_data = {
            'llm': {'basic_model': 'gpt-4'},
            'database': {'type': 'sqlite', 'path': str(tmp_path / 'test.db')}
        }
        
        config_file = tmp_path / 'conf.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(tmp_path))
        service = ConfigurationService(loader)
        
        results = []
        errors = []
        
        def worker():
            try:
                settings = service.get_app_settings()
                results.append(settings.llm.basic_model)
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
        assert all(model == 'gpt-4' for model in results)