#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Comprehensive Unit Tests for Configuration System

This test suite provides comprehensive coverage for the configuration system
integration in DeerFlow, replacing configuration-related demo scripts with proper unit tests.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

# Configuration system components
from src.config.config_manager import ConfigManager
from src.config.config_integration import ConfigurationIntegrator
from src.config.researcher_config_loader import (
    ResearcherConfigLoader, EnhancedReflectionConfig, IsolationConfig
)
from src.config.configuration import Configuration

# Test utilities
from tests.conftest import create_test_config


class TestConfigurationManager:
    """Test ConfigManager functionality."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_config_data(self):
        """Create sample configuration data."""
        return {
            "main": {
                "project_name": "DeerFlow",
                "version": "1.0.0",
                "debug_mode": False,
                "log_level": "INFO"
            },
            "researcher": {
                "enhanced_reflection": {
                    "enable_enhanced_reflection": True,
                    "max_reflection_loops": 3,
                    "reflection_model": "deepseek-chat",
                    "reflection_temperature": 0.7,
                    "knowledge_gap_threshold": 0.6,
                    "sufficiency_threshold": 0.8,
                    "cache_settings": {
                        "enable_cache": True,
                        "cache_ttl": 3600,
                        "max_cache_size": 1000
                    },
                    "trigger_settings": {
                        "trigger_on_low_confidence": True,
                        "trigger_on_knowledge_gaps": True,
                        "min_steps_before_trigger": 2,
                        "adaptive_triggering": True
                    }
                },
                "isolation": {
                    "enable_isolation": True,
                    "isolation_level": "moderate",
                    "max_context_steps": 5,
                    "context_window_size": 4000,
                    "enable_progressive": True
                }
            },
            "integration": {
                "enable_reflection_integration": True,
                "reflection_trigger_threshold": 2,
                "max_reflection_loops": 3,
                "adaptive_reflection": True,
                "performance_monitoring": {
                    "enable_metrics": True,
                    "metrics_collection_interval": 60,
                    "enable_performance_tracking": True
                }
            }
        }
    
    @pytest.fixture
    def config_files(self, temp_config_dir, sample_config_data):
        """Create configuration files in temporary directory."""
        config_files = {}
        
        # Main config file
        main_config_file = os.path.join(temp_config_dir, "config.json")
        with open(main_config_file, 'w') as f:
            json.dump(sample_config_data["main"], f, indent=2)
        config_files["main"] = main_config_file
        
        # Researcher config file
        researcher_config_file = os.path.join(temp_config_dir, "researcher_config.json")
        with open(researcher_config_file, 'w') as f:
            json.dump(sample_config_data["researcher"], f, indent=2)
        config_files["researcher"] = researcher_config_file
        
        # Integration config file
        integration_config_file = os.path.join(temp_config_dir, "integration_config.json")
        with open(integration_config_file, 'w') as f:
            json.dump(sample_config_data["integration"], f, indent=2)
        config_files["integration"] = integration_config_file
        
        return config_files
    
    def test_config_manager_initialization(self, config_files):
        """Test ConfigManager initialization."""
        with patch('src.config.config_manager.ConfigManager._load_config_files') as mock_load:
            mock_load.return_value = None
            
            manager = ConfigManager(config_dir=os.path.dirname(config_files["main"]))
            
            assert hasattr(manager, 'config_dir')
            assert hasattr(manager, '_configs')
            assert hasattr(manager, '_config_cache')
    
    def test_main_config_loading(self, config_files, sample_config_data):
        """Test main configuration loading."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["main"]
            
            manager = ConfigManager()
            config = manager.get_main_config()
            
            assert config is not None
            assert config["project_name"] == "DeerFlow"
            assert config["version"] == "1.0.0"
            assert config["debug_mode"] is False
    
    def test_researcher_config_loading(self, config_files, sample_config_data):
        """Test researcher configuration loading."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["researcher"]
            
            manager = ConfigManager()
            config = manager.get_researcher_config()
            
            assert config is not None
            assert "enhanced_reflection" in config
            assert "isolation" in config
            
            # Test enhanced reflection config
            reflection_config = config["enhanced_reflection"]
            assert reflection_config["enable_enhanced_reflection"] is True
            assert reflection_config["max_reflection_loops"] == 3
            assert reflection_config["knowledge_gap_threshold"] == 0.6
    
    def test_integration_config_loading(self, config_files, sample_config_data):
        """Test integration configuration loading."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["integration"]
            
            manager = ConfigManager()
            config = manager.get_integration_config()
            
            assert config is not None
            assert config["enable_reflection_integration"] is True
            assert config["reflection_trigger_threshold"] == 2
            assert config["adaptive_reflection"] is True
    
    def test_config_validation(self, config_files, sample_config_data):
        """Test configuration validation."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["researcher"]
            
            manager = ConfigManager()
            
            # Test valid configuration
            is_valid = manager.validate_config("researcher")
            assert is_valid is True
            
            # Test invalid configuration
            invalid_config = sample_config_data["researcher"].copy()
            invalid_config["enhanced_reflection"]["max_reflection_loops"] = -1
            
            with patch.object(manager, '_load_config_file', return_value=invalid_config):
                is_valid = manager.validate_config("researcher")
                assert is_valid is False
    
    def test_config_caching(self, config_files, sample_config_data):
        """Test configuration caching mechanism."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["main"]
            
            manager = ConfigManager()
            
            # First call should load from file
            config1 = manager.get_main_config()
            assert mock_load.call_count == 1
            
            # Second call should use cache
            config2 = manager.get_main_config()
            assert mock_load.call_count == 1  # No additional calls
            
            # Configs should be identical
            assert config1 == config2
    
    def test_config_reloading(self, config_files, sample_config_data):
        """Test configuration reloading."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["main"]
            
            manager = ConfigManager()
            
            # Load initial config
            config1 = manager.get_main_config()
            
            # Reload config
            manager.reload_config("main")
            config2 = manager.get_main_config()
            
            # Should have called load twice
            assert mock_load.call_count == 2
    
    def test_config_modification(self, config_files, sample_config_data):
        """Test configuration modification."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["main"].copy()
            
            manager = ConfigManager()
            
            # Modify configuration
            manager.update_config("main", "debug_mode", True)
            
            config = manager.get_main_config()
            assert config["debug_mode"] is True
    
    def test_config_export(self, config_files, sample_config_data):
        """Test configuration export functionality."""
        with patch('src.config.config_manager.ConfigManager._load_config_file') as mock_load:
            mock_load.return_value = sample_config_data["main"]
            
            manager = ConfigManager()
            
            # Export configuration
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                export_file = f.name
            
            try:
                manager.export_config("main", export_file)
                
                # Verify exported file
                with open(export_file, 'r') as f:
                    exported_config = json.load(f)
                
                assert exported_config == sample_config_data["main"]
            finally:
                os.unlink(export_file)


class TestResearcherConfigLoader:
    """Test ResearcherConfigLoader functionality."""
    
    @pytest.fixture
    def sample_researcher_config(self):
        """Create sample researcher configuration."""
        return {
            "enhanced_reflection": {
                "enable_enhanced_reflection": True,
                "max_reflection_loops": 3,
                "reflection_model": "deepseek-chat",
                "reflection_temperature": 0.7,
                "knowledge_gap_threshold": 0.6,
                "sufficiency_threshold": 0.8,
                "cache_settings": {
                    "enable_cache": True,
                    "cache_ttl": 3600,
                    "max_cache_size": 1000
                },
                "trigger_settings": {
                    "trigger_on_low_confidence": True,
                    "trigger_on_knowledge_gaps": True,
                    "min_steps_before_trigger": 2,
                    "adaptive_triggering": True
                }
            },
            "isolation": {
                "enable_isolation": True,
                "isolation_level": "moderate",
                "max_context_steps": 5,
                "context_window_size": 4000,
                "enable_progressive": True
            }
        }
    
    @pytest.fixture
    def temp_researcher_config_file(self, sample_researcher_config):
        """Create temporary researcher configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_researcher_config, f, indent=2)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_researcher_config_loader_initialization(self, temp_researcher_config_file):
        """Test ResearcherConfigLoader initialization."""
        loader = ResearcherConfigLoader(config_file=temp_researcher_config_file)
        
        assert loader.config_file == temp_researcher_config_file
        assert hasattr(loader, '_config_cache')
        assert hasattr(loader, '_validation_cache')
    
    def test_config_loading(self, temp_researcher_config_file, sample_researcher_config):
        """Test configuration loading."""
        loader = ResearcherConfigLoader(config_file=temp_researcher_config_file)
        
        config = loader.load_config()
        
        assert config is not None
        assert hasattr(config, 'enhanced_reflection')
        assert hasattr(config, 'isolation')
        
        # Test enhanced reflection config
        reflection_config = config.enhanced_reflection
        assert isinstance(reflection_config, EnhancedReflectionConfig)
        assert reflection_config.enable_enhanced_reflection is True
        assert reflection_config.max_reflection_loops == 3
        assert reflection_config.reflection_model == "deepseek-chat"
        
        # Test isolation config
        isolation_config = config.isolation
        assert isinstance(isolation_config, IsolationConfig)
        assert isolation_config.enable_isolation is True
        assert isolation_config.isolation_level == "moderate"
    
    def test_config_validation(self, temp_researcher_config_file):
        """Test configuration validation."""
        loader = ResearcherConfigLoader(config_file=temp_researcher_config_file)
        
        config = loader.load_config()
        is_valid = loader.validate_config(config)
        
        assert is_valid is True
    
    def test_invalid_config_validation(self, sample_researcher_config):
        """Test validation of invalid configuration."""
        # Create invalid config
        invalid_config = sample_researcher_config.copy()
        invalid_config["enhanced_reflection"]["max_reflection_loops"] = -1
        invalid_config["enhanced_reflection"]["reflection_temperature"] = 2.0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f, indent=2)
            temp_file = f.name
        
        try:
            loader = ResearcherConfigLoader(config_file=temp_file)
            
            with pytest.raises((ValueError, AssertionError)):
                config = loader.load_config()
                loader.validate_config(config)
        finally:
            os.unlink(temp_file)
    
    def test_config_caching(self, temp_researcher_config_file):
        """Test configuration caching."""
        loader = ResearcherConfigLoader(config_file=temp_researcher_config_file)
        
        # First load
        config1 = loader.load_config()
        
        # Second load should use cache
        config2 = loader.load_config()
        
        # Should be the same object (cached)
        assert config1 is config2
    
    def test_config_reloading(self, temp_researcher_config_file):
        """Test configuration reloading."""
        loader = ResearcherConfigLoader(config_file=temp_researcher_config_file)
        
        # Load initial config
        config1 = loader.load_config()
        
        # Force reload
        config2 = loader.load_config(force_reload=True)
        
        # Should be different objects
        assert config1 is not config2
        
        # But should have same values
        assert config1.enhanced_reflection.enable_enhanced_reflection == config2.enhanced_reflection.enable_enhanced_reflection
    
    def test_enhanced_reflection_config_properties(self, temp_researcher_config_file):
        """Test EnhancedReflectionConfig properties."""
        loader = ResearcherConfigLoader(config_file=temp_researcher_config_file)
        config = loader.load_config()
        
        reflection_config = config.enhanced_reflection
        
        # Test all properties
        assert hasattr(reflection_config, 'enable_enhanced_reflection')
        assert hasattr(reflection_config, 'max_reflection_loops')
        assert hasattr(reflection_config, 'reflection_model')
        assert hasattr(reflection_config, 'reflection_temperature')
        assert hasattr(reflection_config, 'knowledge_gap_threshold')
        assert hasattr(reflection_config, 'sufficiency_threshold')
        assert hasattr(reflection_config, 'cache_settings')
        assert hasattr(reflection_config, 'trigger_settings')
        
        # Test cache settings
        cache_settings = reflection_config.cache_settings
        assert cache_settings["enable_cache"] is True
        assert cache_settings["cache_ttl"] == 3600
        
        # Test trigger settings
        trigger_settings = reflection_config.trigger_settings
        assert trigger_settings["trigger_on_low_confidence"] is True
        assert trigger_settings["min_steps_before_trigger"] == 2
    
    def test_isolation_config_properties(self, temp_researcher_config_file):
        """Test IsolationConfig properties."""
        loader = ResearcherConfigLoader(config_file=temp_researcher_config_file)
        config = loader.load_config()
        
        isolation_config = config.isolation
        
        # Test all properties
        assert hasattr(isolation_config, 'enable_isolation')
        assert hasattr(isolation_config, 'isolation_level')
        assert hasattr(isolation_config, 'max_context_steps')
        assert hasattr(isolation_config, 'context_window_size')
        assert hasattr(isolation_config, 'enable_progressive')
        
        # Test values
        assert isolation_config.enable_isolation is True
        assert isolation_config.isolation_level == "moderate"
        assert isolation_config.max_context_steps == 5
        assert isolation_config.context_window_size == 4000


class TestConfigurationIntegrator:
    """Test ConfigurationIntegrator functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create mock ConfigManager."""
        manager = Mock(spec=ConfigManager)
        manager.get_main_config.return_value = {
            "project_name": "DeerFlow",
            "version": "1.0.0",
            "debug_mode": False
        }
        manager.get_researcher_config.return_value = {
            "enhanced_reflection": {
                "enable_enhanced_reflection": True,
                "max_reflection_loops": 3
            },
            "isolation": {
                "enable_isolation": True,
                "isolation_level": "moderate"
            }
        }
        manager.get_integration_config.return_value = {
            "enable_reflection_integration": True,
            "reflection_trigger_threshold": 2
        }
        return manager
    
    def test_configuration_integrator_initialization(self, mock_config_manager):
        """Test ConfigurationIntegrator initialization."""
        with patch('src.config.config_integration.ConfigManager', return_value=mock_config_manager):
            integrator = ConfigurationIntegrator()
            
            assert hasattr(integrator, 'config_manager')
            assert hasattr(integrator, '_integrated_config_cache')
            assert hasattr(integrator, '_validation_results')
    
    def test_researcher_config_loading(self, mock_config_manager):
        """Test researcher configuration loading through integrator."""
        with patch('src.config.config_integration.ConfigManager', return_value=mock_config_manager):
            integrator = ConfigurationIntegrator()
            
            config = integrator.load_researcher_config()
            
            assert config is not None
            assert "enhanced_reflection" in config
            assert "isolation" in config
    
    def test_configuration_validation(self, mock_config_manager):
        """Test configuration validation through integrator."""
        with patch('src.config.config_integration.ConfigManager', return_value=mock_config_manager):
            integrator = ConfigurationIntegrator()
            
            # Test successful validation
            is_valid = integrator.validate_configuration()
            assert is_valid is True
            
            # Test validation with errors
            mock_config_manager.get_researcher_config.return_value = {
                "enhanced_reflection": {
                    "enable_enhanced_reflection": True,
                    "max_reflection_loops": -1  # Invalid
                }
            }
            
            is_valid = integrator.validate_configuration()
            assert is_valid is False
    
    def test_integrated_reflection_config(self, mock_config_manager):
        """Test integrated reflection configuration generation."""
        with patch('src.config.config_integration.ConfigManager', return_value=mock_config_manager):
            integrator = ConfigurationIntegrator()
            
            reflection_config = integrator.get_integrated_reflection_config()
            
            assert reflection_config is not None
            assert isinstance(reflection_config, dict)
            assert "enhanced_reflection" in reflection_config
            assert "integration_settings" in reflection_config
    
    def test_integrated_isolation_config(self, mock_config_manager):
        """Test integrated isolation configuration generation."""
        with patch('src.config.config_integration.ConfigManager', return_value=mock_config_manager):
            integrator = ConfigurationIntegrator()
            
            isolation_config = integrator.get_integrated_isolation_config()
            
            assert isolation_config is not None
            assert isinstance(isolation_config, dict)
            assert "isolation" in isolation_config
            assert "integration_settings" in isolation_config
    
    def test_configuration_summary(self, mock_config_manager):
        """Test configuration summary generation."""
        with patch('src.config.config_integration.ConfigManager', return_value=mock_config_manager):
            integrator = ConfigurationIntegrator()
            
            summary = integrator.get_configuration_summary()
            
            assert isinstance(summary, dict)
            assert "main_config" in summary
            assert "researcher_config" in summary
            assert "integration_config" in summary
            assert "validation_status" in summary
    
    def test_configuration_export(self, mock_config_manager):
        """Test configuration export functionality."""
        with patch('src.config.config_integration.ConfigManager', return_value=mock_config_manager):
            integrator = ConfigurationIntegrator()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                export_file = f.name
            
            try:
                integrator.export_integrated_config(export_file)
                
                # Verify exported file
                with open(export_file, 'r') as f:
                    exported_config = json.load(f)
                
                assert "main_config" in exported_config
                assert "researcher_config" in exported_config
                assert "integration_config" in exported_config
            finally:
                os.unlink(export_file)


class TestConfigurationEdgeCases:
    """Test configuration system edge cases and error handling."""
    
    def test_missing_config_file_handling(self):
        """Test handling of missing configuration files."""
        non_existent_file = "/path/to/non/existent/config.json"
        
        with pytest.raises(FileNotFoundError):
            ResearcherConfigLoader(config_file=non_existent_file)
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON configuration files."""
        malformed_json = '{"key": "value", "invalid": }'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(malformed_json)
            temp_file = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                ResearcherConfigLoader(config_file=temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_empty_config_file_handling(self):
        """Test handling of empty configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('')
            temp_file = f.name
        
        try:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                ResearcherConfigLoader(config_file=temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_partial_config_handling(self):
        """Test handling of partial configuration files."""
        partial_config = {
            "enhanced_reflection": {
                "enable_enhanced_reflection": True
                # Missing other required fields
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            temp_file = f.name
        
        try:
            loader = ResearcherConfigLoader(config_file=temp_file)
            config = loader.load_config()
            
            # Should handle missing fields with defaults
            assert config.enhanced_reflection.enable_enhanced_reflection is True
            assert hasattr(config.enhanced_reflection, 'max_reflection_loops')
        finally:
            os.unlink(temp_file)
    
    def test_config_type_validation(self):
        """Test configuration type validation."""
        invalid_types_config = {
            "enhanced_reflection": {
                "enable_enhanced_reflection": "true",  # Should be boolean
                "max_reflection_loops": "3",  # Should be integer
                "reflection_temperature": "0.7"  # Should be float
            },
            "isolation": {
                "enable_isolation": 1,  # Should be boolean
                "max_context_steps": "5"  # Should be integer
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_types_config, f)
            temp_file = f.name
        
        try:
            loader = ResearcherConfigLoader(config_file=temp_file)
            
            with pytest.raises((TypeError, ValueError)):
                config = loader.load_config()
                loader.validate_config(config)
        finally:
            os.unlink(temp_file)
    
    def test_config_range_validation(self):
        """Test configuration value range validation."""
        out_of_range_config = {
            "enhanced_reflection": {
                "enable_enhanced_reflection": True,
                "max_reflection_loops": 0,  # Should be > 0
                "reflection_temperature": 2.0,  # Should be <= 1.0
                "knowledge_gap_threshold": -0.1,  # Should be >= 0.0
                "sufficiency_threshold": 1.5  # Should be <= 1.0
            },
            "isolation": {
                "enable_isolation": True,
                "max_context_steps": -1,  # Should be > 0
                "context_window_size": 0  # Should be > 0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(out_of_range_config, f)
            temp_file = f.name
        
        try:
            loader = ResearcherConfigLoader(config_file=temp_file)
            
            with pytest.raises((ValueError, AssertionError)):
                config = loader.load_config()
                loader.validate_config(config)
        finally:
            os.unlink(temp_file)
    
    def test_concurrent_config_access(self):
        """Test concurrent configuration access."""
        import threading
        import time
        
        config_data = {
            "enhanced_reflection": {
                "enable_enhanced_reflection": True,
                "max_reflection_loops": 3
            },
            "isolation": {
                "enable_isolation": True,
                "isolation_level": "moderate"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            loader = ResearcherConfigLoader(config_file=temp_file)
            results = []
            
            def load_config_thread():
                try:
                    config = loader.load_config()
                    results.append(config is not None)
                except Exception as e:
                    results.append(False)
            
            # Create multiple threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=load_config_thread)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # All threads should succeed
            assert all(results)
            assert len(results) == 5
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    # Run the comprehensive configuration test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src.config",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])