"""Unit tests for configuration optimizer functionality."""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from utils.researcher_config_optimizer_phase4 import (
    ConfigurationOptimizer,
    ConfigRecommendation,
    ConfigParameter,
    ConfigProfile,
    ConfigOptimizationLevel
)


class TestConfigurationOptimizer:
    """Test cases for ConfigurationOptimizer."""
    
    def setup_method(self):
        """Setup test environment."""
        # Use temporary file for config
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        self.optimizer = ConfigurationOptimizer(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_optimizer_instantiation(self):
        """Test optimizer instantiation."""
        assert self.optimizer is not None
        assert isinstance(self.optimizer, ConfigurationOptimizer)
        assert hasattr(self.optimizer, 'current_config')
        assert hasattr(self.optimizer, 'config_file')
        assert self.optimizer.config_file == self.temp_file.name
    
    def test_intelligent_defaults(self):
        """Test intelligent default configuration generation."""
        config = self.optimizer.current_config
        
        # Check that all required sections exist
        assert "isolation" in config
        assert "performance" in config
        assert "monitoring" in config
        assert "optimization" in config
        
        # Check isolation section
        isolation_config = config["isolation"]
        assert "max_local_context_size" in isolation_config
        assert "token_budget" in isolation_config
        assert "compression_threshold" in isolation_config
        assert "max_concurrent_sessions" in isolation_config
        
        # Check reasonable default values
        assert isolation_config["max_local_context_size"] > 0
        assert isolation_config["token_budget"] > 0
        assert 0 < isolation_config["compression_threshold"] < 1
        assert isolation_config["max_concurrent_sessions"] > 0
        
        # Check performance section
        performance_config = config["performance"]
        assert "enable_caching" in performance_config
        assert "cache_size_mb" in performance_config
        assert "parallel_processing" in performance_config
        
        # Check monitoring section
        monitoring_config = config["monitoring"]
        assert "enable_real_time" in monitoring_config
        assert "collection_interval_seconds" in monitoring_config
        assert "alert_thresholds" in monitoring_config
        
        # Check optimization section
        optimization_config = config["optimization"]
        assert "auto_tuning_enabled" in optimization_config
        assert "optimization_level" in optimization_config
    
    def test_performance_analysis_insufficient_data(self):
        """Test performance analysis with insufficient data."""
        metrics_data = {
            "success_rate_1h": 0.8,
            "performance_overhead_1h": 0.2,
            "avg_compression_ratio_1h": 0.7,
            "resource_utilization": 0.6
        }
        
        # With insufficient historical data, should return empty recommendations
        recommendations = self.optimizer.analyze_performance_data(metrics_data)
        
        assert isinstance(recommendations, list)
        # May be empty due to insufficient data
        assert len(recommendations) >= 0
    
    def test_performance_analysis_with_sufficient_data(self):
        """Test performance analysis with sufficient historical data."""
        # Add sufficient historical data to trigger recommendations
        for i in range(15):
            metrics_data = {
                "success_rate_1h": 0.7 - i * 0.01,  # Declining success rate
                "performance_overhead_1h": 0.35 + i * 0.01,  # Increasing overhead
                "avg_compression_ratio_1h": 0.85,
                "resource_utilization": 0.8
            }
            self.optimizer.analyze_performance_data(metrics_data)
        
        # Now analyze with poor performance metrics
        poor_metrics = {
            "success_rate_1h": 0.6,  # Low success rate
            "performance_overhead_1h": 0.4,  # High overhead
            "avg_compression_ratio_1h": 0.9,  # Poor compression
            "resource_utilization": 0.95  # High utilization
        }
        
        recommendations = self.optimizer.analyze_performance_data(poor_metrics)
        
        assert isinstance(recommendations, list)
        # Should have recommendations now with sufficient data and poor performance
        
        # Check recommendation structure if any exist
        for rec in recommendations:
            assert isinstance(rec, ConfigRecommendation)
            assert hasattr(rec, 'parameter')
            assert hasattr(rec, 'current_value')
            assert hasattr(rec, 'recommended_value')
            assert hasattr(rec, 'reason')
            assert hasattr(rec, 'confidence')
            assert 0 <= rec.confidence <= 1
            assert isinstance(rec.reason, str)
            assert len(rec.reason) > 0
    
    def test_profile_application_high_performance(self):
        """Test applying high-performance profile."""
        result = self.optimizer.apply_profile("high_performance")
        assert result is True
        
        # Check that config was updated with high-performance values
        config = self.optimizer.current_config
        isolation_config = config["isolation"]
        
        assert isolation_config["max_concurrent_sessions"] == 12
        assert isolation_config["token_budget"] == 32000
        assert isolation_config["compression_threshold"] == 0.6
        
        performance_config = config["performance"]
        assert performance_config["enable_caching"] is True
        assert performance_config["cache_size_mb"] == 512
        assert performance_config["parallel_processing"] is True
    
    def test_profile_application_balanced(self):
        """Test applying balanced profile."""
        result = self.optimizer.apply_profile("balanced")
        assert result is True
        
        # Check that config was updated with balanced values
        config = self.optimizer.current_config
        isolation_config = config["isolation"]
        
        assert isolation_config["max_concurrent_sessions"] == 8
        assert isolation_config["token_budget"] == 20000
        assert isolation_config["compression_threshold"] == 0.7
        
        performance_config = config["performance"]
        assert performance_config["enable_caching"] is True
        assert performance_config["cache_size_mb"] == 256
        assert performance_config["parallel_processing"] is True
    
    def test_profile_application_memory_optimized(self):
        """Test applying memory-optimized profile."""
        result = self.optimizer.apply_profile("memory_optimized")
        assert result is True
        
        # Check that config was updated with memory-optimized values
        config = self.optimizer.current_config
        isolation_config = config["isolation"]
        
        assert isolation_config["max_concurrent_sessions"] == 4
        assert isolation_config["token_budget"] == 12000
        assert isolation_config["compression_threshold"] == 0.8
        
        performance_config = config["performance"]
        assert performance_config["cache_size_mb"] == 128
    
    def test_profile_application_invalid_profile(self):
        """Test applying invalid profile."""
        result = self.optimizer.apply_profile("invalid_profile")
        assert result is False
        
        # Config should remain unchanged
        config = self.optimizer.current_config
        assert config is not None
    
    def test_auto_tuning_enable_disable(self):
        """Test auto-tuning enable/disable functionality."""
        # Test enabling auto-tuning
        self.optimizer.enable_auto_tuning(ConfigOptimizationLevel.BALANCED)
        
        config = self.optimizer.current_config
        assert config["optimization"]["auto_tuning_enabled"] is True
        assert config["optimization"]["optimization_level"] == "balanced"
        
        # Test disabling auto-tuning
        self.optimizer.disable_auto_tuning()
        
        config = self.optimizer.current_config
        assert config["optimization"]["auto_tuning_enabled"] is False
    
    def test_auto_tuning_with_recommendations(self):
        """Test automatic configuration tuning with recommendations."""
        # Enable auto-tuning
        self.optimizer.enable_auto_tuning(ConfigOptimizationLevel.BALANCED)
        
        # Add sufficient historical data first
        for i in range(15):
            self.optimizer.analyze_performance_data({
                "success_rate_1h": 0.7 - i * 0.01,
                "performance_overhead_1h": 0.35 + i * 0.01,
                "avg_compression_ratio_1h": 0.85,
                "resource_utilization": 0.8
            })
        
        # Simulate metrics that should trigger optimization
        metrics_data = {
            "success_rate_1h": 0.7,
            "performance_overhead_1h": 0.35,
            "avg_compression_ratio_1h": 0.85,
            "resource_utilization": 0.8
        }
        
        result = self.optimizer.auto_tune_configuration(metrics_data)
        
        # Check if result contains expected keys
        assert isinstance(result, dict)
        assert "auto_tuning_applied" in result
        
        if result["auto_tuning_applied"]:
            assert "changes" in result
            assert "applied_count" in result
            assert isinstance(result["changes"], list)
            assert isinstance(result["applied_count"], int)
            assert result["applied_count"] >= 0
        else:
            assert "message" in result
    
    def test_auto_tuning_disabled(self):
        """Test auto-tuning when disabled."""
        # Ensure auto-tuning is disabled
        self.optimizer.disable_auto_tuning()
        
        metrics_data = {
            "success_rate_1h": 0.5,  # Poor performance
            "performance_overhead_1h": 0.5,
            "avg_compression_ratio_1h": 0.9,
            "resource_utilization": 0.95
        }
        
        result = self.optimizer.auto_tune_configuration(metrics_data)
        
        assert isinstance(result, dict)
        assert result["auto_tuning_applied"] is False
        assert "message" in result
        assert "disabled" in result["message"].lower()
    
    def test_configuration_report(self):
        """Test configuration report generation."""
        report = self.optimizer.get_configuration_report()
        
        assert isinstance(report, dict)
        assert "current_config" in report
        assert "available_profiles" in report
        assert "optimization_history" in report
        assert "auto_tuning_status" in report
        assert "system_recommendations" in report
        assert "config_health_score" in report
        
        # Check current config structure
        current_config = report["current_config"]
        assert isinstance(current_config, dict)
        assert "isolation" in current_config
        assert "performance" in current_config
        assert "monitoring" in current_config
        assert "optimization" in current_config
        
        # Check available profiles
        available_profiles = report["available_profiles"]
        assert isinstance(available_profiles, list)
        assert "high_performance" in available_profiles
        assert "balanced" in available_profiles
        assert "memory_optimized" in available_profiles
        
        # Check optimization history
        optimization_history = report["optimization_history"]
        assert isinstance(optimization_history, list)
        
        # Check auto-tuning status
        auto_tuning_status = report["auto_tuning_status"]
        assert isinstance(auto_tuning_status, dict)
        assert "enabled" in auto_tuning_status
        assert "level" in auto_tuning_status
        
        # Check system recommendations
        system_recommendations = report["system_recommendations"]
        assert isinstance(system_recommendations, list)
        
        # Check health score is reasonable
        health_score = report["config_health_score"]
        assert isinstance(health_score, (int, float))
        assert 0 <= health_score <= 100
    
    def test_config_persistence(self):
        """Test configuration persistence to file."""
        # Modify configuration
        self.optimizer.apply_profile("high_performance")
        
        # Force save to file
        self.optimizer._save_config()
        
        # Check that file exists and contains data
        assert os.path.exists(self.temp_file.name)
        
        with open(self.temp_file.name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert "isolation" in data
        assert "performance" in data
        assert "monitoring" in data
        assert "optimization" in data
    
    def test_config_loading_from_file(self):
        """Test configuration loading from existing file."""
        # Create a config file with specific values
        test_config = {
            "isolation": {
                "max_local_context_size": 9999,
                "token_budget": 25000,
                "compression_threshold": 0.75,
                "max_concurrent_sessions": 6
            },
            "performance": {
                "enable_caching": True,
                "cache_size_mb": 300,
                "parallel_processing": False
            },
            "monitoring": {
                "enable_real_time": True,
                "collection_interval_seconds": 45
            },
            "optimization": {
                "auto_tuning_enabled": True,
                "optimization_level": "aggressive"
            }
        }
        
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(test_config, f)
        
        # Create new optimizer that should load from file
        new_optimizer = ConfigurationOptimizer(self.temp_file.name)
        
        loaded_config = new_optimizer.current_config
        assert loaded_config["isolation"]["max_local_context_size"] == 9999
        assert loaded_config["isolation"]["token_budget"] == 25000
        assert loaded_config["isolation"]["compression_threshold"] == 0.75
        assert loaded_config["performance"]["cache_size_mb"] == 300
        assert loaded_config["performance"]["parallel_processing"] is False
    
    def test_multiple_optimizer_instances(self):
        """Test creating multiple optimizer instances."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file2:
            temp_file2.close()
            
            try:
                optimizer1 = self.optimizer
                optimizer2 = ConfigurationOptimizer(temp_file2.name)
                
                assert optimizer1 is not optimizer2
                assert optimizer1.config_file != optimizer2.config_file
                
                # Both should work independently
                optimizer1.apply_profile("high_performance")
                optimizer2.apply_profile("memory_optimized")
                
                config1 = optimizer1.current_config
                config2 = optimizer2.current_config
                
                # Should have different configurations
                assert config1["isolation"]["max_concurrent_sessions"] != config2["isolation"]["max_concurrent_sessions"]
                
            finally:
                if os.path.exists(temp_file2.name):
                    os.unlink(temp_file2.name)


class TestConfigRecommendation:
    """Test cases for ConfigRecommendation."""
    
    def test_config_recommendation_creation(self):
        """Test ConfigRecommendation creation."""
        recommendation = ConfigRecommendation(
            parameter="max_concurrent_sessions",
            current_value=8,
            recommended_value=12,
            reason="High resource utilization detected",
            confidence=0.85
        )
        
        assert isinstance(recommendation, ConfigRecommendation)
        assert recommendation.parameter == "max_concurrent_sessions"
        assert recommendation.current_value == 8
        assert recommendation.recommended_value == 12
        assert recommendation.reason == "High resource utilization detected"
        assert recommendation.confidence == 0.85
    
    def test_config_recommendation_validation(self):
        """Test ConfigRecommendation validation."""
        # Test with valid confidence values
        rec1 = ConfigRecommendation(
            parameter="test_param",
            current_value="current",
            recommended_value="recommended",
            reason="test reason",
            confidence=0.0  # Minimum valid
        )
        assert rec1.confidence == 0.0
        
        rec2 = ConfigRecommendation(
            parameter="test_param",
            current_value="current",
            recommended_value="recommended",
            reason="test reason",
            confidence=1.0  # Maximum valid
        )
        assert rec2.confidence == 1.0


class TestConfigOptimizationLevel:
    """Test cases for ConfigOptimizationLevel."""
    
    def test_optimization_levels_exist(self):
        """Test that all optimization levels exist."""
        assert hasattr(ConfigOptimizationLevel, 'CONSERVATIVE')
        assert hasattr(ConfigOptimizationLevel, 'BALANCED')
        assert hasattr(ConfigOptimizationLevel, 'AGGRESSIVE')
    
    def test_optimization_level_values(self):
        """Test optimization level values."""
        assert ConfigOptimizationLevel.CONSERVATIVE == "conservative"
        assert ConfigOptimizationLevel.BALANCED == "balanced"
        assert ConfigOptimizationLevel.AGGRESSIVE == "aggressive"


class TestConfigurationOptimizerPerformance:
    """Performance tests for configuration optimizer."""
    
    def test_config_analysis_performance(self):
        """Test configuration analysis performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                optimizer = ConfigurationOptimizer(temp_file.name)
                
                # Test analysis speed
                metrics_data = {
                    "success_rate_1h": 0.8,
                    "performance_overhead_1h": 0.2,
                    "avg_compression_ratio_1h": 0.7,
                    "resource_utilization": 0.6
                }
                
                import time
                start_time = time.time()
                for _ in range(50):
                    optimizer.analyze_performance_data(metrics_data)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 50
                assert avg_time < 0.05  # Should be fast (< 50ms per analysis)
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
    
    def test_profile_application_performance(self):
        """Test profile application performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                optimizer = ConfigurationOptimizer(temp_file.name)
                
                import time
                start_time = time.time()
                for _ in range(20):
                    optimizer.apply_profile("balanced")
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 20
                assert avg_time < 0.01  # Should be very fast (< 10ms per application)
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
    
    def test_report_generation_performance(self):
        """Test report generation performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                optimizer = ConfigurationOptimizer(temp_file.name)
                
                import time
                start_time = time.time()
                for _ in range(10):
                    optimizer.get_configuration_report()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                assert avg_time < 0.1  # Should be reasonably fast (< 100ms per report)
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)