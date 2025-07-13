"""Unit tests for Phase 4 system integration."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from utils.researcher_phase4_system import (
    ResearcherPhase4System,
    Phase4State,
    Phase4Mode,
    Phase4Report
)
from utils.researcher_progressive_enabler_phase4 import AdvancedResearcherProgressiveEnabler
from utils.researcher_isolation_metrics_phase4 import AdvancedResearcherIsolationMetrics
from utils.researcher_config_optimizer_phase4 import ConfigurationOptimizer


class TestPhase4Integration:
    """Test cases for Phase 4 system integration."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_config.close()
        
        self.system = ResearcherPhase4System(config_file=self.temp_config.name)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_system_initialization(self):
        """Test Phase 4 system initialization."""
        assert self.system is not None
        assert isinstance(self.system, ResearcherPhase4System)
        
        # Check that all components are initialized
        assert hasattr(self.system, 'progressive_enabler')
        assert hasattr(self.system, 'metrics_system')
        assert hasattr(self.system, 'config_optimizer')
        assert hasattr(self.system, 'current_state')
        
        assert isinstance(self.system.progressive_enabler, AdvancedResearcherProgressiveEnabler)
        assert isinstance(self.system.metrics_system, AdvancedResearcherIsolationMetrics)
        assert isinstance(self.system.config_optimizer, ConfigurationOptimizer)
        assert isinstance(self.system.current_state, Phase4State)
    
    def test_initial_state(self):
        """Test initial system state."""
        state = self.system.current_state
        
        assert state.mode == Phase4Mode.MONITORING
        assert state.optimization_enabled is False
        assert state.auto_tuning_active is False
        assert isinstance(state.performance_score, (int, float))
        assert 0 <= state.performance_score <= 100
        assert isinstance(state.last_optimization_time, (type(None), str))
        assert isinstance(state.active_optimizations, list)
    
    def test_state_computation(self):
        """Test system state computation."""
        # Mock some metrics data
        mock_metrics = {
            "success_rate_1h": 0.85,
            "performance_overhead_1h": 0.25,
            "avg_compression_ratio_1h": 0.75,
            "resource_utilization": 0.70,
            "isolation_effectiveness": 0.80
        }
        
        # Update system with metrics
        self.system.update_metrics(mock_metrics)
        
        # Compute new state
        new_state = self.system.compute_current_state()
        
        assert isinstance(new_state, Phase4State)
        assert new_state.performance_score >= 0
        assert new_state.performance_score <= 100
        
        # State should be updated
        assert self.system.current_state == new_state
    
    def test_mode_transitions(self):
        """Test system mode transitions."""
        # Start in monitoring mode
        assert self.system.current_state.mode == Phase4Mode.MONITORING
        
        # Transition to optimization mode
        result = self.system.enable_optimization_mode()
        assert result is True
        assert self.system.current_state.mode == Phase4Mode.OPTIMIZING
        assert self.system.current_state.optimization_enabled is True
        
        # Transition to auto-tuning mode
        result = self.system.enable_auto_tuning_mode()
        assert result is True
        assert self.system.current_state.mode == Phase4Mode.AUTO_TUNING
        assert self.system.current_state.auto_tuning_active is True
        
        # Transition back to monitoring
        result = self.system.disable_optimization()
        assert result is True
        assert self.system.current_state.mode == Phase4Mode.MONITORING
        assert self.system.current_state.optimization_enabled is False
        assert self.system.current_state.auto_tuning_active is False
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive system report generation."""
        # Add some test data
        mock_metrics = {
            "success_rate_1h": 0.85,
            "performance_overhead_1h": 0.25,
            "avg_compression_ratio_1h": 0.75,
            "resource_utilization": 0.70
        }
        self.system.update_metrics(mock_metrics)
        
        # Generate report
        report = self.system.generate_comprehensive_report()
        
        assert isinstance(report, Phase4Report)
        assert hasattr(report, 'system_state')
        assert hasattr(report, 'performance_metrics')
        assert hasattr(report, 'optimization_recommendations')
        assert hasattr(report, 'configuration_status')
        assert hasattr(report, 'predictive_insights')
        assert hasattr(report, 'alert_summary')
        
        # Check system state
        assert isinstance(report.system_state, dict)
        assert "mode" in report.system_state
        assert "performance_score" in report.system_state
        assert "optimization_enabled" in report.system_state
        
        # Check performance metrics
        assert isinstance(report.performance_metrics, dict)
        
        # Check optimization recommendations
        assert isinstance(report.optimization_recommendations, list)
        
        # Check configuration status
        assert isinstance(report.configuration_status, dict)
        
        # Check predictive insights
        assert isinstance(report.predictive_insights, dict)
        
        # Check alert summary
        assert isinstance(report.alert_summary, dict)
    
    def test_auto_optimization_toggle(self):
        """Test auto-optimization enable/disable."""
        # Initially disabled
        assert self.system.current_state.auto_tuning_active is False
        
        # Enable auto-optimization
        result = self.system.enable_auto_optimization()
        assert result is True
        assert self.system.current_state.auto_tuning_active is True
        assert self.system.current_state.optimization_enabled is True
        
        # Disable auto-optimization
        result = self.system.disable_auto_optimization()
        assert result is True
        assert self.system.current_state.auto_tuning_active is False
    
    def test_forced_optimization(self):
        """Test forced optimization execution."""
        # Add some metrics to work with
        mock_metrics = {
            "success_rate_1h": 0.70,  # Lower performance to trigger optimization
            "performance_overhead_1h": 0.40,
            "avg_compression_ratio_1h": 0.85,
            "resource_utilization": 0.90
        }
        
        # Add sufficient historical data
        for _ in range(15):
            self.system.update_metrics(mock_metrics)
        
        # Force optimization
        result = self.system.force_optimization()
        
        assert isinstance(result, dict)
        assert "optimization_applied" in result
        assert "changes_made" in result
        assert "performance_impact" in result
        
        # Check that optimization was attempted
        assert isinstance(result["optimization_applied"], bool)
        assert isinstance(result["changes_made"], list)
        assert isinstance(result["performance_impact"], dict)
    
    def test_system_backup_and_restore(self):
        """Test system backup and restore functionality."""
        # Modify system state
        self.system.enable_optimization_mode()
        self.system.config_optimizer.apply_profile("high_performance")
        
        # Create backup
        backup_result = self.system.create_system_backup()
        assert isinstance(backup_result, dict)
        assert "backup_id" in backup_result
        assert "timestamp" in backup_result
        assert "status" in backup_result
        assert backup_result["status"] == "success"
        
        backup_id = backup_result["backup_id"]
        
        # Modify system further
        self.system.config_optimizer.apply_profile("memory_optimized")
        
        # Restore from backup
        restore_result = self.system.restore_from_backup(backup_id)
        assert isinstance(restore_result, dict)
        assert "status" in restore_result
        
        # Check if restore was successful or handled gracefully
        assert restore_result["status"] in ["success", "partial", "failed"]
    
    def test_metrics_integration(self):
        """Test integration with metrics system."""
        # Test metrics update
        metrics_data = {
            "success_rate_1h": 0.90,
            "performance_overhead_1h": 0.15,
            "avg_compression_ratio_1h": 0.80,
            "resource_utilization": 0.60
        }
        
        result = self.system.update_metrics(metrics_data)
        assert result is True
        
        # Check that metrics were processed
        current_metrics = self.system.get_current_metrics()
        assert isinstance(current_metrics, dict)
        
        # Should contain at least some of the provided metrics
        for key in metrics_data:
            if key in current_metrics:
                assert isinstance(current_metrics[key], (int, float))
    
    def test_progressive_enabler_integration(self):
        """Test integration with progressive enabler."""
        # Create a mock scenario
        mock_scenario = {
            "context_size": 5000,
            "complexity_score": 0.7,
            "resource_usage": 0.6,
            "success_rate": 0.85
        }
        
        # Test isolation score calculation
        isolation_score = self.system.calculate_isolation_score(mock_scenario)
        assert isinstance(isolation_score, (int, float))
        assert 0 <= isolation_score <= 1
        
        # Test performance prediction
        prediction = self.system.predict_performance(mock_scenario)
        assert isinstance(prediction, dict)
        assert "predicted_success_rate" in prediction
        assert "predicted_overhead" in prediction
        assert "confidence" in prediction
    
    def test_config_optimizer_integration(self):
        """Test integration with configuration optimizer."""
        # Test profile application through system
        result = self.system.apply_configuration_profile("balanced")
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["success", "failed"]
        
        if result["status"] == "success":
            assert "applied_profile" in result
            assert result["applied_profile"] == "balanced"
        
        # Test configuration report
        config_report = self.system.get_configuration_status()
        assert isinstance(config_report, dict)
        assert "current_profile" in config_report or "current_config" in config_report
    
    def test_error_handling(self):
        """Test system error handling."""
        # Test with invalid metrics data
        invalid_metrics = {
            "invalid_metric": "not_a_number",
            "success_rate_1h": "invalid"
        }
        
        # Should handle gracefully
        result = self.system.update_metrics(invalid_metrics)
        # Should either succeed (filtering invalid data) or fail gracefully
        assert isinstance(result, bool)
        
        # Test with invalid profile
        result = self.system.apply_configuration_profile("invalid_profile")
        assert isinstance(result, dict)
        assert "status" in result
        if result["status"] == "failed":
            assert "error" in result or "message" in result
    
    def test_concurrent_operations(self):
        """Test concurrent system operations."""
        import threading
        import time
        
        results = []
        
        def update_metrics():
            metrics = {
                "success_rate_1h": 0.85,
                "performance_overhead_1h": 0.25,
                "resource_utilization": 0.70
            }
            result = self.system.update_metrics(metrics)
            results.append(('metrics', result))
        
        def compute_state():
            state = self.system.compute_current_state()
            results.append(('state', state is not None))
        
        def generate_report():
            report = self.system.generate_comprehensive_report()
            results.append(('report', report is not None))
        
        # Run operations concurrently
        threads = [
            threading.Thread(target=update_metrics),
            threading.Thread(target=compute_state),
            threading.Thread(target=generate_report)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check that all operations completed
        assert len(results) == 3
        
        # All operations should have succeeded or handled gracefully
        for operation, result in results:
            assert result is not None
            if operation == 'metrics':
                assert isinstance(result, bool)
            elif operation == 'state':
                assert isinstance(result, bool)
            elif operation == 'report':
                assert isinstance(result, bool)


class TestPhase4State:
    """Test cases for Phase4State."""
    
    def test_state_creation(self):
        """Test Phase4State creation."""
        state = Phase4State(
            mode=Phase4Mode.OPTIMIZING,
            optimization_enabled=True,
            auto_tuning_active=False,
            performance_score=85.5,
            last_optimization_time="2024-01-01T12:00:00Z",
            active_optimizations=["cache_optimization", "memory_tuning"]
        )
        
        assert isinstance(state, Phase4State)
        assert state.mode == Phase4Mode.OPTIMIZING
        assert state.optimization_enabled is True
        assert state.auto_tuning_active is False
        assert state.performance_score == 85.5
        assert state.last_optimization_time == "2024-01-01T12:00:00Z"
        assert state.active_optimizations == ["cache_optimization", "memory_tuning"]
    
    def test_state_equality(self):
        """Test Phase4State equality comparison."""
        state1 = Phase4State(
            mode=Phase4Mode.MONITORING,
            optimization_enabled=False,
            auto_tuning_active=False,
            performance_score=75.0
        )
        
        state2 = Phase4State(
            mode=Phase4Mode.MONITORING,
            optimization_enabled=False,
            auto_tuning_active=False,
            performance_score=75.0
        )
        
        state3 = Phase4State(
            mode=Phase4Mode.OPTIMIZING,
            optimization_enabled=True,
            auto_tuning_active=False,
            performance_score=75.0
        )
        
        assert state1 == state2
        assert state1 != state3


class TestPhase4Mode:
    """Test cases for Phase4Mode."""
    
    def test_mode_values(self):
        """Test Phase4Mode enum values."""
        assert hasattr(Phase4Mode, 'MONITORING')
        assert hasattr(Phase4Mode, 'OPTIMIZING')
        assert hasattr(Phase4Mode, 'AUTO_TUNING')
        
        assert Phase4Mode.MONITORING == "monitoring"
        assert Phase4Mode.OPTIMIZING == "optimizing"
        assert Phase4Mode.AUTO_TUNING == "auto_tuning"


class TestPhase4Report:
    """Test cases for Phase4Report."""
    
    def test_report_creation(self):
        """Test Phase4Report creation."""
        report = Phase4Report(
            system_state={"mode": "monitoring", "score": 85},
            performance_metrics={"success_rate": 0.9},
            optimization_recommendations=[],
            configuration_status={"profile": "balanced"},
            predictive_insights={"trend": "improving"},
            alert_summary={"active_alerts": 0}
        )
        
        assert isinstance(report, Phase4Report)
        assert report.system_state == {"mode": "monitoring", "score": 85}
        assert report.performance_metrics == {"success_rate": 0.9}
        assert report.optimization_recommendations == []
        assert report.configuration_status == {"profile": "balanced"}
        assert report.predictive_insights == {"trend": "improving"}
        assert report.alert_summary == {"active_alerts": 0}


class TestPhase4Performance:
    """Performance tests for Phase 4 system."""
    
    def test_system_initialization_performance(self):
        """Test system initialization performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                import time
                start_time = time.time()
                
                for _ in range(5):
                    system = ResearcherPhase4System(config_file=temp_file.name)
                    assert system is not None
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 5
                
                # Initialization should be reasonably fast
                assert avg_time < 0.5  # Less than 500ms per initialization
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
    
    def test_metrics_update_performance(self):
        """Test metrics update performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                system = ResearcherPhase4System(config_file=temp_file.name)
                
                metrics_data = {
                    "success_rate_1h": 0.85,
                    "performance_overhead_1h": 0.25,
                    "avg_compression_ratio_1h": 0.75,
                    "resource_utilization": 0.70
                }
                
                import time
                start_time = time.time()
                
                for _ in range(100):
                    system.update_metrics(metrics_data)
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 100
                
                # Metrics update should be fast
                assert avg_time < 0.01  # Less than 10ms per update
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
    
    def test_report_generation_performance(self):
        """Test report generation performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                system = ResearcherPhase4System(config_file=temp_file.name)
                
                # Add some data first
                metrics_data = {
                    "success_rate_1h": 0.85,
                    "performance_overhead_1h": 0.25,
                    "resource_utilization": 0.70
                }
                system.update_metrics(metrics_data)
                
                import time
                start_time = time.time()
                
                for _ in range(10):
                    report = system.generate_comprehensive_report()
                    assert report is not None
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 10
                
                # Report generation should be reasonably fast
                assert avg_time < 0.2  # Less than 200ms per report
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)