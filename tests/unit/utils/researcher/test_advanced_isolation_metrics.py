"""Unit tests for advanced isolation metrics functionality."""

import pytest
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from utils.researcher_isolation_metrics_phase4 import (
    AdvancedResearcherIsolationMetrics,
    PredictiveAnalyzer,
    AlertManager,
    RealTimeMetrics,
    SystemAlert,
    AlertLevel,
    PerformanceTrend
)


class TestAdvancedResearcherIsolationMetrics:
    """Test cases for AdvancedResearcherIsolationMetrics."""
    
    def setup_method(self):
        """Setup test environment."""
        # Use temporary file for metrics
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        self.metrics = AdvancedResearcherIsolationMetrics(self.temp_file.name)
        # Disable monitoring for tests
        self.metrics.monitoring_enabled = False
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.metrics.stop_monitoring()
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_metrics_instantiation(self):
        """Test metrics instantiation."""
        assert self.metrics is not None
        assert isinstance(self.metrics, AdvancedResearcherIsolationMetrics)
        assert hasattr(self.metrics, 'real_time_metrics')
        assert hasattr(self.metrics, 'predictive_analyzer')
        assert hasattr(self.metrics, 'alert_manager')
        assert isinstance(self.metrics.predictive_analyzer, PredictiveAnalyzer)
        assert isinstance(self.metrics.alert_manager, AlertManager)
    
    def test_real_time_metrics_collection(self):
        """Test real-time metrics collection."""
        # Manually trigger metrics collection
        self.metrics._collect_real_time_metrics()
        
        # Check that metrics were collected
        assert len(self.metrics.real_time_metrics) > 0
        
        latest_metrics = self.metrics.real_time_metrics[-1]
        assert isinstance(latest_metrics, RealTimeMetrics)
        assert 0 <= latest_metrics.system_health_score <= 100
        assert latest_metrics.resource_utilization >= 0
        assert latest_metrics.active_sessions >= 0
        assert 0 <= latest_metrics.avg_compression_ratio_1h <= 1
        assert 0 <= latest_metrics.token_savings_rate_1h <= 1
        assert 0 <= latest_metrics.success_rate_1h <= 1
        assert latest_metrics.performance_overhead_1h >= 0
    
    def test_multiple_metrics_collection(self):
        """Test multiple metrics collection cycles."""
        initial_count = len(self.metrics.real_time_metrics)
        
        # Collect metrics multiple times
        for _ in range(3):
            self.metrics._collect_real_time_metrics()
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        assert len(self.metrics.real_time_metrics) == initial_count + 3
        
        # Check that timestamps are different
        timestamps = [m.timestamp for m in self.metrics.real_time_metrics[-3:]]
        assert len(set(timestamps)) == 3  # All timestamps should be unique
    
    def test_dashboard_data_generation(self):
        """Test real-time dashboard data generation."""
        # Add some test metrics
        self.metrics._collect_real_time_metrics()
        
        dashboard_data = self.metrics.get_real_time_dashboard()
        
        assert isinstance(dashboard_data, dict)
        assert "current_metrics" in dashboard_data
        assert "trends" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "24h_summary" in dashboard_data
        assert "optimization_suggestions" in dashboard_data
        assert "system_status" in dashboard_data
        
        # Check current metrics structure
        current_metrics = dashboard_data["current_metrics"]
        assert isinstance(current_metrics, dict)
        
        # Check trends structure
        trends = dashboard_data["trends"]
        assert isinstance(trends, dict)
        
        # Check system status
        system_status = dashboard_data["system_status"]
        assert isinstance(system_status, str)
        assert system_status in ["healthy", "warning", "critical"]
    
    def test_predictive_insights_generation(self):
        """Test predictive insights generation."""
        # Add some metrics first
        self.metrics._collect_real_time_metrics()
        
        insights = self.metrics.get_predictive_insights()
        
        assert isinstance(insights, dict)
        assert "trends_analysis" in insights
        assert "load_prediction_24h" in insights
        assert "recommendations" in insights
        
        # Check trends analysis structure
        trends_analysis = insights["trends_analysis"]
        assert isinstance(trends_analysis, dict)
        
        # Check load prediction structure
        load_prediction = insights["load_prediction_24h"]
        assert isinstance(load_prediction, dict)
        
        # Check recommendations structure
        recommendations = insights["recommendations"]
        assert isinstance(recommendations, list)
    
    def test_metrics_persistence(self):
        """Test metrics persistence to file."""
        # Collect some metrics
        self.metrics._collect_real_time_metrics()
        
        # Force save to file
        self.metrics._save_metrics_to_file()
        
        # Check that file exists and contains data
        assert os.path.exists(self.temp_file.name)
        
        with open(self.temp_file.name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert "real_time_metrics" in data or "metrics" in data
    
    def test_monitoring_enable_disable(self):
        """Test monitoring enable/disable functionality."""
        # Test enabling monitoring
        self.metrics.start_monitoring()
        assert self.metrics.monitoring_enabled is True
        
        # Test disabling monitoring
        self.metrics.stop_monitoring()
        assert self.metrics.monitoring_enabled is False
    
    def test_metrics_with_custom_config(self):
        """Test metrics with custom configuration."""
        # Create metrics with custom config
        custom_config = {
            "collection_interval": 30,
            "max_metrics_history": 1000,
            "alert_thresholds": {
                "low_success_rate": 0.7,
                "high_resource_utilization": 0.9
            }
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                metrics = AdvancedResearcherIsolationMetrics(
                    temp_file.name, 
                    config=custom_config
                )
                metrics.monitoring_enabled = False
                
                assert metrics is not None
                assert isinstance(metrics, AdvancedResearcherIsolationMetrics)
                
                # Test metrics collection with custom config
                metrics._collect_real_time_metrics()
                assert len(metrics.real_time_metrics) > 0
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)


class TestPredictiveAnalyzer:
    """Test cases for PredictiveAnalyzer."""
    
    def setup_method(self):
        """Setup test environment."""
        self.analyzer = PredictiveAnalyzer()
    
    def test_analyzer_instantiation(self):
        """Test analyzer instantiation."""
        assert self.analyzer is not None
        assert isinstance(self.analyzer, PredictiveAnalyzer)
        assert hasattr(self.analyzer, 'data_points')
        assert isinstance(self.analyzer.data_points, list)
    
    def test_data_point_addition(self):
        """Test adding data points to analyzer."""
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=3,
            avg_compression_ratio_1h=0.7,
            token_savings_rate_1h=0.3,
            success_rate_1h=0.9,
            performance_overhead_1h=0.2,
            system_health_score=85,
            resource_utilization=0.6
        )
        
        initial_count = len(self.analyzer.data_points)
        self.analyzer.add_data_point(metrics)
        
        assert len(self.analyzer.data_points) == initial_count + 1
        assert self.analyzer.data_points[-1] == metrics
    
    def test_trend_analysis_with_data(self):
        """Test trend analysis with sufficient data."""
        # Add multiple data points
        for i in range(20):
            metrics = RealTimeMetrics(
                timestamp=f"2024-01-01T{i:02d}:00:00",
                active_sessions=i % 5,
                avg_compression_ratio_1h=0.7 + (i % 3) * 0.1,
                token_savings_rate_1h=0.3 + (i % 4) * 0.1,
                success_rate_1h=0.9 - (i % 2) * 0.1,
                performance_overhead_1h=0.2 + (i % 3) * 0.05,
                system_health_score=80 + (i % 2) * 10,
                resource_utilization=0.5 + (i % 4) * 0.1
            )
            self.analyzer.add_data_point(metrics)
        
        # Test trend analysis
        trends = self.analyzer.analyze_trends()
        
        assert isinstance(trends, dict)
        assert len(trends) > 0
        
        # Check that trend values are reasonable
        for key, value in trends.items():
            if isinstance(value, dict) and 'trend' in value:
                assert value['trend'] in ['improving', 'stable', 'degrading']
    
    def test_trend_analysis_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        # Add only a few data points
        for i in range(2):
            metrics = RealTimeMetrics(
                timestamp=f"2024-01-01T{i:02d}:00:00",
                active_sessions=i,
                avg_compression_ratio_1h=0.7,
                token_savings_rate_1h=0.3,
                success_rate_1h=0.9,
                performance_overhead_1h=0.2,
                system_health_score=85,
                resource_utilization=0.6
            )
            self.analyzer.add_data_point(metrics)
        
        trends = self.analyzer.analyze_trends()
        
        # Should still return a dict, but may have limited analysis
        assert isinstance(trends, dict)
    
    def test_load_prediction(self):
        """Test system load prediction."""
        # Add some historical data
        for i in range(15):
            metrics = RealTimeMetrics(
                timestamp=f"2024-01-01T{i:02d}:00:00",
                active_sessions=i % 5 + 1,
                avg_compression_ratio_1h=0.7,
                token_savings_rate_1h=0.3,
                success_rate_1h=0.9,
                performance_overhead_1h=0.2,
                system_health_score=85,
                resource_utilization=0.6
            )
            self.analyzer.add_data_point(metrics)
        
        # Test load prediction
        load_prediction = self.analyzer.predict_system_load(24)
        
        assert isinstance(load_prediction, dict)
        assert "predicted_sessions" in load_prediction
        assert "confidence" in load_prediction
        assert 0 <= load_prediction["confidence"] <= 1
        assert load_prediction["predicted_sessions"] >= 0
    
    def test_load_prediction_no_data(self):
        """Test load prediction with no historical data."""
        load_prediction = self.analyzer.predict_system_load(24)
        
        assert isinstance(load_prediction, dict)
        assert "predicted_sessions" in load_prediction
        assert "confidence" in load_prediction
        # Confidence should be low with no data
        assert load_prediction["confidence"] <= 0.5


class TestAlertManager:
    """Test cases for AlertManager."""
    
    def setup_method(self):
        """Setup test environment."""
        self.alert_manager = AlertManager()
    
    def test_alert_manager_instantiation(self):
        """Test alert manager instantiation."""
        assert self.alert_manager is not None
        assert isinstance(self.alert_manager, AlertManager)
        assert hasattr(self.alert_manager, 'active_alerts')
        assert isinstance(self.alert_manager.active_alerts, list)
    
    def test_alert_creation_low_success_rate(self):
        """Test alert creation for low success rate."""
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=5,
            avg_compression_ratio_1h=0.7,
            token_savings_rate_1h=0.3,
            success_rate_1h=0.5,  # Low success rate
            performance_overhead_1h=0.2,
            system_health_score=85,
            resource_utilization=0.6
        )
        
        initial_count = len(self.alert_manager.active_alerts)
        self.alert_manager.check_and_create_alerts(metrics, {})
        
        # Should have created an alert
        assert len(self.alert_manager.active_alerts) > initial_count
        
        # Check that low success rate alert was created
        alert_ids = [alert.alert_id for alert in self.alert_manager.active_alerts]
        assert "low_success_rate" in alert_ids
    
    def test_alert_creation_high_resource_utilization(self):
        """Test alert creation for high resource utilization."""
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=5,
            avg_compression_ratio_1h=0.7,
            token_savings_rate_1h=0.3,
            success_rate_1h=0.9,
            performance_overhead_1h=0.2,
            system_health_score=85,
            resource_utilization=0.95  # High utilization
        )
        
        initial_count = len(self.alert_manager.active_alerts)
        self.alert_manager.check_and_create_alerts(metrics, {})
        
        # Should have created an alert
        assert len(self.alert_manager.active_alerts) > initial_count
        
        # Check that high resource utilization alert was created
        alert_ids = [alert.alert_id for alert in self.alert_manager.active_alerts]
        assert "high_resource_utilization" in alert_ids
    
    def test_alert_creation_multiple_issues(self):
        """Test alert creation for multiple issues."""
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=5,
            avg_compression_ratio_1h=0.9,  # Poor compression
            token_savings_rate_1h=0.1,  # Low savings
            success_rate_1h=0.5,  # Low success rate
            performance_overhead_1h=0.4,  # High overhead
            system_health_score=30,  # Low health
            resource_utilization=0.95  # High utilization
        )
        
        initial_count = len(self.alert_manager.active_alerts)
        self.alert_manager.check_and_create_alerts(metrics, {})
        
        # Should have created multiple alerts
        assert len(self.alert_manager.active_alerts) > initial_count
        
        # Check that multiple alert types were created
        alert_ids = [alert.alert_id for alert in self.alert_manager.active_alerts]
        assert "low_success_rate" in alert_ids
        assert "high_resource_utilization" in alert_ids
    
    def test_alert_creation_normal_metrics(self):
        """Test alert creation with normal metrics."""
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=3,
            avg_compression_ratio_1h=0.7,  # Good compression
            token_savings_rate_1h=0.4,  # Good savings
            success_rate_1h=0.95,  # High success rate
            performance_overhead_1h=0.15,  # Low overhead
            system_health_score=90,  # High health
            resource_utilization=0.6  # Normal utilization
        )
        
        initial_count = len(self.alert_manager.active_alerts)
        self.alert_manager.check_and_create_alerts(metrics, {})
        
        # Should not have created new alerts for normal metrics
        new_alerts = self.alert_manager.active_alerts[initial_count:]
        assert len(new_alerts) == 0
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Create some alerts first
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=5,
            avg_compression_ratio_1h=0.7,
            token_savings_rate_1h=0.3,
            success_rate_1h=0.5,  # Low success rate
            performance_overhead_1h=0.2,
            system_health_score=85,
            resource_utilization=0.95  # High utilization
        )
        
        self.alert_manager.check_and_create_alerts(metrics, {})
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        assert isinstance(active_alerts, list)
        assert len(active_alerts) > 0
        
        # Check alert structure
        for alert in active_alerts:
            assert isinstance(alert, SystemAlert)
            assert hasattr(alert, 'alert_id')
            assert hasattr(alert, 'level')
            assert hasattr(alert, 'message')
            assert hasattr(alert, 'timestamp')
            assert alert.level in [level for level in AlertLevel]
    
    def test_alert_deduplication(self):
        """Test alert deduplication."""
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=5,
            avg_compression_ratio_1h=0.7,
            token_savings_rate_1h=0.3,
            success_rate_1h=0.5,  # Low success rate
            performance_overhead_1h=0.2,
            system_health_score=85,
            resource_utilization=0.6
        )
        
        # Create alerts multiple times with same conditions
        self.alert_manager.check_and_create_alerts(metrics, {})
        initial_count = len(self.alert_manager.active_alerts)
        
        self.alert_manager.check_and_create_alerts(metrics, {})
        
        # Should not create duplicate alerts
        assert len(self.alert_manager.active_alerts) == initial_count


class TestRealTimeMetrics:
    """Test cases for RealTimeMetrics."""
    
    def test_real_time_metrics_creation(self):
        """Test RealTimeMetrics creation."""
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=3,
            avg_compression_ratio_1h=0.7,
            token_savings_rate_1h=0.3,
            success_rate_1h=0.9,
            performance_overhead_1h=0.2,
            system_health_score=85,
            resource_utilization=0.6
        )
        
        assert isinstance(metrics, RealTimeMetrics)
        assert metrics.timestamp == "2024-01-01T12:00:00"
        assert metrics.active_sessions == 3
        assert metrics.avg_compression_ratio_1h == 0.7
        assert metrics.token_savings_rate_1h == 0.3
        assert metrics.success_rate_1h == 0.9
        assert metrics.performance_overhead_1h == 0.2
        assert metrics.system_health_score == 85
        assert metrics.resource_utilization == 0.6
    
    def test_real_time_metrics_validation(self):
        """Test RealTimeMetrics validation."""
        # Test with valid values
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=0,  # Minimum valid value
            avg_compression_ratio_1h=0.0,  # Minimum valid value
            token_savings_rate_1h=1.0,  # Maximum valid value
            success_rate_1h=1.0,  # Maximum valid value
            performance_overhead_1h=0.0,  # Minimum valid value
            system_health_score=100,  # Maximum valid value
            resource_utilization=1.0  # Maximum valid value
        )
        
        assert isinstance(metrics, RealTimeMetrics)
        assert metrics.active_sessions >= 0
        assert 0 <= metrics.avg_compression_ratio_1h <= 1
        assert 0 <= metrics.token_savings_rate_1h <= 1
        assert 0 <= metrics.success_rate_1h <= 1
        assert metrics.performance_overhead_1h >= 0
        assert 0 <= metrics.system_health_score <= 100
        assert 0 <= metrics.resource_utilization <= 1


class TestAdvancedIsolationMetricsPerformance:
    """Performance tests for advanced isolation metrics."""
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                metrics = AdvancedResearcherIsolationMetrics(temp_file.name)
                metrics.monitoring_enabled = False
                
                # Test metrics collection speed
                start_time = time.time()
                for _ in range(10):
                    metrics._collect_real_time_metrics()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                assert avg_time < 0.1  # Should be fast (< 100ms per collection)
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
    
    def test_dashboard_generation_performance(self):
        """Test dashboard data generation performance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file.close()
            
            try:
                metrics = AdvancedResearcherIsolationMetrics(temp_file.name)
                metrics.monitoring_enabled = False
                
                # Add some metrics first
                metrics._collect_real_time_metrics()
                
                # Test dashboard generation speed
                start_time = time.time()
                for _ in range(5):
                    metrics.get_real_time_dashboard()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 5
                assert avg_time < 0.2  # Should be reasonably fast (< 200ms per generation)
                
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
    
    def test_alert_checking_performance(self):
        """Test alert checking performance."""
        alert_manager = AlertManager()
        
        metrics = RealTimeMetrics(
            timestamp="2024-01-01T12:00:00",
            active_sessions=5,
            avg_compression_ratio_1h=0.7,
            token_savings_rate_1h=0.3,
            success_rate_1h=0.5,
            performance_overhead_1h=0.2,
            system_health_score=85,
            resource_utilization=0.95
        )
        
        # Test alert checking speed
        start_time = time.time()
        for _ in range(100):
            alert_manager.check_and_create_alerts(metrics, {})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be very fast (< 10ms per check)