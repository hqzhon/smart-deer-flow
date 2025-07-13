"""Unit tests for advanced progressive enablement functionality."""

import pytest
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from utils.researcher_progressive_enablement_phase4 import (
    AdvancedResearcherProgressiveEnabler,
    AdvancedScenarioFeatures,
    DecisionConfidence,
    DynamicThreshold,
    PerformancePredictor
)
from utils.researcher_progressive_enablement import ScenarioContext


class TestAdvancedResearcherProgressiveEnabler:
    """Test cases for AdvancedResearcherProgressiveEnabler."""
    
    def setup_method(self):
        """Setup test environment."""
        self.enabler = AdvancedResearcherProgressiveEnabler()
    
    def test_enabler_instantiation(self):
        """Test enabler instantiation."""
        assert self.enabler is not None
        assert isinstance(self.enabler, AdvancedResearcherProgressiveEnabler)
        assert hasattr(self.enabler, 'performance_predictor')
        assert isinstance(self.enabler.performance_predictor, PerformancePredictor)
    
    def test_advanced_feature_extraction_complex_scenario(self):
        """Test advanced feature extraction for complex scenarios."""
        scenario = ScenarioContext(
            task_description="Analyze complex financial data with multiple correlations",
            step_count=8,
            context_size=15000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=15000
        )
        
        features = self.enabler.extract_advanced_features(scenario)
        
        assert isinstance(features, AdvancedScenarioFeatures)
        assert features.query_complexity_score > 0.2  # Should be complex
        assert features.data_diversity_score >= 0
        assert features.resource_pressure >= 0
        assert features.task_urgency_score >= 0
        assert 0 <= features.query_complexity_score <= 1
        assert 0 <= features.data_diversity_score <= 1
        assert 0 <= features.resource_pressure <= 1
        assert 0 <= features.task_urgency_score <= 1
    
    def test_advanced_feature_extraction_simple_scenario(self):
        """Test advanced feature extraction for simple scenarios."""
        scenario = ScenarioContext(
            task_description="Simple lookup",
            step_count=2,
            context_size=1000,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=1000
        )
        
        features = self.enabler.extract_advanced_features(scenario)
        
        assert isinstance(features, AdvancedScenarioFeatures)
        assert features.query_complexity_score < 0.5  # Should be simple
        assert features.data_diversity_score >= 0
        assert features.resource_pressure >= 0
        assert features.task_urgency_score >= 0
    
    def test_isolation_score_calculation_high_complexity(self):
        """Test isolation score calculation for high complexity scenarios."""
        scenario = ScenarioContext(
            task_description="Complex analysis task",
            step_count=8,
            context_size=15000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=15000
        )
        
        features = AdvancedScenarioFeatures(base_scenario=scenario)
        features.query_complexity_score = 0.9
        features.resource_pressure = 0.8
        features.task_urgency_score = 0.7
        features.data_diversity_score = 0.8
        
        score = self.enabler.calculate_isolation_score(features)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0.7  # Should recommend isolation for high complexity
    
    def test_isolation_score_calculation_low_complexity(self):
        """Test isolation score calculation for low complexity scenarios."""
        scenario = ScenarioContext(
            task_description="Simple lookup",
            step_count=2,
            context_size=1000,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=1000
        )
        
        features = AdvancedScenarioFeatures(base_scenario=scenario)
        features.query_complexity_score = 0.2
        features.resource_pressure = 0.3
        features.task_urgency_score = 0.2
        features.data_diversity_score = 0.3
        
        score = self.enabler.calculate_isolation_score(features)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score < 0.5  # Should not recommend isolation for low complexity
    
    def test_performance_prediction(self):
        """Test performance prediction functionality."""
        scenario = ScenarioContext(
            task_description="Medium complexity task",
            step_count=5,
            context_size=8000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=8000
        )
        
        features = AdvancedScenarioFeatures(base_scenario=scenario)
        features.query_complexity_score = 0.7
        features.resource_pressure = 0.6
        features.task_urgency_score = 0.4
        features.data_diversity_score = 0.6
        
        predicted_with, predicted_without = self.enabler.performance_predictor.predict(features, True)
        
        assert isinstance(predicted_with, (int, float))
        assert isinstance(predicted_without, (int, float))
        assert predicted_with > 0
        assert predicted_without > 0
    
    def test_decision_explanation(self):
        """Test decision explanation generation."""
        scenario = ScenarioContext(
            task_description="Simple data lookup",
            step_count=2,
            context_size=1000,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=1000
        )
        
        decision, explanation, factors = self.enabler.should_enable_isolation_with_explanation(scenario)
        
        assert isinstance(decision, bool)
        assert isinstance(explanation, str)
        assert isinstance(factors, dict)
        assert len(explanation) > 0
        assert 'isolation_score' in factors
        assert 'features' in factors
        assert 'decision_confidence' in factors
    
    def test_decision_explanation_complex_scenario(self):
        """Test decision explanation for complex scenarios."""
        scenario = ScenarioContext(
            task_description="Complex multi-step analysis with external dependencies",
            step_count=10,
            context_size=20000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=20000
        )
        
        decision, explanation, factors = self.enabler.should_enable_isolation_with_explanation(scenario)
        
        assert isinstance(decision, bool)
        assert isinstance(explanation, str)
        assert isinstance(factors, dict)
        assert len(explanation) > 0
        # For complex scenarios, decision should likely be True
        if decision:
            assert "recommend" in explanation.lower() or "enable" in explanation.lower()
    
    def test_auto_tuning_basic(self):
        """Test basic auto-tuning functionality."""
        # Simulate some historical data
        for i in range(10):
            scenario = ScenarioContext(
                task_description=f"Test task {i}",
                step_count=3 + i % 5,
                context_size=5000 + i * 1000,
                parallel_execution=i % 2 == 0,
                has_search_results=True,
                has_complex_queries=i % 3 == 0,
                estimated_tokens=5000 + i * 1000
            )
            
            # Record scenario outcome with execution time
            isolation_used = i % 2 == 0
            execution_time = 1.0 + i * 0.1
            
            self.enabler.record_scenario_outcome(scenario, isolation_used, execution_time)
        
        # Test optimization report
        report = self.enabler.get_optimization_report()
        
        assert isinstance(report, dict)
        assert "decision_count" in report
        assert "feature_weights" in report
        assert "dynamic_thresholds" in report
        assert isinstance(report["decision_count"], int)
        assert report["decision_count"] >= 0
    
    def test_scenario_outcome_recording(self):
        """Test scenario outcome recording."""
        scenario = ScenarioContext(
            task_description="Test recording",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=5000
        )
        
        # Record multiple outcomes
        for i in range(5):
            isolation_used = i % 2 == 0
            execution_time = 1.0 + i * 0.2
            
            self.enabler.record_scenario_outcome(scenario, isolation_used, execution_time)
        
        # Check that data was recorded
        report = self.enabler.get_optimization_report()
        assert report["decision_count"] >= 5
    
    def test_performance_predictor_edge_cases(self):
        """Test performance predictor with edge cases."""
        # Test with minimal features
        minimal_scenario = ScenarioContext(
            task_description="Minimal",
            step_count=1,
            context_size=100,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=100
        )
        
        minimal_features = AdvancedScenarioFeatures(base_scenario=minimal_scenario)
        minimal_features.query_complexity_score = 0.0
        minimal_features.resource_pressure = 0.0
        minimal_features.task_urgency_score = 0.0
        minimal_features.data_diversity_score = 0.0
        
        predicted_with, predicted_without = self.enabler.performance_predictor.predict(minimal_features, True)
        
        assert predicted_with >= 0
        assert predicted_without >= 0
        
        # Test with maximal features
        maximal_scenario = ScenarioContext(
            task_description="Maximal complexity task",
            step_count=20,
            context_size=50000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=50000
        )
        
        maximal_features = AdvancedScenarioFeatures(base_scenario=maximal_scenario)
        maximal_features.query_complexity_score = 1.0
        maximal_features.resource_pressure = 1.0
        maximal_features.task_urgency_score = 1.0
        maximal_features.data_diversity_score = 1.0
        
        predicted_with_max, predicted_without_max = self.enabler.performance_predictor.predict(maximal_features, True)
        
        assert predicted_with_max >= 0
        assert predicted_without_max >= 0
    
    def test_enabler_consistency(self):
        """Test enabler decision consistency."""
        scenario = ScenarioContext(
            task_description="Consistency test",
            step_count=5,
            context_size=8000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=8000
        )
        
        # Make multiple decisions for the same scenario
        decisions = []
        for _ in range(5):
            decision = self.enabler.should_enable_isolation(scenario)
            decisions.append(decision)
        
        # All decisions should be the same for the same scenario
        assert all(d == decisions[0] for d in decisions)
    
    def test_multiple_enabler_instances(self):
        """Test creating multiple enabler instances."""
        enabler1 = AdvancedResearcherProgressiveEnabler()
        enabler2 = AdvancedResearcherProgressiveEnabler()
        
        assert enabler1 is not enabler2
        assert isinstance(enabler1, AdvancedResearcherProgressiveEnabler)
        assert isinstance(enabler2, AdvancedResearcherProgressiveEnabler)
        
        # Both should work independently
        scenario = ScenarioContext(
            task_description="Multi-instance test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=5000
        )
        
        decision1 = enabler1.should_enable_isolation(scenario)
        decision2 = enabler2.should_enable_isolation(scenario)
        
        # Should produce same results for same scenario
        assert decision1 == decision2


class TestAdvancedScenarioFeatures:
    """Test cases for AdvancedScenarioFeatures."""
    
    def test_advanced_scenario_features_creation(self):
        """Test AdvancedScenarioFeatures creation."""
        scenario = ScenarioContext(
            task_description="Test scenario",
            step_count=5,
            context_size=8000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=8000
        )
        
        features = AdvancedScenarioFeatures(base_scenario=scenario)
        
        assert isinstance(features, AdvancedScenarioFeatures)
        assert hasattr(features, 'base_scenario')
        assert hasattr(features, 'query_complexity_score')
        assert hasattr(features, 'data_diversity_score')
        assert hasattr(features, 'resource_pressure')
        assert hasattr(features, 'task_urgency_score')
        assert features.base_scenario == scenario
    
    def test_advanced_scenario_features_attributes(self):
        """Test AdvancedScenarioFeatures attributes."""
        scenario = ScenarioContext(
            task_description="Attribute test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=5000
        )
        
        features = AdvancedScenarioFeatures(base_scenario=scenario)
        
        # Test setting attributes
        features.query_complexity_score = 0.7
        features.data_diversity_score = 0.5
        features.resource_pressure = 0.6
        features.task_urgency_score = 0.4
        
        assert features.query_complexity_score == 0.7
        assert features.data_diversity_score == 0.5
        assert features.resource_pressure == 0.6
        assert features.task_urgency_score == 0.4


class TestPerformancePredictor:
    """Test cases for PerformancePredictor."""
    
    def test_performance_predictor_instantiation(self):
        """Test PerformancePredictor instantiation."""
        predictor = PerformancePredictor()
        
        assert predictor is not None
        assert isinstance(predictor, PerformancePredictor)
    
    def test_performance_prediction_basic(self):
        """Test basic performance prediction."""
        predictor = PerformancePredictor()
        
        scenario = ScenarioContext(
            task_description="Prediction test",
            step_count=4,
            context_size=6000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=6000
        )
        
        features = AdvancedScenarioFeatures(base_scenario=scenario)
        features.query_complexity_score = 0.6
        features.resource_pressure = 0.5
        features.task_urgency_score = 0.3
        features.data_diversity_score = 0.4
        
        predicted_with, predicted_without = predictor.predict(features, True)
        
        assert isinstance(predicted_with, (int, float))
        assert isinstance(predicted_without, (int, float))
        assert predicted_with > 0
        assert predicted_without > 0
    
    def test_performance_prediction_consistency(self):
        """Test performance prediction consistency."""
        predictor = PerformancePredictor()
        
        scenario = ScenarioContext(
            task_description="Consistency test",
            step_count=3,
            context_size=4000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=4000
        )
        
        features = AdvancedScenarioFeatures(base_scenario=scenario)
        features.query_complexity_score = 0.5
        features.resource_pressure = 0.4
        features.task_urgency_score = 0.6
        features.data_diversity_score = 0.3
        
        # Make multiple predictions
        predictions = []
        for _ in range(3):
            predicted_with, predicted_without = predictor.predict(features, True)
            predictions.append((predicted_with, predicted_without))
        
        # All predictions should be the same for the same input
        assert all(p == predictions[0] for p in predictions)


class TestAdvancedProgressiveEnablementPerformance:
    """Performance tests for advanced progressive enablement."""
    
    def test_enabler_decision_performance(self):
        """Test enabler decision performance."""
        enabler = AdvancedResearcherProgressiveEnabler()
        
        scenario = ScenarioContext(
            task_description="Performance test",
            step_count=5,
            context_size=10000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=15000
        )
        
        # Test decision speed
        start_time = time.time()
        for _ in range(100):
            enabler.should_enable_isolation(scenario)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be fast (< 10ms per decision)
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance."""
        enabler = AdvancedResearcherProgressiveEnabler()
        
        scenario = ScenarioContext(
            task_description="Feature extraction performance test",
            step_count=8,
            context_size=12000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=12000
        )
        
        # Test feature extraction speed
        start_time = time.time()
        for _ in range(50):
            enabler.extract_advanced_features(scenario)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50
        assert avg_time < 0.005  # Should be very fast (< 5ms per extraction)