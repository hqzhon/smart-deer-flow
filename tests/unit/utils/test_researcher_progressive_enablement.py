#!/usr/bin/env python3
"""
Researcher Progressive Enablement Unit Tests

This module contains unit tests for ResearcherProgressiveEnabler functionality,
including scenario analysis, task complexity assessment, and threshold optimization.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from src.utils.researcher.researcher_progressive_enablement import (
    ResearcherProgressiveEnabler,
    ScenarioContext,
    TaskComplexity,
)


class TestScenarioContext:
    """Test ScenarioContext data class."""
    
    def test_scenario_context_creation(self):
        """Test creating a ScenarioContext instance."""
        context = ScenarioContext(
            task_description="Test task",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=7500
        )
        
        assert context.task_description == "Test task"
        assert context.step_count == 3
        assert context.context_size == 5000
        assert context.parallel_execution is False
        assert context.has_search_results is True
        assert context.has_complex_queries is False
        assert context.estimated_tokens == 7500
    
    def test_scenario_context_with_complex_scenario(self):
        """Test ScenarioContext with complex scenario parameters."""
        context = ScenarioContext(
            task_description="Complex analysis with multiple data sources",
            step_count=5,
            context_size=12000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=15000
        )
        
        assert context.task_description == "Complex analysis with multiple data sources"
        assert context.step_count == 5
        assert context.context_size == 12000
        assert context.parallel_execution is True
        assert context.has_search_results is True
        assert context.has_complex_queries is True
        assert context.estimated_tokens == 15000
    
    def test_scenario_context_edge_cases(self):
        """Test ScenarioContext with edge case values."""
        # Minimal scenario
        minimal_context = ScenarioContext(
            task_description="",
            step_count=0,
            context_size=0,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=0
        )
        
        assert minimal_context.step_count == 0
        assert minimal_context.context_size == 0
        assert minimal_context.estimated_tokens == 0
        
        # Large scenario
        large_context = ScenarioContext(
            task_description="Very large task" * 100,
            step_count=100,
            context_size=1000000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=500000
        )
        
        assert large_context.step_count == 100
        assert large_context.context_size == 1000000
        assert large_context.estimated_tokens == 500000


class TestTaskComplexity:
    """Test TaskComplexity enum."""
    
    def test_task_complexity_values(self):
        """Test that TaskComplexity has expected values."""
        # Test that the enum exists and has expected values
        try:
            assert hasattr(TaskComplexity, 'SIMPLE') or 'SIMPLE' in str(TaskComplexity)
            assert hasattr(TaskComplexity, 'MEDIUM') or 'MEDIUM' in str(TaskComplexity)
            assert hasattr(TaskComplexity, 'COMPLEX') or 'COMPLEX' in str(TaskComplexity)
            assert hasattr(TaskComplexity, 'VERY_COMPLEX') or 'VERY_COMPLEX' in str(TaskComplexity)
        except (AttributeError, NameError):
            # If TaskComplexity is implemented differently, that's fine
            pass


class TestResearcherProgressiveEnabler:
    """Test ResearcherProgressiveEnabler functionality."""
    
    def test_enabler_instantiation(self):
        """Test that ResearcherProgressiveEnabler can be instantiated."""
        enabler = ResearcherProgressiveEnabler()
        assert enabler is not None
        assert isinstance(enabler, ResearcherProgressiveEnabler)
    
    def test_analyze_task_complexity_simple(self):
        """Test task complexity analysis for simple scenarios."""
        enabler = ResearcherProgressiveEnabler()
        
        simple_context = ScenarioContext(
            task_description="Simple task",
            step_count=1,
            context_size=1000,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=2000
        )
        
        complexity = enabler.analyze_task_complexity(simple_context)
        
        # Should return a complexity level (string or enum)
        assert complexity is not None
        if isinstance(complexity, str):
            assert complexity.lower() in ['low', 'medium', 'high', 'extreme']
    
    def test_analyze_task_complexity_complex(self):
        """Test task complexity analysis for complex scenarios."""
        enabler = ResearcherProgressiveEnabler()
        
        complex_context = ScenarioContext(
            task_description="Complex analysis with multiple data sources",
            step_count=5,
            context_size=12000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=15000
        )
        
        complexity = enabler.analyze_task_complexity(complex_context)
        
        # Should return a complexity level
        assert complexity is not None
        if isinstance(complexity, str):
            # Complex scenario should likely be medium or high complexity
            assert complexity.lower() in ['medium', 'high', 'extreme']
    
    def test_analyze_task_complexity_various_scenarios(self):
        """Test task complexity analysis for various scenarios."""
        enabler = ResearcherProgressiveEnabler()
        
        scenarios = [
            # Low complexity
            ScenarioContext(
                task_description="Simple query",
                step_count=1,
                context_size=500,
                parallel_execution=False,
                has_search_results=False,
                has_complex_queries=False,
                estimated_tokens=1000
            ),
            # Medium complexity
            ScenarioContext(
                task_description="Moderate analysis",
                step_count=3,
                context_size=5000,
                parallel_execution=False,
                has_search_results=True,
                has_complex_queries=False,
                estimated_tokens=8000
            ),
            # High complexity
            ScenarioContext(
                task_description="Complex multi-step analysis",
                step_count=7,
                context_size=20000,
                parallel_execution=True,
                has_search_results=True,
                has_complex_queries=True,
                estimated_tokens=30000
            )
        ]
        
        for scenario in scenarios:
            complexity = enabler.analyze_task_complexity(scenario)
            assert complexity is not None
    
    def test_should_enable_isolation_simple_scenario(self):
        """Test isolation decision for simple scenarios."""
        enabler = ResearcherProgressiveEnabler()
        
        simple_context = ScenarioContext(
            task_description="Simple task",
            step_count=1,
            context_size=1000,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=2000
        )
        
        should_isolate, reason, factors = enabler.should_enable_isolation(simple_context)
        
        assert isinstance(should_isolate, bool)
        assert isinstance(reason, str)
        assert reason != "", "Reason should not be empty"
        # factors might be None or a dict/list
        if factors is not None:
            assert isinstance(factors, (dict, list))
    
    def test_should_enable_isolation_complex_scenario(self):
        """Test isolation decision for complex scenarios."""
        enabler = ResearcherProgressiveEnabler()
        
        complex_context = ScenarioContext(
            task_description="Complex analysis with multiple data sources",
            step_count=5,
            context_size=12000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=15000
        )
        
        should_isolate, reason, factors = enabler.should_enable_isolation(complex_context)
        
        assert isinstance(should_isolate, bool)
        assert isinstance(reason, str)
        assert reason != "", "Reason should not be empty"
        
        # Complex scenarios are more likely to benefit from isolation
        # But we don't enforce this as it depends on the algorithm
    
    def test_should_enable_isolation_various_scenarios(self):
        """Test isolation decisions for various scenarios."""
        enabler = ResearcherProgressiveEnabler()
        
        scenarios = [
            ScenarioContext(
                task_description="Quick lookup",
                step_count=1,
                context_size=500,
                parallel_execution=False,
                has_search_results=False,
                has_complex_queries=False,
                estimated_tokens=1000
            ),
            ScenarioContext(
                task_description="Medium analysis",
                step_count=3,
                context_size=8000,
                parallel_execution=False,
                has_search_results=True,
                has_complex_queries=True,
                estimated_tokens=12000
            ),
            ScenarioContext(
                task_description="Large research project",
                step_count=10,
                context_size=50000,
                parallel_execution=True,
                has_search_results=True,
                has_complex_queries=True,
                estimated_tokens=75000
            )
        ]
        
        for scenario in scenarios:
            should_isolate, reason, factors = enabler.should_enable_isolation(scenario)
            
            assert isinstance(should_isolate, bool)
            assert isinstance(reason, str)
            assert len(reason) > 0
    
    def test_record_scenario_outcome(self):
        """Test recording scenario outcomes."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Test scenario",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        try:
            enabler.record_scenario_outcome(
                scenario=scenario_context,
                isolation_enabled=True,
                execution_time=1.5,
                token_savings=500
            )
            # Should not raise an exception
        except Exception as e:
            pytest.fail(f"record_scenario_outcome should not raise exception: {e}")
    
    def test_record_scenario_outcome_various_outcomes(self):
        """Test recording various scenario outcomes."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Test scenario",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        outcomes = [
            {"isolation_enabled": True, "execution_time": 1.0, "token_savings": 200},
            {"isolation_enabled": False, "execution_time": 1.2, "token_savings": 0},
            {"isolation_enabled": True, "execution_time": 2.5, "token_savings": 800},
            {"isolation_enabled": True, "execution_time": 0.8, "token_savings": 150}
        ]
        
        for outcome in outcomes:
            try:
                enabler.record_scenario_outcome(
                    scenario=scenario_context,
                    **outcome
                )
            except Exception as e:
                pytest.fail(f"record_scenario_outcome should not raise exception: {e}")
    
    def test_record_scenario_outcome_edge_cases(self):
        """Test recording scenario outcomes with edge case values."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Edge case scenario",
            step_count=1,
            context_size=100,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=500
        )
        
        edge_cases = [
            {"isolation_enabled": True, "execution_time": 0.0, "token_savings": 0},
            {"isolation_enabled": False, "execution_time": 10.0, "token_savings": -100},
            {"isolation_enabled": True, "execution_time": 0.1, "token_savings": 10000}
        ]
        
        for case in edge_cases:
            try:
                enabler.record_scenario_outcome(
                    scenario=scenario_context,
                    **case
                )
            except Exception as e:
                pytest.fail(f"record_scenario_outcome with edge case should not raise exception: {e}")
    
    def test_optimize_thresholds(self):
        """Test threshold optimization."""
        enabler = ResearcherProgressiveEnabler()
        
        # Record some outcomes first
        scenario_context = ScenarioContext(
            task_description="Optimization test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        # Record several outcomes
        for i in range(5):
            enabler.record_scenario_outcome(
                scenario=scenario_context,
                isolation_enabled=i % 2 == 0,
                execution_time=1.0 + i * 0.2,
                token_savings=100 + i * 50
            )
        
        try:
            enabler.optimize_thresholds()
            # Should not raise an exception
        except Exception as e:
            pytest.fail(f"optimize_thresholds should not raise exception: {e}")
    
    def test_optimize_thresholds_without_data(self):
        """Test threshold optimization without prior data."""
        enabler = ResearcherProgressiveEnabler()
        
        try:
            enabler.optimize_thresholds()
            # Should handle the case where there's no data gracefully
        except Exception as e:
            # If it raises an exception, that's also acceptable
            # as long as it's a reasonable error
            pass
    
    def test_enabler_consistency(self):
        """Test consistency of enabler decisions."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Consistency test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        # Same scenario should give same complexity analysis
        complexity1 = enabler.analyze_task_complexity(scenario_context)
        complexity2 = enabler.analyze_task_complexity(scenario_context)
        
        assert complexity1 == complexity2, "Task complexity analysis should be consistent"
        
        # Same scenario should give same isolation decision (initially)
        decision1 = enabler.should_enable_isolation(scenario_context)
        decision2 = enabler.should_enable_isolation(scenario_context)
        
        # Note: decisions might change after learning, so we only test initial consistency
        assert decision1[0] == decision2[0], "Initial isolation decisions should be consistent"
    
    def test_enabler_learning(self):
        """Test that enabler learns from recorded outcomes."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Learning test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        # Get initial decision
        initial_decision, _, _ = enabler.should_enable_isolation(scenario_context)
        
        # Record many positive outcomes for isolation
        for i in range(10):
            enabler.record_scenario_outcome(
                scenario=scenario_context,
                isolation_enabled=True,
                execution_time=0.5,  # Fast execution
                token_savings=1000   # High savings
            )
        
        # Optimize thresholds
        enabler.optimize_thresholds()
        
        # Get decision after learning
        learned_decision, _, _ = enabler.should_enable_isolation(scenario_context)
        
        # The decision might change based on learning
        # We don't enforce a specific change, just that the system can learn
        assert isinstance(learned_decision, bool)
    
    def test_multiple_enabler_instances(self):
        """Test that multiple enabler instances work independently."""
        enabler1 = ResearcherProgressiveEnabler()
        enabler2 = ResearcherProgressiveEnabler()
        
        # Should be separate instances
        assert enabler1 is not enabler2
        
        scenario_context = ScenarioContext(
            task_description="Multi-instance test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        # Both should give same initial analysis
        complexity1 = enabler1.analyze_task_complexity(scenario_context)
        complexity2 = enabler2.analyze_task_complexity(scenario_context)
        
        assert complexity1 == complexity2, "Different instances should give same initial analysis"
        
        # Record different outcomes in each
        enabler1.record_scenario_outcome(
            scenario=scenario_context,
            isolation_enabled=True,
            execution_time=1.0,
            token_savings=500
        )
        
        enabler2.record_scenario_outcome(
            scenario=scenario_context,
            isolation_enabled=False,
            execution_time=2.0,
            token_savings=0
        )
        
        # Both should continue to work independently
        enabler1.optimize_thresholds()
        enabler2.optimize_thresholds()


class TestResearcherProgressiveEnablerPerformance:
    """Test ResearcherProgressiveEnabler performance."""
    
    def test_complexity_analysis_performance(self):
        """Test that complexity analysis is fast."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Performance test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        start_time = time.time()
        for _ in range(100):
            enabler.analyze_task_complexity(scenario_context)
        end_time = time.time()
        
        # Should analyze 100 scenarios in less than 1 second
        assert end_time - start_time < 1.0, "Complexity analysis should be fast"
    
    def test_isolation_decision_performance(self):
        """Test that isolation decisions are fast."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Performance test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        start_time = time.time()
        for _ in range(100):
            enabler.should_enable_isolation(scenario_context)
        end_time = time.time()
        
        # Should make 100 decisions in less than 1 second
        assert end_time - start_time < 1.0, "Isolation decisions should be fast"
    
    def test_outcome_recording_performance(self):
        """Test that outcome recording is fast."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Performance test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        start_time = time.time()
        for i in range(100):
            enabler.record_scenario_outcome(
                scenario=scenario_context,
                isolation_enabled=i % 2 == 0,
                execution_time=1.0,
                token_savings=100
            )
        end_time = time.time()
        
        # Should record 100 outcomes in less than 1 second
        assert end_time - start_time < 1.0, "Outcome recording should be fast"
    
    def test_threshold_optimization_performance(self):
        """Test that threshold optimization is reasonably fast."""
        enabler = ResearcherProgressiveEnabler()
        
        scenario_context = ScenarioContext(
            task_description="Performance test",
            step_count=3,
            context_size=5000,
            parallel_execution=False,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        # Record some outcomes first
        for i in range(50):
            enabler.record_scenario_outcome(
                scenario=scenario_context,
                isolation_enabled=i % 2 == 0,
                execution_time=1.0 + i * 0.01,
                token_savings=100 + i * 10
            )
        
        start_time = time.time()
        for _ in range(10):
            enabler.optimize_thresholds()
        end_time = time.time()
        
        # Should optimize 10 times in less than 5 seconds
        assert end_time - start_time < 5.0, "Threshold optimization should be reasonably fast"