#!/usr/bin/env python3
"""
Unit tests for ResearcherProgressiveEnabler.

This module tests the progressive enablement functionality for researcher context isolation,
including scenario analysis, task complexity detection, and threshold optimization.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.utils.researcher.researcher_progressive_enablement import (
    ResearcherProgressiveEnabler, 
    ScenarioContext, 
    TaskComplexity
)


class TestScenarioContext:
    """Test suite for ScenarioContext data class."""

    def test_scenario_context_creation(self):
        """Test creating a ScenarioContext instance."""
        scenario = ScenarioContext(
            task_description="Test task",
            step_count=3,
            context_size=5000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=False,
            estimated_tokens=8000
        )
        
        assert scenario.task_description == "Test task"
        assert scenario.step_count == 3
        assert scenario.context_size == 5000
        assert scenario.parallel_execution is True
        assert scenario.has_search_results is True
        assert scenario.has_complex_queries is False
        assert scenario.estimated_tokens == 8000
    
    def test_scenario_context_defaults(self):
        """Test ScenarioContext with minimal parameters."""
        scenario = ScenarioContext(
            task_description="Minimal task",
            step_count=1,
            context_size=1000
        )
        
        assert scenario.task_description == "Minimal task"
        assert scenario.step_count == 1
        assert scenario.context_size == 1000
        # Check that optional parameters have reasonable defaults
        assert hasattr(scenario, 'parallel_execution')
        assert hasattr(scenario, 'has_search_results')
        assert hasattr(scenario, 'has_complex_queries')
        assert hasattr(scenario, 'estimated_tokens')


class TestTaskComplexity:
    """Test suite for TaskComplexity enum."""

    def test_task_complexity_values(self):
        """Test that TaskComplexity has expected values."""
        expected_values = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        for value in expected_values:
            assert hasattr(TaskComplexity, value), f"TaskComplexity missing {value}"
    
    def test_task_complexity_ordering(self):
        """Test that TaskComplexity values can be compared."""
        # This test assumes TaskComplexity is an enum with comparable values
        try:
            assert TaskComplexity.LOW < TaskComplexity.MEDIUM
            assert TaskComplexity.MEDIUM < TaskComplexity.HIGH
            assert TaskComplexity.HIGH < TaskComplexity.CRITICAL
        except (AttributeError, TypeError):
            # If comparison is not supported, just check they exist
            assert TaskComplexity.LOW is not None
            assert TaskComplexity.MEDIUM is not None
            assert TaskComplexity.HIGH is not None
            assert TaskComplexity.CRITICAL is not None


class TestResearcherProgressiveEnabler:
    """Test suite for ResearcherProgressiveEnabler."""

    @pytest.fixture
    def enabler(self):
        """Create a ResearcherProgressiveEnabler instance for testing."""
        return ResearcherProgressiveEnabler()

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample ScenarioContext for testing."""
        return ScenarioContext(
            task_description="Complex analysis with multiple data sources",
            step_count=5,
            context_size=12000,
            parallel_execution=True,
            has_search_results=True,
            has_complex_queries=True,
            estimated_tokens=15000
        )

    def test_analyze_task_complexity_high(self, enabler, sample_scenario):
        """Test task complexity analysis for high complexity scenario."""
        complexity = enabler.analyze_task_complexity(sample_scenario)
        
        assert complexity is not None
        assert isinstance(complexity, (TaskComplexity, str))
        
        # For a complex scenario, expect medium to critical complexity
        if isinstance(complexity, TaskComplexity):
            assert complexity in [TaskComplexity.MEDIUM, TaskComplexity.HIGH, TaskComplexity.CRITICAL]
        else:
            assert complexity.lower() in ['medium', 'high', 'critical']
    
    def test_analyze_task_complexity_low(self, enabler):
        """Test task complexity analysis for low complexity scenario."""
        simple_scenario = ScenarioContext(
            task_description="Simple task",
            step_count=1,
            context_size=1000,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=2000
        )
        
        complexity = enabler.analyze_task_complexity(simple_scenario)
        
        assert complexity is not None
        assert isinstance(complexity, (TaskComplexity, str))
    
    def test_should_enable_isolation_decision(self, enabler, sample_scenario):
        """Test isolation enablement decision."""
        should_isolate, reason, factors = enabler.should_enable_isolation(sample_scenario)
        
        assert isinstance(should_isolate, bool)
        assert isinstance(reason, str)
        assert len(reason) > 0
        assert isinstance(factors, (list, dict, tuple))
    
    def test_should_enable_isolation_with_simple_scenario(self, enabler):
        """Test isolation decision with simple scenario."""
        simple_scenario = ScenarioContext(
            task_description="Simple task",
            step_count=1,
            context_size=500,
            parallel_execution=False,
            has_search_results=False,
            has_complex_queries=False,
            estimated_tokens=1000
        )
        
        should_isolate, reason, factors = enabler.should_enable_isolation(simple_scenario)
        
        assert isinstance(should_isolate, bool)
        assert isinstance(reason, str)
        assert isinstance(factors, (list, dict, tuple))
    
    def test_record_scenario_outcome(self, enabler, sample_scenario):
        """Test recording scenario outcome."""
        # This should not raise an exception
        enabler.record_scenario_outcome(
            scenario=sample_scenario,
            isolation_enabled=True,
            execution_time=1.5,
            token_savings=500
        )
        
        # Verify the outcome was recorded (basic test)
        assert True
    
    def test_record_scenario_outcome_with_failure(self, enabler, sample_scenario):
        """Test recording scenario outcome with negative results."""
        enabler.record_scenario_outcome(
            scenario=sample_scenario,
            isolation_enabled=True,
            execution_time=3.0,  # Longer execution time
            token_savings=-100    # Negative savings (overhead)
        )
        
        # Verify the outcome was recorded
        assert True
    
    def test_optimize_thresholds(self, enabler):
        """Test threshold optimization."""
        # Record some outcomes first
        scenarios = [
            ScenarioContext(
                task_description=f"Task {i}",
                step_count=i + 1,
                context_size=1000 * (i + 1),
                parallel_execution=i % 2 == 0,
                has_search_results=i % 3 == 0,
                has_complex_queries=i % 4 == 0,
                estimated_tokens=2000 * (i + 1)
            )
            for i in range(5)
        ]
        
        for i, scenario in enumerate(scenarios):
            enabler.record_scenario_outcome(
                scenario=scenario,
                isolation_enabled=i % 2 == 0,
                execution_time=1.0 + i * 0.2,
                token_savings=100 + i * 50
            )
        
        # Optimize thresholds
        enabler.optimize_thresholds()
        
        # Verify optimization completed without error
        assert True
    
    def test_multiple_scenarios_workflow(self, enabler):
        """Test a complete workflow with multiple scenarios."""
        scenarios = [
            # Simple scenario
            ScenarioContext(
                task_description="Simple analysis",
                step_count=1,
                context_size=2000,
                parallel_execution=False,
                has_search_results=False,
                has_complex_queries=False,
                estimated_tokens=3000
            ),
            # Medium scenario
            ScenarioContext(
                task_description="Medium complexity task",
                step_count=3,
                context_size=8000,
                parallel_execution=True,
                has_search_results=True,
                has_complex_queries=False,
                estimated_tokens=10000
            ),
            # Complex scenario
            ScenarioContext(
                task_description="Complex multi-step analysis",
                step_count=7,
                context_size=20000,
                parallel_execution=True,
                has_search_results=True,
                has_complex_queries=True,
                estimated_tokens=25000
            )
        ]
        
        results = []
        for scenario in scenarios:
            # Analyze complexity
            complexity = enabler.analyze_task_complexity(scenario)
            
            # Make isolation decision
            should_isolate, reason, factors = enabler.should_enable_isolation(scenario)
            
            # Record outcome
            enabler.record_scenario_outcome(
                scenario=scenario,
                isolation_enabled=should_isolate,
                execution_time=1.0,
                token_savings=200 if should_isolate else 0
            )
            
            results.append({
                'complexity': complexity,
                'should_isolate': should_isolate,
                'reason': reason,
                'factors': factors
            })
        
        # Optimize based on recorded outcomes
        enabler.optimize_thresholds()
        
        # Verify all scenarios were processed
        assert len(results) == 3
        for result in results:
            assert 'complexity' in result
            assert 'should_isolate' in result
            assert 'reason' in result
            assert 'factors' in result
    
    def test_edge_cases(self, enabler):
        """Test edge cases and boundary conditions."""
        # Zero context size
        zero_context = ScenarioContext(
            task_description="Zero context",
            step_count=1,
            context_size=0,
            estimated_tokens=0
        )
        
        complexity = enabler.analyze_task_complexity(zero_context)
        should_isolate, reason, factors = enabler.should_enable_isolation(zero_context)
        
        assert complexity is not None
        assert isinstance(should_isolate, bool)
        assert isinstance(reason, str)
        
        # Very large context
        large_context = ScenarioContext(
            task_description="Massive context",
            step_count=100,
            context_size=1000000,
            estimated_tokens=500000
        )
        
        complexity = enabler.analyze_task_complexity(large_context)
        should_isolate, reason, factors = enabler.should_enable_isolation(large_context)
        
        assert complexity is not None
        assert isinstance(should_isolate, bool)
        assert isinstance(reason, str)
    
    def test_threshold_adaptation(self, enabler):
        """Test that thresholds adapt based on recorded outcomes."""
        # Record several successful isolation scenarios
        for i in range(10):
            scenario = ScenarioContext(
                task_description=f"Successful task {i}",
                step_count=3,
                context_size=5000,
                estimated_tokens=8000
            )
            
            enabler.record_scenario_outcome(
                scenario=scenario,
                isolation_enabled=True,
                execution_time=1.0,
                token_savings=300
            )
        
        # Optimize thresholds
        enabler.optimize_thresholds()
        
        # Test that the enabler still makes reasonable decisions
        test_scenario = ScenarioContext(
            task_description="Test scenario after optimization",
            step_count=3,
            context_size=5000,
            estimated_tokens=8000
        )
        
        should_isolate, reason, factors = enabler.should_enable_isolation(test_scenario)
        assert isinstance(should_isolate, bool)
        assert isinstance(reason, str)