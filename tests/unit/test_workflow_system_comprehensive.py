#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Comprehensive Unit Tests for Workflow System

This test suite provides comprehensive coverage for the workflow system
integration in DeerFlow, replacing workflow-related demo scripts with proper unit tests.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Workflow system components
from src.workflow.reflection_workflow import (
    ReflectionWorkflow, WorkflowStage, WorkflowResult, WorkflowMetrics
)
from src.utils.performance.workflow_optimizer import WorkflowOptimizer
from src.workflow.execution_context_manager import ExecutionContextManager

# Graph and state components
from src.graph.types import State
from src.graph.nodes import researcher_node_with_isolation, planner_node

# Reflection and context components
from src.utils.reflection.enhanced_reflection import EnhancedReflectionAgent
from src.utils.researcher.researcher_progressive_enablement import ResearcherProgressiveEnabler
from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.monitoring.researcher_isolation_metrics import ResearcherIsolationMetrics

# Configuration
from src.config.config_manager import ConfigManager


class TestReflectionWorkflow:
    """Test ReflectionWorkflow functionality."""
    
    @pytest.fixture
    def mock_workflow_config(self):
        """Create mock workflow configuration."""
        return {
            "enabled": True,
            "max_loops": 3,
            "confidence_threshold": 0.7,
            "enable_progressive": True,
            "enable_context_expansion": True,
            "enable_isolation_metrics": True,
            "stages": [
                {
                    "name": "context_analysis",
                    "description": "Analyze research context",
                    "requires_reflection": True,
                    "isolation_level": "low",
                    "timeout": 30
                },
                {
                    "name": "research_planning",
                    "description": "Plan research approach",
                    "requires_reflection": True,
                    "isolation_level": "moderate",
                    "timeout": 60
                },
                {
                    "name": "information_gathering",
                    "description": "Gather relevant information",
                    "requires_reflection": False,
                    "isolation_level": "high",
                    "timeout": 120
                },
                {
                    "name": "synthesis",
                    "description": "Synthesize findings",
                    "requires_reflection": True,
                    "isolation_level": "moderate",
                    "timeout": 90
                },
                {
                    "name": "validation",
                    "description": "Validate results",
                    "requires_reflection": True,
                    "isolation_level": "low",
                    "timeout": 45
                }
            ]
        }
    
    @pytest.fixture
    def reflection_workflow(self, mock_workflow_config):
        """Create ReflectionWorkflow instance."""
        return ReflectionWorkflow(mock_workflow_config)
    
    def test_workflow_initialization(self, reflection_workflow, mock_workflow_config):
        """Test ReflectionWorkflow initialization."""
        assert reflection_workflow.config == mock_workflow_config
        assert hasattr(reflection_workflow, 'stages')
        assert hasattr(reflection_workflow, 'reflection_agent')
        assert hasattr(reflection_workflow, 'progressive_enabler')
        assert hasattr(reflection_workflow, 'context_expansion')
        assert hasattr(reflection_workflow, 'isolation_metrics')
        
        # Verify stages are properly configured
        assert len(reflection_workflow.stages) == 5
        for stage in reflection_workflow.stages:
            assert isinstance(stage, WorkflowStage)
            assert hasattr(stage, 'name')
            assert hasattr(stage, 'description')
            assert hasattr(stage, 'requires_reflection')
            assert hasattr(stage, 'isolation_level')
    
    def test_workflow_stage_creation(self, mock_workflow_config):
        """Test workflow stage creation from configuration."""
        workflow = ReflectionWorkflow(mock_workflow_config)
        
        expected_stage_names = [
            "context_analysis", "research_planning", "information_gathering",
            "synthesis", "validation"
        ]
        
        stage_names = [stage.name for stage in workflow.stages]
        assert stage_names == expected_stage_names
        
        # Test specific stage properties
        context_stage = workflow.stages[0]
        assert context_stage.name == "context_analysis"
        assert context_stage.requires_reflection is True
        assert context_stage.isolation_level == "low"
        assert context_stage.timeout == 30
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, reflection_workflow):
        """Test complete workflow execution."""
        research_query = "Impact of machine learning on software testing efficiency"
        initial_context = {
            "domain": "software_engineering",
            "focus": "testing_automation",
            "technologies": ["ml", "automated_testing", "ci_cd"]
        }
        
        with patch.object(reflection_workflow, '_execute_stage') as mock_execute:
            # Mock stage execution results
            mock_execute.side_effect = [
                {
                    "success": True,
                    "stage_metrics": {
                        "execution_time": 1.2,
                        "reflection_applied": True,
                        "context_expanded": True,
                        "tokens_used": 500
                    },
                    "stage_data": {
                        "context_analysis": "Identified key research areas",
                        "confidence_score": 0.8
                    }
                },
                {
                    "success": True,
                    "stage_metrics": {
                        "execution_time": 2.1,
                        "reflection_applied": True,
                        "context_expanded": False,
                        "tokens_used": 750
                    },
                    "stage_data": {
                        "research_plan": "Systematic literature review approach",
                        "confidence_score": 0.85
                    }
                },
                {
                    "success": True,
                    "stage_metrics": {
                        "execution_time": 3.5,
                        "reflection_applied": False,
                        "context_expanded": True,
                        "tokens_used": 1200
                    },
                    "stage_data": {
                        "information_gathered": "Found 25 relevant studies",
                        "confidence_score": 0.9
                    }
                },
                {
                    "success": True,
                    "stage_metrics": {
                        "execution_time": 2.8,
                        "reflection_applied": True,
                        "context_expanded": True,
                        "tokens_used": 900
                    },
                    "stage_data": {
                        "synthesis_results": "ML improves testing efficiency by 40%",
                        "confidence_score": 0.88
                    }
                },
                {
                    "success": True,
                    "stage_metrics": {
                        "execution_time": 1.8,
                        "reflection_applied": True,
                        "context_expanded": False,
                        "tokens_used": 600
                    },
                    "stage_data": {
                        "validation_results": "Results validated against industry benchmarks",
                        "confidence_score": 0.92
                    }
                }
            ]
            
            result = await reflection_workflow.execute_workflow(research_query, initial_context)
            
            # Verify workflow result
            assert isinstance(result, WorkflowResult)
            assert result.success is True
            assert result.execution_time > 0
            assert len(result.stage_results) == 5
            assert hasattr(result, 'reflection_insights')
            assert hasattr(result, 'metrics')
            
            # Verify stage execution was called for each stage
            assert mock_execute.call_count == 5
    
    @pytest.mark.asyncio
    async def test_workflow_stage_execution(self, reflection_workflow):
        """Test individual stage execution."""
        stage = reflection_workflow.stages[0]  # context_analysis
        context = {
            "research_query": "Test query",
            "domain": "test_domain"
        }
        
        with patch.object(reflection_workflow.reflection_agent, 'analyze_knowledge_gaps') as mock_analyze:
            with patch.object(reflection_workflow.progressive_enabler, 'should_enable_reflection') as mock_should_enable:
                with patch.object(reflection_workflow.context_expansion, 'expand_context') as mock_expand:
                    # Mock reflection analysis
                    mock_analyze.return_value = Mock(
                        is_sufficient=True,
                        confidence_score=0.8,
                        knowledge_gaps=["Minor gap in recent studies"],
                        follow_up_queries=["What are latest developments?"]
                    )
                    
                    # Mock progressive enabler
                    mock_should_enable.return_value = True
                    
                    # Mock context expansion
                    mock_expand.return_value = {
                        "expanded_context": "Additional context information",
                        "expansion_metrics": {"tokens_added": 200}
                    }
                    
                    result = await reflection_workflow._execute_stage(stage, context)
                    
                    assert result["success"] is True
                    assert "stage_metrics" in result
                    assert "stage_data" in result
                    assert result["stage_metrics"]["reflection_applied"] is True
    
    def test_workflow_status_tracking(self, reflection_workflow):
        """Test workflow status tracking."""
        # Initial status
        status = reflection_workflow.get_workflow_status()
        
        assert isinstance(status, dict)
        assert "current_stage" in status
        assert "total_stages" in status
        assert "reflection_enabled" in status
        assert "active_sessions" in status
        assert "execution_time" in status
        
        # Verify initial values
        assert status["current_stage"] == 0
        assert status["total_stages"] == 5
        assert status["reflection_enabled"] is True
    
    def test_workflow_reset(self, reflection_workflow):
        """Test workflow reset functionality."""
        # Simulate some workflow progress
        reflection_workflow._current_stage = 2
        reflection_workflow._start_time = time.time() - 100
        
        # Reset workflow
        reflection_workflow.reset_workflow()
        
        # Verify reset state
        status = reflection_workflow.get_workflow_status()
        assert status["current_stage"] == 0
        assert status["execution_time"] == 0
    
    def test_workflow_metrics_collection(self, reflection_workflow):
        """Test workflow metrics collection."""
        # Simulate stage completion
        stage_metrics = {
            "execution_time": 2.5,
            "reflection_applied": True,
            "context_expanded": True,
            "tokens_used": 800,
            "confidence_score": 0.85
        }
        
        reflection_workflow._record_stage_metrics("context_analysis", stage_metrics)
        
        # Get workflow metrics
        metrics = reflection_workflow.get_workflow_metrics()
        
        assert isinstance(metrics, WorkflowMetrics)
        assert hasattr(metrics, 'total_execution_time')
        assert hasattr(metrics, 'stages_completed')
        assert hasattr(metrics, 'reflection_applications')
        assert hasattr(metrics, 'total_tokens_used')
        assert hasattr(metrics, 'average_confidence')
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, reflection_workflow):
        """Test workflow error handling."""
        research_query = "Test query"
        initial_context = {"domain": "test"}
        
        with patch.object(reflection_workflow, '_execute_stage') as mock_execute:
            # Simulate stage failure
            mock_execute.side_effect = [
                {"success": True, "stage_metrics": {}, "stage_data": {}},
                Exception("Stage execution failed"),
                {"success": True, "stage_metrics": {}, "stage_data": {}}
            ]
            
            result = await reflection_workflow.execute_workflow(research_query, initial_context)
            
            # Workflow should handle the error gracefully
            assert isinstance(result, WorkflowResult)
            assert result.success is False
            assert "error" in result.error_details
    
    def test_workflow_configuration_validation(self):
        """Test workflow configuration validation."""
        # Test valid configuration
        valid_config = {
            "enabled": True,
            "max_loops": 3,
            "confidence_threshold": 0.7,
            "stages": [
                {
                    "name": "test_stage",
                    "description": "Test stage",
                    "requires_reflection": True,
                    "isolation_level": "moderate",
                    "timeout": 30
                }
            ]
        }
        
        workflow = ReflectionWorkflow(valid_config)
        assert len(workflow.stages) == 1
        
        # Test invalid configuration
        invalid_config = {
            "enabled": True,
            "max_loops": -1,  # Invalid
            "confidence_threshold": 2.0,  # Invalid
            "stages": []
        }
        
        with pytest.raises((ValueError, AssertionError)):
            ReflectionWorkflow(invalid_config)


class TestWorkflowOptimizer:
    """Test WorkflowOptimizer functionality."""
    
    @pytest.fixture
    def workflow_optimizer(self):
        """Create WorkflowOptimizer instance."""
        return WorkflowOptimizer()
    
    @pytest.fixture
    def sample_workflow_data(self):
        """Create sample workflow execution data."""
        return {
            "stages": [
                {
                    "name": "context_analysis",
                    "execution_time": 1.5,
                    "tokens_used": 500,
                    "confidence_score": 0.8,
                    "reflection_applied": True
                },
                {
                    "name": "research_planning",
                    "execution_time": 2.2,
                    "tokens_used": 750,
                    "confidence_score": 0.85,
                    "reflection_applied": True
                },
                {
                    "name": "information_gathering",
                    "execution_time": 4.1,
                    "tokens_used": 1200,
                    "confidence_score": 0.9,
                    "reflection_applied": False
                },
                {
                    "name": "synthesis",
                    "execution_time": 3.2,
                    "tokens_used": 900,
                    "confidence_score": 0.88,
                    "reflection_applied": True
                },
                {
                    "name": "validation",
                    "execution_time": 1.8,
                    "tokens_used": 600,
                    "confidence_score": 0.92,
                    "reflection_applied": True
                }
            ],
            "total_execution_time": 12.8,
            "total_tokens_used": 3950,
            "overall_confidence": 0.87
        }
    
    def test_workflow_optimizer_initialization(self, workflow_optimizer):
        """Test WorkflowOptimizer initialization."""
        assert hasattr(workflow_optimizer, 'optimization_history')
        assert hasattr(workflow_optimizer, 'performance_metrics')
        assert hasattr(workflow_optimizer, 'optimization_strategies')
    
    def test_workflow_analysis(self, workflow_optimizer, sample_workflow_data):
        """Test workflow performance analysis."""
        analysis = workflow_optimizer.analyze_workflow_performance(sample_workflow_data)
        
        assert isinstance(analysis, dict)
        assert "execution_time_analysis" in analysis
        assert "token_usage_analysis" in analysis
        assert "confidence_analysis" in analysis
        assert "reflection_impact_analysis" in analysis
        assert "bottleneck_identification" in analysis
        
        # Verify execution time analysis
        time_analysis = analysis["execution_time_analysis"]
        assert "total_time" in time_analysis
        assert "average_stage_time" in time_analysis
        assert "slowest_stage" in time_analysis
        assert "fastest_stage" in time_analysis
        
        # Verify token usage analysis
        token_analysis = analysis["token_usage_analysis"]
        assert "total_tokens" in token_analysis
        assert "average_tokens_per_stage" in token_analysis
        assert "most_token_intensive_stage" in token_analysis
    
    def test_optimization_recommendations(self, workflow_optimizer, sample_workflow_data):
        """Test optimization recommendations generation."""
        recommendations = workflow_optimizer.generate_optimization_recommendations(sample_workflow_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for recommendation in recommendations:
            assert "type" in recommendation
            assert "description" in recommendation
            assert "impact" in recommendation
            assert "implementation" in recommendation
    
    def test_bottleneck_identification(self, workflow_optimizer, sample_workflow_data):
        """Test bottleneck identification."""
        bottlenecks = workflow_optimizer.identify_bottlenecks(sample_workflow_data)
        
        assert isinstance(bottlenecks, list)
        
        for bottleneck in bottlenecks:
            assert "stage" in bottleneck
            assert "type" in bottleneck  # time, tokens, confidence
            assert "severity" in bottleneck
            assert "recommendation" in bottleneck
    
    def test_reflection_impact_analysis(self, workflow_optimizer, sample_workflow_data):
        """Test reflection impact analysis."""
        impact_analysis = workflow_optimizer.analyze_reflection_impact(sample_workflow_data)
        
        assert isinstance(impact_analysis, dict)
        assert "stages_with_reflection" in impact_analysis
        assert "stages_without_reflection" in impact_analysis
        assert "reflection_efficiency" in impact_analysis
        assert "confidence_improvement" in impact_analysis
        
        # Verify reflection efficiency metrics
        efficiency = impact_analysis["reflection_efficiency"]
        assert "average_time_with_reflection" in efficiency
        assert "average_time_without_reflection" in efficiency
        assert "time_overhead" in efficiency
    
    def test_optimization_strategy_application(self, workflow_optimizer):
        """Test optimization strategy application."""
        strategy = {
            "name": "reduce_token_usage",
            "target_stages": ["information_gathering"],
            "parameters": {
                "token_reduction_target": 0.2,
                "maintain_quality_threshold": 0.85
            }
        }
        
        result = workflow_optimizer.apply_optimization_strategy(strategy)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "applied_changes" in result
        assert "expected_impact" in result
    
    def test_performance_comparison(self, workflow_optimizer):
        """Test performance comparison between workflow runs."""
        baseline_data = {
            "total_execution_time": 15.0,
            "total_tokens_used": 4500,
            "overall_confidence": 0.82
        }
        
        optimized_data = {
            "total_execution_time": 12.8,
            "total_tokens_used": 3950,
            "overall_confidence": 0.87
        }
        
        comparison = workflow_optimizer.compare_performance(baseline_data, optimized_data)
        
        assert isinstance(comparison, dict)
        assert "execution_time_improvement" in comparison
        assert "token_usage_improvement" in comparison
        assert "confidence_improvement" in comparison
        assert "overall_improvement_score" in comparison
        
        # Verify improvements are calculated correctly
        assert comparison["execution_time_improvement"] > 0  # Faster is better
        assert comparison["token_usage_improvement"] > 0  # Less tokens is better
        assert comparison["confidence_improvement"] > 0  # Higher confidence is better


class TestExecutionContextManager:
    """Test ExecutionContextManager functionality."""
    
    @pytest.fixture
    def context_manager(self):
        """Create ExecutionContextManager instance."""
        return ExecutionContextManager()
    
    def test_context_manager_initialization(self, context_manager):
        """Test ExecutionContextManager initialization."""
        assert hasattr(context_manager, 'active_contexts')
        assert hasattr(context_manager, 'context_history')
        assert hasattr(context_manager, 'isolation_settings')
    
    def test_context_creation(self, context_manager):
        """Test execution context creation."""
        context_id = "test_context_001"
        context_config = {
            "isolation_level": "moderate",
            "max_context_size": 4000,
            "enable_caching": True
        }
        
        context = context_manager.create_context(context_id, context_config)
        
        assert context is not None
        assert context.context_id == context_id
        assert context.config == context_config
        assert context_id in context_manager.active_contexts
    
    def test_context_isolation(self, context_manager):
        """Test context isolation functionality."""
        context_id = "isolated_context_001"
        context_config = {"isolation_level": "high"}
        
        context = context_manager.create_context(context_id, context_config)
        
        # Test context isolation
        isolated_data = {
            "research_query": "Test query",
            "findings": ["Finding 1", "Finding 2"],
            "metadata": {"confidence": 0.8}
        }
        
        context_manager.isolate_context_data(context_id, isolated_data)
        
        # Verify data is isolated
        retrieved_data = context_manager.get_isolated_data(context_id)
        assert retrieved_data == isolated_data
        
        # Verify isolation from other contexts
        other_context_id = "other_context_001"
        other_context = context_manager.create_context(other_context_id, context_config)
        other_data = context_manager.get_isolated_data(other_context_id)
        assert other_data != isolated_data
    
    def test_context_cleanup(self, context_manager):
        """Test context cleanup functionality."""
        context_id = "cleanup_test_context"
        context_config = {"isolation_level": "moderate"}
        
        # Create and use context
        context = context_manager.create_context(context_id, context_config)
        context_manager.isolate_context_data(context_id, {"test": "data"})
        
        # Verify context exists
        assert context_id in context_manager.active_contexts
        
        # Cleanup context
        cleanup_result = context_manager.cleanup_context(context_id)
        
        assert cleanup_result is True
        assert context_id not in context_manager.active_contexts
        assert context_id in context_manager.context_history
    
    def test_context_metrics_collection(self, context_manager):
        """Test context metrics collection."""
        context_id = "metrics_test_context"
        context_config = {"isolation_level": "moderate"}
        
        context = context_manager.create_context(context_id, context_config)
        
        # Record some metrics
        metrics_data = {
            "execution_time": 2.5,
            "memory_usage": 1024,
            "token_count": 500,
            "isolation_effectiveness": 0.9
        }
        
        context_manager.record_context_metrics(context_id, metrics_data)
        
        # Retrieve metrics
        context_metrics = context_manager.get_context_metrics(context_id)
        
        assert context_metrics is not None
        assert context_metrics["execution_time"] == 2.5
        assert context_metrics["memory_usage"] == 1024
        assert context_metrics["token_count"] == 500
    
    def test_concurrent_context_management(self, context_manager):
        """Test concurrent context management."""
        import threading
        import time
        
        results = []
        
        def create_context_thread(thread_id):
            try:
                context_id = f"concurrent_context_{thread_id}"
                context_config = {"isolation_level": "moderate"}
                
                context = context_manager.create_context(context_id, context_config)
                time.sleep(0.1)  # Simulate some work
                
                # Test data isolation
                test_data = {"thread_id": thread_id, "data": f"data_{thread_id}"}
                context_manager.isolate_context_data(context_id, test_data)
                
                retrieved_data = context_manager.get_isolated_data(context_id)
                
                results.append({
                    "thread_id": thread_id,
                    "success": retrieved_data["thread_id"] == thread_id,
                    "context_created": context is not None
                })
            except Exception as e:
                results.append({
                    "thread_id": thread_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_context_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all threads succeeded
        assert len(results) == 5
        for result in results:
            assert result["success"] is True
            assert result["context_created"] is True


class TestWorkflowIntegration:
    """Test workflow system integration."""
    
    @pytest.fixture
    def mock_state(self):
        """Create mock state for integration testing."""
        state = Mock(spec=State)
        state.research_topic = "AI in Healthcare Diagnostics"
        state.steps = []
        state.observations = []
        state.current_step = 0
        state.total_steps = 3
        state.config = Mock()
        state.config.enable_isolation = True
        state.config.enable_enhanced_reflection = True
        return state
    
    @pytest.mark.asyncio
    async def test_researcher_node_workflow_integration(self, mock_state):
        """Test researcher node integration with workflow."""
        with patch('src.graph.nodes.EnhancedReflectionAgent') as mock_agent_class:
            with patch('src.graph.nodes.ResearcherContextIsolator') as mock_isolator_class:
                with patch('src.graph.nodes.ReflectionWorkflow') as mock_workflow_class:
                    # Mock reflection agent
                    mock_agent = Mock()
                    mock_agent.analyze_knowledge_gaps = AsyncMock(return_value=Mock(
                        is_sufficient=True,
                        confidence_score=0.8,
                        knowledge_gaps=["Minor gap"],
                        follow_up_queries=["Follow-up query"]
                    ))
                    mock_agent_class.return_value = mock_agent
                    
                    # Mock context isolator
                    mock_isolator = Mock()
                    mock_isolator.execute_isolated_research = AsyncMock(return_value={
                        "research_findings": ["Finding 1", "Finding 2"],
                        "isolation_metrics": {"context_size": 1000, "execution_time": 2.5},
                        "workflow_metrics": {"stages_completed": 3, "total_time": 5.2}
                    })
                    mock_isolator_class.return_value = mock_isolator
                    
                    # Mock workflow
                    mock_workflow = Mock()
                    mock_workflow.execute_workflow = AsyncMock(return_value=Mock(
                        success=True,
                        execution_time=5.2,
                        stage_results=[{"stage": "analysis", "result": "success"}],
                        reflection_insights={"gaps": ["gap1"], "confidence": 0.8}
                    ))
                    mock_workflow_class.return_value = mock_workflow
                    
                    # Execute node function
                    result = await researcher_node_with_isolation(mock_state)
                    
                    # Verify integration
                    assert isinstance(result, dict)
                    assert "research_findings" in result
                    assert "isolation_metrics" in result
                    
                    # Verify workflow integration
                    if "workflow_metrics" in result:
                        assert "stages_completed" in result["workflow_metrics"]
                        assert "total_time" in result["workflow_metrics"]
    
    def test_workflow_configuration_integration(self):
        """Test workflow configuration integration."""
        with patch('src.config.config_manager.ConfigManager') as mock_config_class:
            mock_config_manager = Mock()
            mock_config_manager.get_workflow_config.return_value = {
                "enabled": True,
                "max_loops": 3,
                "stages": [
                    {"name": "analysis", "requires_reflection": True},
                    {"name": "synthesis", "requires_reflection": False}
                ]
            }
            mock_config_class.return_value = mock_config_manager
            
            # Test configuration loading
            config_manager = mock_config_class()
            workflow_config = config_manager.get_workflow_config()
            
            assert workflow_config["enabled"] is True
            assert len(workflow_config["stages"]) == 2
            
            # Test workflow creation with config
            workflow = ReflectionWorkflow(workflow_config)
            assert len(workflow.stages) == 2
    
    def test_end_to_end_workflow_simulation(self):
        """Test end-to-end workflow simulation."""
        # This test simulates a complete workflow execution
        # without actually calling external services
        
        workflow_config = {
            "enabled": True,
            "max_loops": 2,
            "confidence_threshold": 0.7,
            "stages": [
                {
                    "name": "analysis",
                    "description": "Analyze input",
                    "requires_reflection": True,
                    "isolation_level": "moderate",
                    "timeout": 30
                },
                {
                    "name": "synthesis",
                    "description": "Synthesize results",
                    "requires_reflection": False,
                    "isolation_level": "low",
                    "timeout": 20
                }
            ]
        }
        
        workflow = ReflectionWorkflow(workflow_config)
        
        # Verify workflow is properly configured
        assert len(workflow.stages) == 2
        assert workflow.config["enabled"] is True
        
        # Test workflow status tracking
        initial_status = workflow.get_workflow_status()
        assert initial_status["current_stage"] == 0
        assert initial_status["total_stages"] == 2
        
        # Test workflow reset
        workflow.reset_workflow()
        reset_status = workflow.get_workflow_status()
        assert reset_status["current_stage"] == 0
        assert reset_status["execution_time"] == 0


if __name__ == "__main__":
    # Run the comprehensive workflow test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src.workflow",
        "--cov=src.graph.nodes",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])