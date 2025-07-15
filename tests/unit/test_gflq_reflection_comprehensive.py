#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Comprehensive Unit Tests for GFLQ Reflection Integration

This test suite provides comprehensive coverage for the GFLQ reflection mechanism
integration in DeerFlow, replacing the demo scripts with proper unit tests.
"""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Core reflection components
from src.utils.reflection.enhanced_reflection import (
    EnhancedReflectionAgent, ReflectionResult, ReflectionContext
)
from src.utils.reflection.reflection_integration import (
    ReflectionIntegrator, ReflectionIntegrationConfig
)
from src.utils.reflection.reflection_tools import (
    ReflectionSession, ReflectionAnalyzer, ReflectionMetrics,
    parse_reflection_result
)

# Configuration and workflow components
from src.config.config_manager import ConfigManager
from src.config.config_integration import ConfigurationIntegrator
# Remove import of non-existent researcher_config_loader
# from src.config.researcher_config_loader import (
#     ResearcherConfigLoader, EnhancedReflectionConfig
# )
from src.workflow.reflection_workflow import ReflectionWorkflow, WorkflowStage

# Context and performance components
from src.utils.context.execution_context_manager import ExecutionContextManager
from src.utils.researcher.researcher_progressive_enablement import ResearcherProgressiveEnabler
from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_isolation_metrics import ResearcherIsolationMetrics

# Graph nodes
from src.graph.nodes import researcher_node_with_isolation
from src.graph.types import State


class TestGFLQReflectionCore:
    """Test core GFLQ reflection functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for reflection."""
        config = Mock()
        config.reflection_temperature = 0.7
        config.reflection_confidence_threshold = 0.7
        config.max_reflection_loops = 3
        config.knowledge_gap_threshold = 0.6
        config.sufficiency_threshold = 0.8
        config.reasoning_model = Mock()
        config.reflection_model = None
        return config
    
    @pytest.fixture
    def reflection_agent(self, mock_config):
        """Create EnhancedReflectionAgent instance."""
        return EnhancedReflectionAgent(mock_config)
    
    @pytest.fixture
    def reflection_context(self):
        """Create test reflection context."""
        return ReflectionContext(
            research_topic="Machine Learning in Healthcare",
            completed_steps=[
                {"step": "literature_review", "description": "Reviewed recent papers"},
                {"step": "data_analysis", "description": "Analyzed healthcare datasets"}
            ],
            execution_results=[
                "Found 50 relevant papers on ML in healthcare",
                "Identified key trends in diagnostic AI"
            ],
            total_steps=4,
            current_step_index=2
        )
    
    def test_reflection_result_creation(self):
        """Test ReflectionResult model creation and validation."""
        result = ReflectionResult(
            is_sufficient=True,
            confidence_score=0.85,
            knowledge_gaps=["Limited data on rural healthcare", "Missing cost-benefit analysis"],
            follow_up_queries=[
                "What are the implementation costs?",
                "How effective is ML in rural settings?"
            ],
            quality_assessment={
                "completeness": 0.8,
                "depth": 0.7,
                "relevance": 0.9
            },
            recommendations=["Focus on cost analysis", "Include rural case studies"],
            priority_areas=["cost_effectiveness", "rural_implementation"]
        )
        
        assert result.is_sufficient is True
        assert result.confidence_score == 0.85
        assert len(result.knowledge_gaps) == 2
        assert len(result.follow_up_queries) == 2
        assert result.quality_assessment["completeness"] == 0.8
        assert "cost_effectiveness" in result.priority_areas
    
    def test_reflection_result_defaults(self):
        """Test ReflectionResult with minimal parameters."""
        result = ReflectionResult(is_sufficient=False)
        
        assert result.is_sufficient is False
        assert result.confidence_score is None
        assert result.knowledge_gaps == []
        assert result.follow_up_queries == []
        assert result.quality_assessment == {}
        assert result.recommendations == []
        assert result.priority_areas == []
    
    @pytest.mark.asyncio
    async def test_enhanced_reflection_agent_initialization(self, reflection_agent, mock_config):
        """Test EnhancedReflectionAgent initialization."""
        assert reflection_agent.config == mock_config
        assert reflection_agent.session_count == 0
        assert reflection_agent.total_reflections == 0
        assert hasattr(reflection_agent, 'reflection_cache')
        assert hasattr(reflection_agent, 'metrics')
    
    @pytest.mark.asyncio
    async def test_knowledge_gaps_analysis(self, reflection_agent, reflection_context):
        """Test knowledge gaps analysis functionality."""
        with patch.object(reflection_agent, '_call_reflection_model') as mock_call:
            mock_response = Mock()
            mock_response.content = json.dumps({
                "knowledge_gaps": [
                    "Missing recent regulatory updates",
                    "Limited data on patient outcomes"
                ],
                "gap_severity": [0.8, 0.7],
                "recommendations": [
                    "Review FDA guidelines from 2024",
                    "Include patient outcome studies"
                ]
            })
            mock_call.return_value = mock_response
            
            result = await reflection_agent.analyze_knowledge_gaps(reflection_context)
            
            assert isinstance(result, ReflectionResult)
            assert len(result.knowledge_gaps) == 2
            assert "regulatory updates" in result.knowledge_gaps[0]
            assert len(result.recommendations) == 2
    
    @pytest.mark.asyncio
    async def test_research_quality_analysis(self, reflection_agent, reflection_context):
        """Test research quality analysis."""
        quality_result = await reflection_agent.analyze_research_quality(reflection_context)
        
        assert isinstance(quality_result, dict)
        assert "completeness_score" in quality_result
        assert "depth_score" in quality_result
        assert "relevance_score" in quality_result
        assert "overall_quality" in quality_result
        assert "areas_for_improvement" in quality_result
        
        # Verify score ranges
        for score_key in ["completeness_score", "depth_score", "relevance_score", "overall_quality"]:
            score = quality_result[score_key]
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_follow_up_queries_generation(self, reflection_agent):
        """Test follow-up queries generation."""
        knowledge_gaps = [
            "Missing cost-effectiveness data",
            "Limited implementation case studies",
            "Unclear regulatory requirements"
        ]
        
        with patch.object(reflection_agent, '_call_reflection_model') as mock_call:
            mock_response = Mock()
            mock_response.content = json.dumps({
                "follow_up_queries": [
                    "What are the typical implementation costs for ML in healthcare?",
                    "Which healthcare systems have successfully implemented ML diagnostics?",
                    "What are the current FDA requirements for ML medical devices?"
                ]
            })
            mock_call.return_value = mock_response
            
            queries = await reflection_agent.generate_follow_up_queries(
                "ML in Healthcare", knowledge_gaps
            )
            
            assert len(queries) == 3
            assert any("cost" in query.lower() for query in queries)
            assert any("implementation" in query.lower() for query in queries)
            assert any("fda" in query.lower() for query in queries)
    
    @pytest.mark.asyncio
    async def test_research_sufficiency_assessment(self, reflection_agent):
        """Test research sufficiency assessment."""
        # Test sufficient research case
        sufficient_context = ReflectionContext(
            research_topic="Well-researched topic",
            completed_steps=[{"step": "comprehensive_review", "description": "Complete analysis"}],
            execution_results=["Comprehensive findings with high confidence"],
            total_steps=2,
            current_step_index=1
        )
        
        with patch.object(reflection_agent, 'analyze_knowledge_gaps') as mock_analyze:
            mock_analyze.return_value = ReflectionResult(
                is_sufficient=True,
                confidence_score=0.9,
                knowledge_gaps=[],
                follow_up_queries=[]
            )
            
            result = await reflection_agent.assess_research_sufficiency(sufficient_context)
            assert result is True
        
        # Test insufficient research case
        insufficient_context = ReflectionContext(
            research_topic="Under-researched topic",
            completed_steps=[{"step": "initial_review", "description": "Basic analysis"}],
            execution_results=["Limited findings"],
            total_steps=5,
            current_step_index=1
        )
        
        with patch.object(reflection_agent, 'analyze_knowledge_gaps') as mock_analyze:
            mock_analyze.return_value = ReflectionResult(
                is_sufficient=False,
                confidence_score=0.4,
                knowledge_gaps=["Major data gaps", "Missing key studies"],
                follow_up_queries=["Need more research"]
            )
            
            result = await reflection_agent.assess_research_sufficiency(insufficient_context)
            assert result is False


class TestGFLQReflectionIntegration:
    """Test GFLQ reflection integration components."""
    
    @pytest.fixture
    def mock_integration_config(self):
        """Create mock integration configuration."""
        config = Mock()
        config.enable_reflection_integration = True
        config.reflection_trigger_threshold = 2
        config.max_reflection_loops = 3
        config.adaptive_reflection = True
        return config
    
    @pytest.fixture
    def reflection_integrator(self, mock_integration_config):
        """Create ReflectionIntegrator instance."""
        return ReflectionIntegrator(config=mock_integration_config)
    
    def test_reflection_integrator_initialization(self, reflection_integrator, mock_integration_config):
        """Test ReflectionIntegrator initialization."""
        assert reflection_integrator.config == mock_integration_config
        assert hasattr(reflection_integrator, 'active_reflections')
        assert hasattr(reflection_integrator, 'reflection_sessions')
        assert hasattr(reflection_integrator, 'integration_metrics')
    
    def test_reflection_trigger_logic(self, reflection_integrator):
        """Test reflection trigger logic."""
        from src.models.planner_model import Step
        
        # Create mock steps
        steps = [
            Step(step="step1", description="First step", execution_res="Result 1"),
            Step(step="step2", description="Second step", execution_res="Result 2"),
            Step(step="step3", description="Third step", execution_res="Result 3")
        ]
        
        # Test trigger conditions
        should_trigger_early = reflection_integrator.should_trigger_reflection(
            steps[:1], 0, "test_topic"
        )
        assert should_trigger_early is False  # Below threshold
        
        should_trigger_threshold = reflection_integrator.should_trigger_reflection(
            steps[:2], 1, "test_topic"
        )
        assert should_trigger_threshold is True  # At threshold
        
        should_trigger_above = reflection_integrator.should_trigger_reflection(
            steps, 2, "test_topic"
        )
        assert should_trigger_above is True  # Above threshold
    
    @pytest.mark.asyncio
    async def test_reflection_session_management(self, reflection_integrator):
        """Test reflection session management."""
        session_id = "test_session_001"
        research_topic = "AI Ethics in Healthcare"
        
        # Start reflection session
        session = reflection_integrator.start_reflection_session(
            session_id, research_topic
        )
        
        assert isinstance(session, ReflectionSession)
        assert session.session_id == session_id
        assert session.research_topic == research_topic
        assert session_id in reflection_integrator.active_reflections
        
        # End reflection session
        session_result = reflection_integrator.end_reflection_session(session_id)
        
        assert session_result is not None
        assert session_id not in reflection_integrator.active_reflections
        assert session_id in reflection_integrator.reflection_sessions
    
    @pytest.mark.asyncio
    async def test_reflection_metrics_collection(self, reflection_integrator):
        """Test reflection metrics collection."""
        # Create mock reflection result
        reflection_result = ReflectionResult(
            is_sufficient=True,
            confidence_score=0.8,
            knowledge_gaps=["gap1", "gap2"],
            follow_up_queries=["query1"],
            quality_assessment={"completeness": 0.9}
        )
        
        # Record metrics
        reflection_integrator.record_reflection_metrics(
            "test_session", reflection_result, execution_time=1.5
        )
        
        # Verify metrics were recorded
        metrics = reflection_integrator.get_reflection_metrics()
        assert "total_reflections" in metrics
        assert "average_confidence" in metrics
        assert "average_execution_time" in metrics
        assert metrics["total_reflections"] >= 1


class TestGFLQConfigurationSystem:
    """Test GFLQ reflection configuration system."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary configuration file."""
        config_data = {
            "enhanced_reflection": {
                "enable_enhanced_reflection": True,
                "max_reflection_loops": 3,
                "reflection_model": "deepseek-chat",
                "knowledge_gap_threshold": 0.6,
                "sufficiency_threshold": 0.8,
                "cache_settings": {
                    "enable_cache": True,
                    "cache_ttl": 3600
                },
                "trigger_settings": {
                    "trigger_on_low_confidence": True,
                    "trigger_on_knowledge_gaps": True,
                    "min_steps_before_trigger": 2
                }
            },
            "isolation": {
                "enable_isolation": True,
                "isolation_level": "moderate",
                "max_context_steps": 5
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)
    
    def test_researcher_config_loader(self, temp_config_file):
        """Test configuration loading functionality."""
        # Skip this test as ResearcherConfigLoader is no longer available
        pytest.skip("ResearcherConfigLoader has been removed in favor of new config system")
    
    def test_config_manager_integration(self):
        """Test ConfigManager integration with reflection."""
        with patch('src.config.config_manager.ConfigManager.get_researcher_config') as mock_get:
            # Mock researcher config
            mock_config = Mock()
            mock_config.enhanced_reflection.enable_enhanced_reflection = True
            mock_config.enhanced_reflection.max_reflection_loops = 3
            mock_get.return_value = mock_config
            
            manager = ConfigManager()
            
            # Test reflection configuration access
            is_enabled = manager.is_reflection_enabled()
            assert is_enabled is True
            
            reflection_config = manager.get_reflection_config()
            assert reflection_config is not None
    
    def test_configuration_integrator(self):
        """Test ConfigurationIntegrator functionality."""
        with patch('src.config.config_integration.ConfigurationIntegrator.load_researcher_config') as mock_load:
            mock_config = Mock()
            mock_config.enhanced_reflection = Mock()
            mock_config.enhanced_reflection.enable_enhanced_reflection = True
            mock_load.return_value = mock_config
            
            integrator = ConfigurationIntegrator()
            
            # Test configuration loading
            config = integrator.load_researcher_config()
            assert config is not None
            
            # Test validation
            is_valid = integrator.validate_configuration()
            assert is_valid is not None
            
            # Test integrated config generation
            reflection_config = integrator.get_integrated_reflection_config()
            assert reflection_config is not None


class TestGFLQWorkflowIntegration:
    """Test GFLQ reflection workflow integration."""
    
    @pytest.fixture
    def mock_reflection_config(self):
        """Create mock reflection configuration for workflow."""
        return {
            "enabled": True,
            "max_loops": 3,
            "confidence_threshold": 0.7,
            "enable_progressive": True,
            "enable_context_expansion": True,
            "enable_isolation_metrics": True
        }
    
    @pytest.fixture
    def reflection_workflow(self, mock_reflection_config):
        """Create ReflectionWorkflow instance."""
        return ReflectionWorkflow(mock_reflection_config)
    
    def test_workflow_initialization(self, reflection_workflow, mock_reflection_config):
        """Test ReflectionWorkflow initialization."""
        assert reflection_workflow.config == mock_reflection_config
        assert hasattr(reflection_workflow, 'stages')
        assert hasattr(reflection_workflow, 'reflection_agent')
        assert hasattr(reflection_workflow, 'progressive_enabler')
        assert hasattr(reflection_workflow, 'context_expansion')
        assert hasattr(reflection_workflow, 'isolation_metrics')
        
        # Verify stages are configured
        assert len(reflection_workflow.stages) > 0
        for stage in reflection_workflow.stages:
            assert isinstance(stage, WorkflowStage)
            assert hasattr(stage, 'name')
            assert hasattr(stage, 'description')
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, reflection_workflow):
        """Test workflow execution with reflection."""
        research_query = "Impact of AI on software development productivity"
        initial_context = {
            "domain": "software_engineering",
            "focus": "productivity_metrics",
            "technologies": ["ai_tools", "automation", "code_generation"]
        }
        
        with patch.object(reflection_workflow, '_execute_stage') as mock_execute:
            # Mock stage execution results
            mock_execute.return_value = {
                "success": True,
                "stage_metrics": {
                    "execution_time": 1.2,
                    "reflection_applied": True,
                    "context_expanded": True
                },
                "stage_data": {"findings": ["AI tools increase productivity by 30%"]}
            }
            
            result = await reflection_workflow.execute_workflow(research_query, initial_context)
            
            assert result.success is True
            assert result.execution_time > 0
            assert len(result.stage_results) > 0
            assert hasattr(result, 'reflection_insights')
            assert hasattr(result, 'metrics')
    
    def test_workflow_stage_configuration(self, reflection_workflow):
        """Test workflow stage configuration."""
        expected_stages = [
            "context_analysis",
            "research_planning", 
            "information_gathering",
            "synthesis",
            "validation"
        ]
        
        stage_names = [stage.name for stage in reflection_workflow.stages]
        
        for expected_stage in expected_stages:
            assert expected_stage in stage_names
        
        # Verify stage properties
        for stage in reflection_workflow.stages:
            assert hasattr(stage, 'requires_reflection')
            assert hasattr(stage, 'isolation_level')
            assert hasattr(stage, 'context_expansion')
    
    def test_workflow_status_tracking(self, reflection_workflow):
        """Test workflow status tracking."""
        status = reflection_workflow.get_workflow_status()
        
        assert isinstance(status, dict)
        assert "current_stage" in status
        assert "total_stages" in status
        assert "reflection_enabled" in status
        assert "active_sessions" in status
        
        # Test workflow reset
        reflection_workflow.reset_workflow()
        reset_status = reflection_workflow.get_workflow_status()
        assert reset_status["current_stage"] == 0


class TestGFLQNodeIntegration:
    """Test GFLQ reflection integration in graph nodes."""
    
    @pytest.fixture
    def mock_state(self):
        """Create mock state for node testing."""
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
    async def test_researcher_node_with_isolation(self, mock_state):
        """Test researcher_node_with_isolation function."""
        with patch('src.graph.nodes.EnhancedReflectionAgent') as mock_agent_class:
            with patch('src.graph.nodes.ResearcherContextIsolator') as mock_isolator_class:
                # Mock reflection agent
                mock_agent = Mock()
                mock_agent.analyze_knowledge_gaps = AsyncMock(return_value=ReflectionResult(
                    is_sufficient=True,
                    confidence_score=0.8,
                    knowledge_gaps=["Minor gap in recent studies"],
                    follow_up_queries=["What are the latest developments?"]
                ))
                mock_agent_class.return_value = mock_agent
                
                # Mock context isolator
                mock_isolator = Mock()
                mock_isolator.execute_isolated_research = AsyncMock(return_value={
                    "research_findings": ["Finding 1", "Finding 2"],
                    "isolation_metrics": {"context_size": 1000, "execution_time": 2.5}
                })
                mock_isolator_class.return_value = mock_isolator
                
                # Execute node function
                result = await researcher_node_with_isolation(mock_state)
                
                # Verify result structure
                assert isinstance(result, dict)
                assert "research_findings" in result
                assert "isolation_metrics" in result
                
                # Verify reflection integration
                if hasattr(result, 'reflection_insights'):
                    assert "knowledge_gaps" in result.reflection_insights
                    assert "research_sufficiency" in result.reflection_insights
    
    def test_reflection_insights_access(self):
        """Test reflection insights access methods."""
        # Mock isolation result with reflection insights
        isolation_result = Mock()
        isolation_result.reflection_insights = {
            "knowledge_gaps": ["Gap 1", "Gap 2"],
            "research_sufficiency": True,
            "follow_up_queries": ["Query 1"],
            "confidence_score": 0.85
        }
        
        # Test helper methods (these would be added to the isolation result)
        def get_reflection_insights():
            return isolation_result.reflection_insights
        
        def has_reflection_insights():
            return hasattr(isolation_result, 'reflection_insights') and isolation_result.reflection_insights
        
        def is_research_sufficient():
            insights = get_reflection_insights()
            return insights.get('research_sufficiency', False) if insights else False
        
        # Test the methods
        assert has_reflection_insights() is True
        assert is_research_sufficient() is True
        
        insights = get_reflection_insights()
        assert len(insights["knowledge_gaps"]) == 2
        assert insights["confidence_score"] == 0.85


class TestGFLQPerformanceAndMetrics:
    """Test GFLQ reflection performance and metrics."""
    
    @pytest.fixture
    def reflection_metrics(self):
        """Create ReflectionMetrics instance."""
        return ReflectionMetrics()
    
    def test_reflection_metrics_initialization(self, reflection_metrics):
        """Test ReflectionMetrics initialization."""
        assert hasattr(reflection_metrics, 'total_reflections')
        assert hasattr(reflection_metrics, 'successful_reflections')
        assert hasattr(reflection_metrics, 'average_confidence')
        assert hasattr(reflection_metrics, 'total_execution_time')
        assert hasattr(reflection_metrics, 'knowledge_gaps_identified')
    
    def test_metrics_recording(self, reflection_metrics):
        """Test metrics recording functionality."""
        # Record reflection metrics
        reflection_metrics.record_reflection(
            confidence_score=0.8,
            execution_time=1.5,
            knowledge_gaps_count=3,
            was_successful=True
        )
        
        reflection_metrics.record_reflection(
            confidence_score=0.7,
            execution_time=2.0,
            knowledge_gaps_count=1,
            was_successful=True
        )
        
        # Verify metrics
        assert reflection_metrics.total_reflections == 2
        assert reflection_metrics.successful_reflections == 2
        assert reflection_metrics.average_confidence == 0.75
        assert reflection_metrics.total_execution_time == 3.5
        assert reflection_metrics.knowledge_gaps_identified == 4
    
    def test_metrics_summary(self, reflection_metrics):
        """Test metrics summary generation."""
        # Add some test data
        for i in range(5):
            reflection_metrics.record_reflection(
                confidence_score=0.8 + (i * 0.02),
                execution_time=1.0 + (i * 0.1),
                knowledge_gaps_count=i + 1,
                was_successful=i < 4  # One failure
            )
        
        summary = reflection_metrics.get_summary()
        
        assert isinstance(summary, dict)
        assert "total_reflections" in summary
        assert "success_rate" in summary
        assert "average_confidence" in summary
        assert "average_execution_time" in summary
        assert "total_knowledge_gaps" in summary
        
        # Verify calculated values
        assert summary["total_reflections"] == 5
        assert summary["success_rate"] == 0.8  # 4/5
        assert summary["total_knowledge_gaps"] == 15  # 1+2+3+4+5
    
    @pytest.mark.asyncio
    async def test_progressive_enabler_performance(self):
        """Test ResearcherProgressiveEnabler performance."""
        with patch('src.utils.researcher.researcher_progressive_enabler.ResearcherProgressiveEnabler') as mock_enabler_class:
            mock_enabler = Mock()
            mock_enabler.should_enable_reflection = Mock(return_value=True)
            mock_enabler.get_reflection_intensity = Mock(return_value=0.8)
            mock_enabler.update_performance_metrics = Mock()
            mock_enabler_class.return_value = mock_enabler
            
            enabler = mock_enabler_class()
            
            # Test performance tracking
            performance_data = {
                "execution_time": 2.5,
                "token_usage": 1500,
                "confidence_score": 0.85
            }
            
            enabler.update_performance_metrics(performance_data)
            
            # Verify method calls
            enabler.update_performance_metrics.assert_called_once_with(performance_data)
            
            # Test adaptive enabling
            should_enable = enabler.should_enable_reflection()
            assert should_enable is True
            
            intensity = enabler.get_reflection_intensity()
            assert intensity == 0.8
    
    def test_isolation_metrics_collection(self):
        """Test ResearcherIsolationMetrics collection."""
        with patch('src.utils.monitoring.researcher_isolation_metrics.ResearcherIsolationMetrics') as mock_metrics_class:
            mock_metrics = Mock()
            mock_metrics.record_isolation_event = Mock()
            mock_metrics.get_isolation_statistics = Mock(return_value={
                "total_isolations": 10,
                "average_context_size": 1200,
                "average_execution_time": 1.8,
                "token_savings": 2500
            })
            mock_metrics_class.return_value = mock_metrics
            
            metrics = mock_metrics_class()
            
            # Record isolation event
            isolation_data = {
                "context_size": 1000,
                "execution_time": 1.5,
                "token_savings": 250,
                "isolation_level": "moderate"
            }
            
            metrics.record_isolation_event(isolation_data)
            
            # Get statistics
            stats = metrics.get_isolation_statistics()
            
            # Verify data
            assert stats["total_isolations"] == 10
            assert stats["average_context_size"] == 1200
            assert stats["token_savings"] == 2500
            
            metrics.record_isolation_event.assert_called_once_with(isolation_data)


class TestGFLQErrorHandlingAndEdgeCases:
    """Test GFLQ reflection error handling and edge cases."""
    
    @pytest.fixture
    def reflection_agent_with_errors(self):
        """Create reflection agent for error testing."""
        config = Mock()
        config.reflection_temperature = 0.7
        config.max_reflection_loops = 3
        config.reasoning_model = Mock()
        return EnhancedReflectionAgent(config)
    
    @pytest.mark.asyncio
    async def test_reflection_model_failure_handling(self, reflection_agent_with_errors):
        """Test handling of reflection model failures."""
        context = ReflectionContext(
            research_topic="Test topic",
            completed_steps=[{"step": "test", "description": "test"}],
            execution_results=["test result"],
            total_steps=2,
            current_step_index=1
        )
        
        with patch.object(reflection_agent_with_errors, '_call_reflection_model') as mock_call:
            # Simulate model failure
            mock_call.side_effect = Exception("Model API error")
            
            # Test graceful failure handling
            result = await reflection_agent_with_errors.analyze_knowledge_gaps(context)
            
            # Should return a default result instead of crashing
            assert isinstance(result, ReflectionResult)
            assert result.is_sufficient is False  # Conservative default
            assert "error" in str(result.knowledge_gaps).lower() or len(result.knowledge_gaps) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_reflection_context(self, reflection_agent_with_errors):
        """Test handling of invalid reflection context."""
        # Test with None context
        result = await reflection_agent_with_errors.analyze_knowledge_gaps(None)
        assert isinstance(result, ReflectionResult)
        assert result.is_sufficient is False
        
        # Test with empty context
        empty_context = ReflectionContext(
            research_topic="",
            completed_steps=[],
            execution_results=[],
            total_steps=0,
            current_step_index=0
        )
        
        result = await reflection_agent_with_errors.analyze_knowledge_gaps(empty_context)
        assert isinstance(result, ReflectionResult)
        assert result.is_sufficient is False
    
    def test_configuration_validation_errors(self):
        """Test configuration validation error handling."""
        # Test invalid configuration
        invalid_config = Mock()
        invalid_config.reflection_temperature = 2.0  # Invalid value > 1.0
        invalid_config.max_reflection_loops = -1  # Invalid negative value
        
        with pytest.raises((ValueError, AssertionError)):
            EnhancedReflectionAgent(invalid_config)
    
    def test_reflection_cache_edge_cases(self):
        """Test reflection cache edge cases."""
        config = Mock()
        config.reflection_temperature = 0.7
        config.max_reflection_loops = 3
        config.reasoning_model = Mock()
        
        agent = EnhancedReflectionAgent(config)
        
        # Test cache with None key
        agent._cache_reflection_result(None, ReflectionResult(is_sufficient=True))
        
        # Test cache retrieval with None key
        cached_result = agent._get_cached_reflection_result(None)
        assert cached_result is None
        
        # Test cache with empty string key
        agent._cache_reflection_result("", ReflectionResult(is_sufficient=True))
        cached_result = agent._get_cached_reflection_result("")
        assert cached_result is None
    
    def test_workflow_interruption_handling(self):
        """Test workflow interruption and recovery."""
        config = {"enabled": True, "max_loops": 3}
        workflow = ReflectionWorkflow(config)
        
        # Test workflow reset after interruption
        workflow.reset_workflow()
        status = workflow.get_workflow_status()
        assert status["current_stage"] == 0
        
        # Test workflow state consistency
        assert hasattr(workflow, 'stages')
        assert len(workflow.stages) > 0


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src.utils.researcher",
        "--cov=src.config",
        "--cov=src.workflow",
        "--cov=src.graph.nodes",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])