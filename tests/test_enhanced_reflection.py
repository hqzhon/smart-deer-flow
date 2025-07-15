# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Tests for enhanced reflection components (Phase 1).
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.utils.reflection.enhanced_reflection import EnhancedReflectionAgent, ReflectionResult
from src.utils.reflection.reflection_integration import ReflectionIntegrator, ReflectionIntegrationConfig
from src.utils.reflection.reflection_tools import (
    ReflectionSession, ReflectionAnalyzer, ReflectionMetrics, 
    parse_reflection_result, calculate_research_complexity
)
# from src.config.configuration import Configuration  # Removed - using new config system
from src.graph.types import State


class TestReflectionResult:
    """Test ReflectionResult model."""
    
    def test_reflection_result_creation(self):
        """Test basic ReflectionResult creation."""
        result = ReflectionResult(
            is_sufficient=True,
            confidence_score=0.8,
            knowledge_gaps=["gap1", "gap2"],
            follow_up_queries=["query1"],
            quality_assessment={"completeness": 0.9},
            recommendations=["rec1"],
            priority_areas=["area1"]
        )
        
        assert result.is_sufficient is True
        assert result.confidence_score == 0.8
        assert len(result.knowledge_gaps) == 2
        assert len(result.follow_up_queries) == 1
        assert result.quality_assessment["completeness"] == 0.9
    
    def test_reflection_result_defaults(self):
        """Test ReflectionResult with default values."""
        result = ReflectionResult(is_sufficient=False)
        
        assert result.is_sufficient is False
        assert result.confidence_score is None
        assert result.knowledge_gaps == []
        assert result.follow_up_queries == []
        assert result.quality_assessment == {}
        assert result.recommendations == []
        assert result.priority_areas == []


class TestEnhancedReflectionAgent:
    """Test EnhancedReflectionAgent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()  # Removed spec=Configuration
        config.reflection_temperature = 0.7
        config.reflection_confidence_threshold = 0.7
        config.max_reflection_loops = 3
        config.reasoning_model = Mock()
        config.reflection_model = None
        return config
    
    @pytest.fixture
    def reflection_agent(self, mock_config):
        """Create ReflectionAgent instance."""
        return EnhancedReflectionAgent(mock_config)
    
    def test_agent_initialization(self, reflection_agent, mock_config):
        """Test agent initialization."""
        assert reflection_agent.config == mock_config
        assert reflection_agent.session_count == 0
        assert reflection_agent.total_reflections == 0
    
    @pytest.mark.asyncio
    async def test_analyze_research_quality(self, reflection_agent):
        """Test research quality analysis."""
        from src.utils.reflection.enhanced_reflection import ReflectionContext
        
        # Create mock context
        context = ReflectionContext(
            research_topic="AI trends",
            completed_steps=[{"step": "test", "description": "test step"}],
            execution_results=["AI is growing"],
            total_steps=2,
            current_step_index=1
        )
        
        # Test the analyze_research_quality method
        result = await reflection_agent.analyze_research_quality(context)
        
        assert isinstance(result, dict)
        assert "completeness_score" in result
        assert "depth_score" in result
        assert "relevance_score" in result
        assert "overall_quality" in result
        assert "areas_for_improvement" in result
    
    @pytest.mark.asyncio
    async def test_generate_follow_up_queries(self, reflection_agent):
        """Test follow-up query generation."""
        knowledge_gaps = ["Missing recent data", "Unclear methodology"]
        
        with patch.object(reflection_agent, '_call_reflection_model') as mock_call:
            mock_response = Mock()
            mock_response.content = json.dumps({
                "follow_up_queries": [
                    "What is the latest data on this topic?",
                    "What methodology was used in recent studies?"
                ]
            })
            mock_call.return_value = mock_response
            
            queries = await reflection_agent.generate_follow_up_queries(
                "AI research", knowledge_gaps
            )
            
            assert len(queries) == 2
            assert "latest data" in queries[0]
            assert "methodology" in queries[1]
    
    @pytest.mark.asyncio
    async def test_assess_research_sufficiency(self, reflection_agent):
        """Test research sufficiency assessment."""
        from src.utils.reflection.enhanced_reflection import ReflectionContext
        
        # Test case 1: Sufficient research (enhanced reflection confirms)
        context_sufficient = ReflectionContext(
            research_topic="Test topic",
            completed_steps=[{"step": "test", "description": "test step"}],
            execution_results=["Test result"],
            total_steps=2,
            current_step_index=1  # At final step
        )
        
        # Mock the analyze_knowledge_gaps method
        with patch.object(reflection_agent, 'analyze_knowledge_gaps') as mock_analyze:
            # Test sufficient research - high confidence and is_sufficient=True
            mock_analyze.return_value = ReflectionResult(
                is_sufficient=True,
                confidence_score=0.9,  # Above threshold (0.7)
                knowledge_gaps=[],
                follow_up_queries=[]
            )
            result = await reflection_agent.assess_research_sufficiency(context_sufficient)
            assert isinstance(result, ReflectionResult)
            assert result.is_sufficient is True
            
            # Test case 2: Insufficient research (not at final step, low confidence)
            context_insufficient = ReflectionContext(
                research_topic="Test topic",
                completed_steps=[{"step": "test", "description": "test step"}],
                execution_results=[],  # No results yet
                total_steps=5,
                current_step_index=1  # Early in process
            )
            
            mock_analyze.return_value = ReflectionResult(
                is_sufficient=False,
                confidence_score=0.5,  # Below threshold (0.7)
                knowledge_gaps=["Need more data"],
                follow_up_queries=["What else?"]
            )
            result = await reflection_agent.assess_research_sufficiency(context_insufficient)
            assert isinstance(result, ReflectionResult)
            assert result.is_sufficient is False


class TestReflectionIntegrator:
    """Test ReflectionIntegrator functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()  # Removed spec=Configuration
        config.enable_reflection_integration = True
        config.reflection_trigger_threshold = 2
        config.max_reflection_loops = 3
        return config
    
    @pytest.fixture
    def integrator(self, mock_config):
        """Create ReflectionIntegrator instance."""
        return ReflectionIntegrator(config=mock_config)
    
    def test_integrator_initialization(self, integrator, mock_config):
        """Test integrator initialization."""
        assert integrator.config == mock_config
        assert hasattr(integrator, 'active_reflections')
        assert hasattr(integrator, 'reflection_sessions')
    
    def test_should_trigger_reflection(self, integrator):
        """Test reflection trigger logic."""
        from src.models.planner_model import Step
        
        state = Mock(spec=State)
        state.get = Mock(return_value=[])
        
        # Create a mock step
        mock_step = Mock(spec=Step)
        mock_step.need_search = True
        mock_step.description = "Test step"
        
        # Test trigger conditions - with enough observations
        state.get.side_effect = lambda key, default=None: {
            "observations": ["obs1", "obs2", "obs3"],  # More than threshold
            "current_plan": Mock(),
            "resources": []
        }.get(key, default)
        
        should_trigger, reason, factors = integrator.should_trigger_reflection(state, current_step=mock_step)
        assert should_trigger is True
        assert "Step count" in reason  # Match actual format
        assert "step_count" in factors
        
        # Test no trigger - not enough observations
        state.get.side_effect = lambda key, default=None: {
            "observations": ["obs1"],  # Less than threshold
            "current_plan": Mock(),
            "resources": []
        }.get(key, default)
        
        should_trigger, reason, factors = integrator.should_trigger_reflection(state, current_step=mock_step)
        assert should_trigger is False
        assert "integration_enabled" in factors
    
    @pytest.mark.asyncio
    async def test_execute_reflection_analysis(self, integrator):
        """Test reflection analysis execution."""
        # Create a mock plan with steps
        mock_plan = Mock()
        mock_plan.steps = []  # Empty list to avoid iteration error
        
        state = Mock(spec=State)
        state.get = Mock(side_effect=lambda key, default=None: {
            "research_topic": "Test topic",
            "observations": ["Finding 1", "Finding 2"],
            "current_plan": mock_plan,
            "execution_results": ["Result 1"],
            "resources": [],
            "locale": "en-US"
        }.get(key, default))
        
        # Ensure the config has max_reflection_iterations
        integrator.config.max_reflection_iterations = 3
        
        # Mock reflection agent
        mock_agent = Mock()
        mock_result = ReflectionResult(
            is_sufficient=False,
            confidence_score=0.6,
            knowledge_gaps=["Need more data"]
        )
        mock_agent.analyze_knowledge_gaps = AsyncMock(return_value=mock_result)
        
        with patch.object(integrator, 'reflection_agent', mock_agent):
            result, context = await integrator.execute_reflection_analysis(
                state=state,
                session_id="session_123"
            )
            
            assert isinstance(result, ReflectionResult)
            assert result.confidence_score == 0.6
            mock_agent.analyze_knowledge_gaps.assert_called_once()


class TestReflectionTools:
    """Test reflection utility tools."""
    
    def test_parse_reflection_result_valid(self):
        """Test parsing valid reflection result."""
        raw_output = '''
        Here is the analysis:
        {
            "is_sufficient": false,
            "confidence_score": 0.7,
            "knowledge_gaps": ["gap1", "gap2"],
            "follow_up_queries": ["query1"]
        }
        '''
        
        result = parse_reflection_result(raw_output)
        
        assert result is not None
        assert result.is_sufficient is False
        assert result.confidence_score == 0.7
        assert len(result.knowledge_gaps) == 2
        assert len(result.follow_up_queries) == 1
    
    def test_parse_reflection_result_invalid(self):
        """Test parsing invalid reflection result."""
        raw_output = "This is not valid JSON output"
        
        result = parse_reflection_result(raw_output)
        
        assert result is None
    
    def test_calculate_research_complexity(self):
        """Test research complexity calculation."""
        # Simple research
        complexity = calculate_research_complexity(
            research_topic="AI",
            step_count=2,
            context_size=1000,
            findings_count=3
        )
        assert 0.0 <= complexity <= 1.0
        
        # Complex research
        complex_complexity = calculate_research_complexity(
            research_topic="Advanced machine learning algorithms for natural language processing",
            step_count=15,
            context_size=50000,
            findings_count=30
        )
        assert complex_complexity > complexity
        assert complex_complexity <= 1.0
    
    def test_reflection_session(self):
        """Test ReflectionSession functionality."""
        session = ReflectionSession(
            session_id="test-123",
            research_topic="AI trends",
            step_count=3,
            timestamp=datetime.now(),
            context_size=5000,
            complexity_score=0.6
        )
        
        assert session.session_id == "test-123"
        assert session.research_topic == "AI trends"
        assert session.step_count == 3
        assert session.context_size == 5000
        assert session.complexity_score == 0.6
        
        # Test to_dict conversion
        session_dict = session.to_dict()
        assert isinstance(session_dict, dict)
        assert session_dict["session_id"] == "test-123"
        assert "timestamp" in session_dict


class TestReflectionAnalyzer:
    """Test ReflectionAnalyzer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Configuration)
        config.reflection_confidence_threshold = 0.7
        return config
    
    @pytest.fixture
    def analyzer(self, mock_config):
        """Create ReflectionAnalyzer instance."""
        return ReflectionAnalyzer(mock_config)
    
    def test_analyze_sufficiency_trend_improving(self, analyzer):
        """Test sufficiency trend analysis - improving trend."""
        sessions = [
            ReflectionSession(
                session_id="1", research_topic="test", step_count=1,
                timestamp=datetime.now(),
                reflection_result=ReflectionResult(is_sufficient=False, confidence_score=0.5)
            ),
            ReflectionSession(
                session_id="2", research_topic="test", step_count=2,
                timestamp=datetime.now(),
                reflection_result=ReflectionResult(is_sufficient=False, confidence_score=0.7)
            ),
            ReflectionSession(
                session_id="3", research_topic="test", step_count=3,
                timestamp=datetime.now(),
                reflection_result=ReflectionResult(is_sufficient=True, confidence_score=0.9)
            )
        ]
        
        trend = analyzer.analyze_sufficiency_trend(sessions)
        
        assert trend["trend"] == "improving"
        assert trend["confidence"] == 0.9
        assert trend["recommendation"] == "continue_current_approach"
    
    def test_identify_recurring_gaps(self, analyzer):
        """Test identification of recurring knowledge gaps."""
        sessions = [
            ReflectionSession(
                session_id="1", research_topic="test", step_count=1,
                timestamp=datetime.now(),
                reflection_result=ReflectionResult(
                    is_sufficient=False,
                    knowledge_gaps=["Need recent data", "Missing methodology"]
                )
            ),
            ReflectionSession(
                session_id="2", research_topic="test", step_count=2,
                timestamp=datetime.now(),
                reflection_result=ReflectionResult(
                    is_sufficient=False,
                    knowledge_gaps=["Need recent data", "Unclear sources"]
                )
            )
        ]
        
        recurring_gaps = analyzer.identify_recurring_gaps(sessions)
        
        assert "need recent data" in recurring_gaps
        assert "missing methodology" not in recurring_gaps


class TestReflectionMetrics:
    """Test ReflectionMetrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ReflectionMetrics()
        
        assert "confidence_scores" in metrics.metrics_data
        assert "gap_counts" in metrics.metrics_data
        assert "query_counts" in metrics.metrics_data
        assert len(metrics.metrics_data["confidence_scores"]) == 0
    
    def test_record_session_metrics(self):
        """Test recording session metrics."""
        metrics = ReflectionMetrics()
        
        session = ReflectionSession(
            session_id="test",
            research_topic="AI",
            step_count=3,
            timestamp=datetime.now(),
            complexity_score=0.6,
            reflection_result=ReflectionResult(
                is_sufficient=False,
                confidence_score=0.7,
                knowledge_gaps=["gap1", "gap2"],
                follow_up_queries=["query1"]
            )
        )
        
        metrics.record_session_metrics(session, 1500.0)
        
        assert len(metrics.metrics_data["confidence_scores"]) == 1
        assert metrics.metrics_data["confidence_scores"][0] == 0.7
        assert metrics.metrics_data["gap_counts"][0] == 2
        assert metrics.metrics_data["query_counts"][0] == 1
        assert metrics.metrics_data["session_durations"][0] == 1500.0
        assert metrics.metrics_data["complexity_scores"][0] == 0.6
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        metrics = ReflectionMetrics()
        
        # Add some test data
        metrics.metrics_data["confidence_scores"] = [0.5, 0.7, 0.9]
        metrics.metrics_data["gap_counts"] = [3, 2, 1]
        
        summary = metrics.get_performance_summary()
        
        assert summary["confidence_scores"]["count"] == 3
        assert abs(summary["confidence_scores"]["average"] - 0.7) < 0.001
        assert summary["confidence_scores"]["min"] == 0.5
        assert summary["confidence_scores"]["max"] == 0.9
        assert summary["confidence_scores"]["latest"] == 0.9
        
        assert summary["gap_counts"]["average"] == 2.0
    
    def test_export_metrics(self):
        """Test metrics export functionality."""
        metrics = ReflectionMetrics()
        metrics.metrics_data["confidence_scores"] = [0.8]
        
        export_data = metrics.export_metrics()
        
        assert isinstance(export_data, str)
        parsed_data = json.loads(export_data)
        assert "metrics" in parsed_data
        assert "summary" in parsed_data
        assert "export_timestamp" in parsed_data