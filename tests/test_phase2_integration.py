# -*- coding: utf-8 -*-
"""
Phase 2 Integration Tests - GFLQ Reflection Integration

Tests for the integration of enhanced reflection capabilities into existing
researcher components as part of Phase 2 implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.utils.researcher.researcher_progressive_enablement import (
    ResearcherProgressiveEnabler, ScenarioContext, TaskComplexity
)
from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_isolation_metrics import (
    ResearcherIsolationMetrics, IsolationSession
)
from src.utils.reflection.enhanced_reflection import EnhancedReflectionAgent, ReflectionResult


class TestPhase2ProgressiveEnablerIntegration:
    """Test reflection integration in ResearcherProgressiveEnabler"""
    
    def test_progressive_enabler_with_reflection_config(self):
        """Test progressive enabler initialization with reflection config"""
        # Create a mock config object with the required attribute
        config = Mock()
        config.enable_enhanced_reflection = True
        config.max_reflection_loops = 3
        config.reflection_model = 'gpt-4'
        
        with patch('src.utils.researcher.enhanced_reflection.EnhancedReflectionAgent') as mock_agent:
            with patch('src.utils.researcher.researcher_isolation_metrics.get_isolation_metrics') as mock_metrics:
                enabler = ResearcherProgressiveEnabler(config=config)
                
                assert enabler.reflection_agent is not None
                mock_agent.assert_called_once()
    
    def test_progressive_enabler_without_reflection_config(self):
        """Test progressive enabler initialization without reflection config"""
        config = Mock()
        config.enable_enhanced_reflection = False
        
        enabler = ResearcherProgressiveEnabler(config=config)
        assert enabler.reflection_agent is None
    
    def test_adaptive_enablement_with_reflection_insights(self):
        """Test adaptive enablement with reflection-driven decisions"""
        config = Mock()
        config.enable_enhanced_reflection = True
        
        # Mock reflection agent
        mock_reflection_agent = Mock()
        mock_reflection_result = ReflectionResult(
            is_sufficient=False,
            knowledge_gaps=['gap1', 'gap2'],
            follow_up_queries=['query1', 'query2'],
            research_quality_score=0.6
        )
        mock_reflection_agent.assess_research_sufficiency.return_value = mock_reflection_result
        
        with patch('src.utils.researcher.enhanced_reflection.EnhancedReflectionAgent', return_value=mock_reflection_agent):
            enabler = ResearcherProgressiveEnabler(config=config)
            
            scenario = ScenarioContext(
                task_description="Test task",
                step_count=2,
                context_size=1000,
                parallel_execution=False,
                has_search_results=True,
                has_complex_queries=False,
                estimated_tokens=1500
            )
            
            should_enable, reason, factors = enabler._adaptive_enablement(
                scenario, TaskComplexity.MEDIUM, config, None, {}
            )
            
            assert should_enable is True
            assert "reflection analysis indicates knowledge gaps" in reason.lower()
            assert 'reflection_insights' in factors
            assert factors['reflection_insights']['is_sufficient'] is False
            assert factors['reflection_insights']['knowledge_gaps_count'] == 2
    
    def test_adaptive_enablement_with_sufficient_reflection(self):
        """Test adaptive enablement when reflection indicates sufficient knowledge"""
        config = Mock()
        config.enable_enhanced_reflection = True
        
        # Mock reflection agent with sufficient knowledge
        mock_reflection_agent = Mock()
        mock_reflection_result = ReflectionResult(
            is_sufficient=True,
            knowledge_gaps=[],
            follow_up_queries=[],
            research_quality_score=0.9
        )
        mock_reflection_agent.assess_research_sufficiency.return_value = mock_reflection_result
        
        with patch('src.utils.researcher.enhanced_reflection.EnhancedReflectionAgent', return_value=mock_reflection_agent):
            enabler = ResearcherProgressiveEnabler(config=config)
            
            scenario = ScenarioContext(
                task_description="Test task",
                step_count=2,
                context_size=1000,
                parallel_execution=False,
                has_search_results=True,
                has_complex_queries=False,
                estimated_tokens=1500
            )
            
            # Mock historical scenarios to test fallback logic
            enabler.scenario_history = []
            
            # Mock metrics object
            mock_metrics = Mock()
            mock_metrics.get_isolation_health.return_value = {
                'estimated_token_savings': 1000,
                'total_isolation_sessions': 5,
                'recent_success_rate': 0.9
            }
            
            should_enable, reason, factors = enabler._adaptive_enablement(
                scenario, TaskComplexity.MEDIUM, config, mock_metrics, {}
            )
            
            # Should fall back to selective enablement logic
            assert 'reflection_insights' in factors
            assert factors['reflection_insights']['is_sufficient'] is True


class TestPhase2ContextExtensionIntegration:
    """Test reflection integration in ResearcherContextExtension"""
    
    def test_context_extension_with_reflection_config(self):
        """Test context extension initialization with reflection config"""
        config = Mock()
        config.enable_enhanced_reflection = True
        config.max_reflection_loops = 3
        
        with patch('src.utils.researcher.enhanced_reflection.EnhancedReflectionAgent') as mock_agent:
            extension = ResearcherContextExtension(config=config)
            
            assert extension.reflection_agent is not None
            mock_agent.assert_called_once()
    
    def test_context_extension_without_reflection_config(self):
        """Test context extension initialization without reflection config"""
        config = {'enable_enhanced_reflection': False}
        
        extension = ResearcherContextExtension(config=config)
        assert extension.reflection_agent is None
    
    def test_add_researcher_guidance_with_reflection(self):
        """Test researcher guidance enhancement with reflection insights"""
        config = Mock()
        config.enable_enhanced_reflection = True
        
        # Mock reflection agent
        mock_reflection_agent = Mock()
        mock_reflection_result = ReflectionResult(
            is_sufficient=False,
            knowledge_gaps=['Knowledge gap 1', 'Knowledge gap 2'],
            follow_up_queries=['Follow-up query 1', 'Follow-up query 2'],
            research_quality_score=0.6
        )
        mock_reflection_agent.assess_research_sufficiency.return_value = mock_reflection_result
        
        with patch('src.utils.researcher.enhanced_reflection.EnhancedReflectionAgent', return_value=mock_reflection_agent):
            extension = ResearcherContextExtension(config=config)
            
            # Mock state with plan and observations
            mock_plan = Mock()
            mock_plan.description = "Test plan description"
            mock_plan.steps = [Mock(execution_res="result1"), Mock(execution_res=None)]
            
            state = {
                'current_plan': mock_plan,
                'observations': ['obs1', 'obs2']
            }
            
            agent_input = {'messages': []}
            
            result = extension._add_researcher_guidance(agent_input, state)
            
            # Check that reflection guidance was added
            assert len(result['messages']) > 0
            
            # Find the reflection guidance message
            reflection_message = None
            for msg in result['messages']:
                if hasattr(msg, 'name') and msg.name == 'reflection_guidance':
                    reflection_message = msg
                    break
            
            assert reflection_message is not None
            assert 'Research Enhancement Suggestions' in reflection_message.content
            assert 'Follow-up query 1' in reflection_message.content
            assert 'Knowledge gap 1' in reflection_message.content
    
    def test_add_researcher_guidance_without_reflection_insights(self):
        """Test researcher guidance when reflection indicates sufficient knowledge"""
        config = Mock()
        config.enable_enhanced_reflection = True
        
        # Mock reflection agent with sufficient knowledge
        mock_reflection_agent = Mock()
        mock_reflection_result = ReflectionResult(
            is_sufficient=True,
            knowledge_gaps=[],
            follow_up_queries=[],
            research_quality_score=0.9
        )
        mock_reflection_agent.assess_research_sufficiency.return_value = mock_reflection_result
        
        with patch('src.utils.researcher.enhanced_reflection.EnhancedReflectionAgent', return_value=mock_reflection_agent):
            extension = ResearcherContextExtension(config=config)
            
            # Mock state
            mock_plan = Mock()
            mock_plan.description = "Test plan description"
            mock_plan.steps = []
            
            state = {
                'current_plan': mock_plan,
                'observations': []
            }
            
            agent_input = {'messages': []}
            
            result = extension._add_researcher_guidance(agent_input, state)
            
            # Should not add reflection guidance when knowledge is sufficient
            reflection_messages = [
                msg for msg in result['messages'] 
                if hasattr(msg, 'name') and msg.name == 'reflection_guidance'
            ]
            assert len(reflection_messages) == 0


class TestPhase2IsolationMetricsIntegration:
    """Test reflection metrics integration in ResearcherIsolationMetrics"""
    
    def test_isolation_session_with_reflection_fields(self):
        """Test IsolationSession with new reflection fields"""
        session = IsolationSession(
            session_id="test_session",
            start_time=1000.0,
            reflection_enabled=True,
            reflection_insights_count=5,
            knowledge_gaps_identified=3,
            follow_up_queries_generated=4,
            reflection_processing_time=0.5
        )
        
        assert session.reflection_enabled is True
        assert session.reflection_insights_count == 5
        assert session.knowledge_gaps_identified == 3
        assert session.follow_up_queries_generated == 4
        assert session.reflection_processing_time == 0.5
    
    def test_update_reflection_metrics(self):
        """Test updating reflection metrics for a session"""
        metrics = ResearcherIsolationMetrics()
        
        # Start a session
        session_id = "test_reflection_session"
        metrics.start_isolation_session(session_id)
        
        # Update reflection metrics
        metrics.update_reflection_metrics(
            session_id=session_id,
            insights_count=3,
            knowledge_gaps=2,
            follow_up_queries=4,
            processing_time=0.3
        )
        
        # Check session was updated
        session = metrics.sessions[session_id]
        assert session.reflection_enabled is True
        assert session.reflection_insights_count == 3
        assert session.knowledge_gaps_identified == 2
        assert session.follow_up_queries_generated == 4
        assert session.reflection_processing_time == 0.3
    
    def test_get_reflection_metrics_summary_empty(self):
        """Test reflection metrics summary with no reflection sessions"""
        metrics = ResearcherIsolationMetrics()
        
        summary = metrics.get_reflection_metrics_summary()
        
        assert summary['total_reflection_sessions'] == 0
        assert summary['average_insights_per_session'] == 0.0
        assert summary['average_knowledge_gaps'] == 0.0
        assert summary['average_follow_up_queries'] == 0.0
        assert summary['average_processing_time'] == 0.0
        assert summary['reflection_effectiveness'] == 0.0
    
    def test_get_reflection_metrics_summary_with_data(self):
        """Test reflection metrics summary with reflection session data"""
        metrics = ResearcherIsolationMetrics()
        
        # Create and complete reflection sessions
        for i in range(3):
            session_id = f"session_{i}"
            metrics.start_isolation_session(session_id)
            metrics.update_reflection_metrics(
                session_id=session_id,
                insights_count=i + 1,
                knowledge_gaps=i + 1,
                follow_up_queries=(i + 1) * 2,
                processing_time=0.1 * (i + 1)
            )
            metrics.end_isolation_session(session_id, success=True)
        
        summary = metrics.get_reflection_metrics_summary()
        
        assert summary['total_reflection_sessions'] == 3
        assert summary['average_insights_per_session'] == 2.0  # (1+2+3)/3
        assert summary['average_knowledge_gaps'] == 2.0  # (1+2+3)/3
        assert summary['average_follow_up_queries'] == 4.0  # (2+4+6)/3
        assert abs(summary['average_processing_time'] - 0.2) < 0.001  # (0.1+0.2+0.3)/3
        assert summary['reflection_effectiveness'] == 2.0  # 12 queries / 6 gaps
        assert summary['total_knowledge_gaps_identified'] == 6
        assert summary['total_follow_up_queries_generated'] == 12
    
    def test_aggregate_metrics_with_reflection(self):
        """Test aggregate metrics update with reflection data"""
        metrics = ResearcherIsolationMetrics()
        
        # Create and complete a reflection session
        session_id = "reflection_session"
        metrics.start_isolation_session(session_id)
        metrics.update_reflection_metrics(
            session_id=session_id,
            insights_count=5,
            knowledge_gaps=3,
            follow_up_queries=6,
            processing_time=0.4
        )
        metrics.end_isolation_session(session_id, success=True)
        
        # Check aggregate metrics were updated
        assert metrics.reflection_sessions == 1
        assert metrics.total_knowledge_gaps == 3
        assert metrics.total_follow_up_queries == 6
        assert metrics.average_reflection_time == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])