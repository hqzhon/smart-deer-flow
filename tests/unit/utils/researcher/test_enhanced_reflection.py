# -*- coding: utf-8 -*-
"""
Enhanced Reflection Agent Unit Tests

This module contains comprehensive unit tests for the Enhanced Reflection Agent,
testing knowledge gap analysis, sufficiency assessment, and reflection workflows.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from src.utils.reflection.enhanced_reflection import (
    EnhancedReflectionAgent,
    ReflectionResult,
    ReflectionContext
)
from src.config.configuration import Configuration


class TestEnhancedReflectionAgent:
    """Test cases for Enhanced Reflection Agent."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Configuration()
        config.enable_enhanced_reflection = True
        config.max_reflection_loops = 3
        config.reflection_temperature = 0.7
        config.reflection_confidence_threshold = 0.7
        return config
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics system."""
        metrics = Mock()
        metrics.record_reflection_event = Mock()
        metrics.get_reflection_stats = Mock(return_value={
            'total_reflections': 10,
            'avg_confidence': 0.85,
            'gap_detection_rate': 0.75
        })
        return metrics
    
    @pytest.fixture
    def reflection_agent(self, config):
        """Create reflection agent instance."""
        return EnhancedReflectionAgent(config)
    
    @pytest.fixture
    def sample_research_context(self):
        """Create sample research context."""
        return {
            'current_findings': 'Python is a programming language used for web development.',
            'original_query': 'What are the main applications of Python programming language?',
            'research_context': {
                'domain': 'technology',
                'depth_level': 'intermediate',
                'user_expertise': 'beginner'
            }
        }
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, reflection_agent, config):
        """Test reflection agent initialization."""
        assert reflection_agent is not None
        assert isinstance(reflection_agent, EnhancedReflectionAgent)
        assert reflection_agent.config == config
        assert reflection_agent.reflection_history == []
        assert reflection_agent.enable_enhanced_reflection == True
        assert reflection_agent.max_reflection_loops == 3
    
    @pytest.mark.asyncio
    async def test_knowledge_gap_analysis(self, reflection_agent):
        """Test knowledge gap analysis functionality."""
        # Create a reflection context
        context = ReflectionContext(
            research_topic="Python programming applications",
            completed_steps=[{"step": "Initial research", "description": "Basic Python info"}],
            execution_results=["Python is a programming language"],
            total_steps=3,
            current_step_index=1
        )
        
        # Mock the LLM call to return a structured result
        with patch.object(reflection_agent, '_get_reflection_model') as mock_model:
            mock_structured_model = Mock()
            mock_model.return_value.with_structured_output.return_value = mock_structured_model
            
            # Mock the safe_llm_call_async function
            with patch('src.utils.researcher.enhanced_reflection.safe_llm_call_async', new_callable=AsyncMock) as mock_safe_call:
                mock_result = ReflectionResult(
                    is_sufficient=False,
                    knowledge_gaps=["Missing data science applications", "Missing web development details"],
                    follow_up_queries=["What are Python data science libraries?", "How is Python used in web development?"],
                    confidence_score=0.85
                )
                mock_safe_call.return_value = mock_result
                
                result = await reflection_agent.analyze_knowledge_gaps(context)
                
                assert isinstance(result, ReflectionResult)
                assert result.is_sufficient is False
                assert len(result.knowledge_gaps) == 2
                assert len(result.follow_up_queries) == 2
                assert result.confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_sufficiency_assessment(self, reflection_agent):
        """Test research sufficiency assessment."""
        # Create a reflection context
        context = ReflectionContext(
            research_topic="Python programming applications",
            completed_steps=[{"step": "Initial research", "description": "Basic Python info"}],
            execution_results=["Python is a programming language used for web development"],
            total_steps=3,
            current_step_index=2
        )
        
        # Create a reflection result
        reflection_result = ReflectionResult(
            is_sufficient=False,
            knowledge_gaps=["machine learning", "data analysis", "automation"],
            confidence_score=0.65
        )
        
        is_sufficient, reasoning, details = reflection_agent.assess_sufficiency(context, reflection_result)
        
        assert isinstance(is_sufficient, bool)
        assert isinstance(reasoning, str)
        assert isinstance(details, dict)
        
        # Check details structure
        assert 'basic_sufficient' in details
        assert 'enhanced_sufficient' in details
        assert 'confidence_score' in details
        assert 'knowledge_gaps_count' in details
        assert details['knowledge_gaps_count'] == 3
        
        # Since we have execution results and are at the last step, basic_sufficient should be True
        assert details['basic_sufficient'] is True
        # This should make is_sufficient True regardless of enhanced assessment
        assert is_sufficient is True
    
    @pytest.mark.asyncio
    async def test_follow_up_query_generation(self, reflection_agent):
        """Test follow-up query generation."""
        knowledge_gaps = [
            "Missing web development details",
            "Shallow coverage of data science"
        ]
        
        # Mock the model call
        with patch.object(reflection_agent, '_get_reflection_model') as mock_model:
            mock_model.return_value = Mock()
            
            with patch.object(reflection_agent, '_call_reflection_model', new_callable=AsyncMock) as mock_call:
                # Mock response with follow-up queries
                class MockResponse:
                    content = '{"follow_up_queries": ["What are Python web frameworks?", "How is Python used in data science?"]}'
                
                mock_call.return_value = MockResponse()
                
                queries = await reflection_agent.generate_follow_up_queries(
                    "Python programming applications",
                    knowledge_gaps
                )
                
                assert len(queries) == 2
                assert any('web' in query.lower() for query in queries)
                assert any('data' in query.lower() for query in queries)
    
    @pytest.mark.asyncio
    async def test_reflection_metrics(self, reflection_agent):
        """Test reflection metrics collection."""
        # Initially no metrics
        metrics = reflection_agent.get_reflection_metrics()
        assert metrics['total_reflections'] == 0
        
        # Enable enhanced reflection for this test
        reflection_agent.enable_enhanced_reflection = True
        
        # Create a reflection context and result
        context = ReflectionContext(
            research_topic="Test topic",
            completed_steps=[],
            execution_results=["Some result"],
            total_steps=1,
            current_step_index=0
        )
        
        # Create a mock model that returns a valid response
        mock_model = Mock()
        mock_structured_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_model
        
        # Mock safe_llm_call_async to return a reflection result
        with patch.object(reflection_agent, '_get_reflection_model') as mock_get_model, \
             patch('src.utils.researcher.enhanced_reflection.safe_llm_call_async', new_callable=AsyncMock) as mock_safe_call:
            
            mock_get_model.return_value = mock_model
            mock_safe_call.return_value = ReflectionResult(
                is_sufficient=True,
                confidence_score=0.8,
                knowledge_gaps=["test gap"],
                follow_up_queries=["test query"]
            )
            
            result = await reflection_agent.analyze_knowledge_gaps(context)
            
            # Check metrics after reflection
            metrics = reflection_agent.get_reflection_metrics()
            assert metrics['total_reflections'] == 1
            assert 'sufficient_rate' in metrics
            assert 'average_confidence' in metrics
            assert metrics['sufficient_rate'] == 1.0
            assert metrics['average_confidence'] == 0.8
    
    @pytest.mark.asyncio
    async def test_reflection_result_structure(self, reflection_agent):
        """Test reflection result structure and completeness."""
        context = ReflectionContext(
            research_topic="Python programming",
            completed_steps=[{"step": "Research", "description": "Basic info"}],
            execution_results=["Python is a programming language"],
            total_steps=2,
            current_step_index=1
        )
        
        # Mock the reflection model to return None (triggers basic reflection)
        with patch.object(reflection_agent, '_get_reflection_model') as mock_model:
            mock_model.return_value = None
            
            result = await reflection_agent.analyze_knowledge_gaps(context)
            
            # Verify result structure
            assert isinstance(result, ReflectionResult)
            assert hasattr(result, 'knowledge_gaps')
            assert hasattr(result, 'is_sufficient')
            assert hasattr(result, 'follow_up_queries')
            assert hasattr(result, 'confidence_score')
            assert hasattr(result, 'recommendations')
            assert hasattr(result, 'priority_areas')
            
            # Verify types
            assert isinstance(result.knowledge_gaps, list)
            assert isinstance(result.follow_up_queries, list)
            assert isinstance(result.recommendations, list)
            assert isinstance(result.is_sufficient, bool)
            
    @pytest.mark.asyncio
    async def test_research_quality_analysis(self, reflection_agent):
        """Test research quality analysis functionality."""
        context = ReflectionContext(
            research_topic="Machine Learning",
            completed_steps=[
                {"step": "Initial research", "description": "Basic ML concepts"},
                {"step": "Deep dive", "description": "Advanced algorithms"}
            ],
            execution_results=[
                "Machine learning is a subset of artificial intelligence",
                "Popular algorithms include neural networks, decision trees, and SVM"
            ],
            total_steps=3,
            current_step_index=2
        )
        
        quality_metrics = await reflection_agent.analyze_research_quality(context)
        
        # Verify quality metrics structure
        assert 'completeness_score' in quality_metrics
        assert 'depth_score' in quality_metrics
        assert 'relevance_score' in quality_metrics
        assert 'overall_quality' in quality_metrics
        assert 'areas_for_improvement' in quality_metrics
        
        # Verify score ranges
        assert 0.0 <= quality_metrics['completeness_score'] <= 1.0
        assert 0.0 <= quality_metrics['depth_score'] <= 1.0
        assert 0.0 <= quality_metrics['relevance_score'] <= 1.0
        assert 0.0 <= quality_metrics['overall_quality'] <= 1.0
        
    @pytest.mark.asyncio
    async def test_research_sufficiency_assessment(self, reflection_agent):
        """Test research sufficiency assessment."""
        context = ReflectionContext(
            research_topic="Python web frameworks",
            completed_steps=[
                {"step": "Framework overview", "description": "Django and Flask comparison"}
            ],
            execution_results=[
                "Django is a high-level Python web framework. Flask is a micro web framework."
            ],
            total_steps=2,
            current_step_index=1
        )
        
        # Mock the analyze_knowledge_gaps method
        with patch.object(reflection_agent, 'analyze_knowledge_gaps', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = ReflectionResult(
                is_sufficient=True,
                confidence_score=0.85
            )
            
            result = await reflection_agent.assess_research_sufficiency(context)
            
            assert isinstance(result, ReflectionResult)
            mock_analyze.assert_called_once_with(context)
            assert result.is_sufficient is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, reflection_agent):
        """Test error handling in reflection analysis."""
        # Enable enhanced reflection for this test
        reflection_agent.enable_enhanced_reflection = True
        
        context = ReflectionContext(
            research_topic="Test topic",
            completed_steps=[],
            execution_results=["Some result"],
            total_steps=1,
            current_step_index=0
        )
        
        # Mock the model to raise an exception
        mock_model = Mock()
        mock_structured_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_model
        
        with patch.object(reflection_agent, '_get_reflection_model') as mock_get_model, \
             patch('src.utils.researcher.enhanced_reflection.safe_llm_call_async', new_callable=AsyncMock) as mock_safe_call:
            
            mock_get_model.return_value = mock_model
            mock_safe_call.side_effect = Exception("LLM API error")
            
            result = await reflection_agent.analyze_knowledge_gaps(context)
            
            # Should return default result on error
            assert isinstance(result, ReflectionResult)
            assert result.knowledge_gaps == []
            assert result.is_sufficient is True  # Default to sufficient on error
            assert result.confidence_score == 0.6  # Basic reflection confidence when sufficient
    
    @pytest.mark.asyncio
    async def test_reflection_history_tracking(self, reflection_agent):
        """Test reflection history tracking."""
        # Enable enhanced reflection for this test
        reflection_agent.enable_enhanced_reflection = True
        
        context1 = ReflectionContext(
            research_topic="Test topic",
            completed_steps=[],
            execution_results=["Some result"],
            total_steps=1,
            current_step_index=0
        )
        
        context2 = ReflectionContext(
            research_topic="Test topic",
            completed_steps=[],
            execution_results=["Updated result"],
            total_steps=1,
            current_step_index=0
        )
        
        # Create a mock model that returns a valid response
        mock_model = Mock()
        mock_structured_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_model
        
        with patch.object(reflection_agent, '_get_reflection_model') as mock_get_model, \
             patch('src.utils.researcher.enhanced_reflection.safe_llm_call_async', new_callable=AsyncMock) as mock_safe_call:
            
            mock_get_model.return_value = mock_model
            mock_safe_call.return_value = ReflectionResult(
                is_sufficient=True,
                confidence_score=0.8,
                knowledge_gaps=["test gap"],
                follow_up_queries=["test query"]
            )
            
            # Perform multiple reflections
            await reflection_agent.analyze_knowledge_gaps(context1)
            await reflection_agent.analyze_knowledge_gaps(context2)
            
            assert len(reflection_agent.reflection_history) == 2
            # Check that history contains tuples of (context, result)
            assert all(len(entry) == 2 for entry in reflection_agent.reflection_history)
            assert all(isinstance(entry[0], ReflectionContext) for entry in reflection_agent.reflection_history)
            assert all(isinstance(entry[1], ReflectionResult) for entry in reflection_agent.reflection_history)
    
    @pytest.mark.asyncio
    async def test_metrics_integration(self, reflection_agent, mock_metrics):
        """Test integration with metrics system."""
        # Enable enhanced reflection for this test
        reflection_agent.enable_enhanced_reflection = True
        
        context = ReflectionContext(
            research_topic="Test topic",
            completed_steps=[],
            execution_results=["Some result"],
            total_steps=1,
            current_step_index=0
        )
        
        # Create a mock model that returns a valid response
        mock_model = Mock()
        mock_structured_model = Mock()
        mock_model.with_structured_output.return_value = mock_structured_model
        
        with patch.object(reflection_agent, '_get_reflection_model') as mock_get_model, \
             patch('src.utils.researcher.enhanced_reflection.safe_llm_call_async', new_callable=AsyncMock) as mock_safe_call:
            
            mock_get_model.return_value = mock_model
            mock_safe_call.return_value = ReflectionResult(
                is_sufficient=True,
                confidence_score=0.8,
                knowledge_gaps=["test gap"],
                follow_up_queries=["test query"]
            )
            
            await reflection_agent.analyze_knowledge_gaps(context)
            
            # Verify reflection was performed and history updated
            assert len(reflection_agent.reflection_history) == 1
            
            # Verify metrics can be retrieved
            metrics = reflection_agent.get_reflection_metrics()
            assert metrics['total_reflections'] == 1
            assert metrics['sufficient_rate'] == 1.0
            assert metrics['average_confidence'] == 0.8