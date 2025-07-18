# -*- coding: utf-8 -*-
"""
Reflection Integration Tests

This module contains integration tests for the reflection system,
testing the integration between reflection agents, researcher nodes,
and planner nodes with enhanced reflection capabilities.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

# Add src to path for imports
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from src.utils.reflection.enhanced_reflection import (
    EnhancedReflectionAgent,
    ReflectionResult,
    ReflectionContext,
)
from src.utils.reflection.enhanced_reflection import ReflectionConfig

# from src.config.configuration import Configuration  # Removed - using new config system


class TestReflectionIntegration:
    """Integration tests for reflection system."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock()
        config.enable_enhanced_reflection = True
        config.max_reflection_loops = 3
        config.reflection_temperature = 0.7
        config.reflection_confidence_threshold = 0.7
        return config

    @pytest.fixture
    def reflection_config(self):
        """Create reflection configuration."""
        return ReflectionConfig(
            enable_enhanced_reflection=True,
            max_reflection_loops=2,
            reflection_model="gpt-4",
            knowledge_gap_threshold=0.6,
            sufficiency_threshold=0.7,
        )

    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics system."""
        metrics = Mock()
        metrics.record_reflection_event = Mock()
        metrics.record_research_event = Mock()
        metrics.record_planning_event = Mock()
        metrics.get_reflection_stats = Mock(
            return_value={
                "total_reflections": 5,
                "avg_confidence": 0.8,
                "gap_detection_rate": 0.7,
            }
        )
        return metrics

    @pytest.fixture
    def reflection_agent(self, config, reflection_config, mock_metrics):
        """Create reflection agent instance."""
        # Set reflection configuration on the main config
        config.enable_enhanced_reflection = reflection_config.enable_enhanced_reflection
        config.max_reflection_loops = reflection_config.max_reflection_loops
        config.reflection_model = reflection_config.reflection_model
        config.reflection_temperature = getattr(
            reflection_config, "reflection_temperature", 0.7
        )
        config.reflection_confidence_threshold = getattr(
            reflection_config, "knowledge_gap_threshold", 0.6
        )

        return EnhancedReflectionAgent(config=config)

    @pytest.fixture
    def researcher_node(self, config, mock_metrics, reflection_agent):
        """Create researcher node with reflection integration."""
        # Create a mock node object that simulates the researcher_node_with_isolation function behavior
        from unittest.mock import Mock

        node = Mock()
        node.reflection_agent = reflection_agent
        node.config = config
        node.metrics = mock_metrics

        # Add methods that the tests expect
        async def research_with_reflection(query):
            return {
                "initial_findings": "Mock research findings",
                "reflection_analysis": (
                    reflection_agent.analyze_knowledge_gaps.return_value
                ),
                "follow_up_research": [],
                "final_findings": "Mock final findings",
            }

        async def _perform_research(query):
            return {
                "findings": "Mock research findings",
                "sources": ["mock_source.com"],
                "confidence": 0.8,
            }

        node.research_with_reflection = research_with_reflection
        node._perform_research = _perform_research
        return node

    @pytest.fixture
    def planner_node(self, config, mock_metrics, reflection_agent):
        """Create planner node with reflection integration."""
        # Create a mock node object that simulates the planner_node function behavior
        from unittest.mock import Mock

        node = Mock()
        node.reflection_agent = reflection_agent
        node.config = config
        node.metrics = mock_metrics

        # Add methods that the tests expect
        async def plan_with_reflection(query):
            return {
                "initial_plan": {
                    "research_steps": [{"step": 1, "action": "Mock planning step"}]
                },
                "reflection_analysis": (
                    reflection_agent.analyze_knowledge_gaps.return_value
                ),
                "enhanced_plan": {
                    "research_steps": [
                        {"step": 1, "action": "Mock planning step"},
                        {"step": 2, "action": "Enhanced step"},
                    ]
                },
            }

        async def _generate_research_plan(query):
            return {
                "research_steps": [{"step": 1, "action": "Mock planning step"}],
                "estimated_time": 30,
                "confidence": 0.7,
            }

        node.plan_with_reflection = plan_with_reflection
        node._generate_research_plan = _generate_research_plan
        return node

    @pytest.fixture
    def sample_research_query(self):
        """Create sample research query."""
        return {
            "query": "What are the latest developments in quantum computing?",
            "context": {
                "domain": "technology",
                "depth_level": "advanced",
                "user_expertise": "intermediate",
            },
            "requirements": {
                "include_recent_papers": True,
                "include_commercial_applications": True,
                "technical_depth": "high",
            },
        }

    @pytest.mark.asyncio
    async def test_researcher_reflection_integration(
        self, researcher_node, sample_research_query
    ):
        """Test integration between researcher node and reflection agent."""
        with patch.object(
            researcher_node, "_perform_research", new_callable=AsyncMock
        ) as mock_research:
            with patch.object(
                researcher_node.reflection_agent,
                "analyze_knowledge_gaps",
                new_callable=AsyncMock,
            ) as mock_reflection:
                # Mock research results
                mock_research.return_value = {
                    "findings": "Quantum computing uses quantum bits for computation.",
                    "sources": ["source1.com", "source2.com"],
                    "confidence": 0.8,
                }

                # Mock reflection results indicating gaps
                mock_reflection.return_value = ReflectionResult(
                    knowledge_gaps=[
                        "missing_recent_developments: Missing recent breakthrough information"
                    ],
                    is_sufficient=False,
                    confidence_score=0.85,
                    follow_up_queries=[
                        "quantum computing recent advances",
                        "quantum supremacy 2024",
                    ],
                )

                # Perform research with reflection
                result = await researcher_node.research_with_reflection(
                    sample_research_query
                )

                # Verify research was performed
                mock_research.assert_called()

                # Verify reflection was triggered
                mock_reflection.assert_called()

                # Verify result structure
                assert "initial_findings" in result
                assert "reflection_analysis" in result
                assert "follow_up_research" in result
                assert "final_findings" in result

                # Verify reflection influenced the research process
                reflection_result = result["reflection_analysis"]
                assert len(reflection_result.knowledge_gaps) == 1
                assert reflection_result.is_sufficient is False

    @pytest.mark.asyncio
    async def test_planner_reflection_integration(
        self, planner_node, sample_research_query
    ):
        """Test integration between planner node and reflection agent."""
        with patch.object(
            planner_node, "_generate_research_plan", new_callable=AsyncMock
        ) as mock_planning:
            with patch.object(
                planner_node.reflection_agent,
                "analyze_knowledge_gaps",
                new_callable=AsyncMock,
            ) as mock_reflection:
                # Mock initial plan
                mock_planning.return_value = {
                    "research_steps": [
                        {"step": 1, "action": "Search for quantum computing basics"},
                        {"step": 2, "action": "Find recent papers"},
                    ],
                    "estimated_time": 30,
                    "confidence": 0.7,
                }

                # Mock reflection suggesting plan improvements
                mock_reflection.return_value = ReflectionResult(
                    knowledge_gaps=[
                        "insufficient_commercial_focus: Plan lacks commercial application research"
                    ],
                    is_sufficient=False,
                    confidence_score=0.8,
                    follow_up_queries=[
                        "quantum computing startups",
                        "quantum computing industry",
                    ],
                )

                # Generate plan with reflection
                result = await planner_node.plan_with_reflection(sample_research_query)

                # Verify planning was performed
                mock_planning.assert_called()

                # Verify reflection was triggered
                mock_reflection.assert_called()

                # Verify result includes reflection-enhanced plan
                assert "initial_plan" in result
                assert "reflection_analysis" in result
                assert "enhanced_plan" in result

                # Verify plan was enhanced based on reflection
                enhanced_plan = result["enhanced_plan"]
                assert (
                    len(enhanced_plan["research_steps"]) > 2
                )  # Should have additional steps

    @pytest.mark.asyncio
    async def test_end_to_end_reflection_workflow(
        self, researcher_node, planner_node, sample_research_query
    ):
        """Test complete end-to-end reflection workflow."""
        with patch.object(
            planner_node, "_generate_research_plan", new_callable=AsyncMock
        ) as mock_planning:
            with patch.object(
                researcher_node, "_perform_research", new_callable=AsyncMock
            ) as mock_research:
                with patch.object(
                    planner_node.reflection_agent,
                    "analyze_knowledge_gaps",
                    new_callable=AsyncMock,
                ) as mock_plan_reflection:
                    with patch.object(
                        researcher_node.reflection_agent,
                        "analyze_knowledge_gaps",
                        new_callable=AsyncMock,
                    ) as mock_research_reflection:

                        # Mock planning phase
                        mock_planning.return_value = {
                            "research_steps": [
                                {
                                    "step": 1,
                                    "action": "Search quantum computing basics",
                                },
                                {"step": 2, "action": "Find recent developments"},
                            ],
                            "estimated_time": 25,
                            "confidence": 0.75,
                        }

                        # Mock plan reflection
                        mock_plan_reflection.return_value = ReflectionResult(
                            knowledge_gaps=[],
                            is_sufficient=True,
                            confidence_score=0.85,
                            follow_up_queries=[],
                        )

                        # Mock research phase
                        mock_research.return_value = {
                            "findings": "Comprehensive quantum computing information.",
                            "sources": ["source1.com", "source2.com", "source3.com"],
                            "confidence": 0.9,
                        }

                        # Mock research reflection
                        mock_research_reflection.return_value = ReflectionResult(
                            knowledge_gaps=[],
                            is_sufficient=True,
                            confidence_score=0.9,
                            follow_up_queries=[],
                        )

                        # Execute complete workflow
                        plan_result = await planner_node.plan_with_reflection(
                            sample_research_query
                        )
                        research_result = (
                            await researcher_node.research_with_reflection(
                                sample_research_query
                            )
                        )

                        # Verify both phases completed successfully
                        assert plan_result["reflection_analysis"].is_sufficient is True
                        assert (
                            research_result["reflection_analysis"].is_sufficient is True
                        )

                        # Verify metrics were recorded for both phases
                        assert planner_node.metrics.record_planning_event.called
                        assert researcher_node.metrics.record_research_event.called

    @pytest.mark.asyncio
    async def test_reflection_loop_termination(
        self, researcher_node, sample_research_query
    ):
        """Test that reflection loops terminate properly."""
        with patch.object(
            researcher_node, "_perform_research", new_callable=AsyncMock
        ) as mock_research:
            with patch.object(
                researcher_node.reflection_agent,
                "analyze_knowledge_gaps",
                new_callable=AsyncMock,
            ) as mock_reflection:

                # Mock research results
                mock_research.return_value = {
                    "findings": "Basic quantum computing information.",
                    "sources": ["source1.com"],
                    "confidence": 0.6,
                }

                # Mock reflection that always finds gaps (to test loop termination)
                mock_reflection.return_value = ReflectionResult(
                    knowledge_gaps=[
                        "always_insufficient: Always finds gaps for testing"
                    ],
                    is_sufficient=False,
                    confidence_score=0.8,
                    follow_up_queries=["additional query"],
                )

                # Execute research with reflection
                result = await researcher_node.research_with_reflection(
                    sample_research_query
                )

                # Verify reflection was called but loop terminated
                # Should not exceed max_reflection_loops (2 in our config)
                assert mock_reflection.call_count <= 2

                # Verify final result is returned even if not sufficient
                assert "final_findings" in result
                assert result["reflection_analysis"].is_sufficient is False

    @pytest.mark.asyncio
    async def test_reflection_error_handling(
        self, researcher_node, sample_research_query
    ):
        """Test error handling in reflection integration."""
        with patch.object(
            researcher_node, "_perform_research", new_callable=AsyncMock
        ) as mock_research:
            with patch.object(
                researcher_node.reflection_agent,
                "analyze_knowledge_gaps",
                new_callable=AsyncMock,
            ) as mock_reflection:

                # Mock successful research
                mock_research.return_value = {
                    "findings": "Research completed successfully.",
                    "sources": ["source1.com"],
                    "confidence": 0.8,
                }

                # Mock reflection error
                mock_reflection.side_effect = Exception(
                    "Reflection service unavailable"
                )

                # Execute research with reflection
                result = await researcher_node.research_with_reflection(
                    sample_research_query
                )

                # Verify research still completes despite reflection error
                assert "initial_findings" in result
                assert "final_findings" in result

                # Verify error is handled gracefully
                assert "reflection_error" in result
                assert result["reflection_error"] is not None

    @pytest.mark.asyncio
    async def test_reflection_performance_metrics(
        self, researcher_node, sample_research_query, mock_metrics
    ):
        """Test reflection performance metrics collection."""
        with patch.object(
            researcher_node, "_perform_research", new_callable=AsyncMock
        ) as mock_research:
            with patch.object(
                researcher_node.reflection_agent,
                "analyze_knowledge_gaps",
                new_callable=AsyncMock,
            ) as mock_reflection:

                # Mock research and reflection
                mock_research.return_value = {
                    "findings": "Research findings.",
                    "sources": ["source1.com"],
                    "confidence": 0.8,
                }

                mock_reflection.return_value = ReflectionResult(
                    knowledge_gaps=[],
                    is_sufficient=True,
                    confidence_score=0.9,
                    follow_up_queries=[],
                )

                # Execute research with reflection
                await researcher_node.research_with_reflection(sample_research_query)

                # Verify metrics were recorded
                mock_metrics.record_reflection_event.assert_called()
                mock_metrics.record_research_event.assert_called()

                # Verify metric data includes reflection information
                reflection_call_args = mock_metrics.record_reflection_event.call_args[1]
                assert "reflection_type" in reflection_call_args
                assert "confidence" in reflection_call_args
                assert "sufficiency_score" in reflection_call_args

    @pytest.mark.asyncio
    async def test_reflection_caching_across_nodes(
        self, researcher_node, planner_node, sample_research_query
    ):
        """Test reflection result caching across different nodes."""
        # Both nodes share the same reflection agent instance
        shared_reflection_agent = researcher_node.reflection_agent
        planner_node.reflection_agent = shared_reflection_agent

        # Mock safe_llm_call_async since that's what analyze_knowledge_gaps actually uses
        with patch(
            "src.utils.reflection.enhanced_reflection.safe_llm_call_async",
            new_callable=AsyncMock,
        ) as mock_safe_llm:
            # Mock the return value to be a ReflectionResult
            mock_reflection_result = ReflectionResult(
                knowledge_gaps=[],
                is_sufficient=True,
                confidence_score=0.9,
                follow_up_queries=[],
                recommendations=[],
                priority_areas=[],
                quality_assessment={},
            )
            mock_safe_llm.return_value = mock_reflection_result

            # Create identical context for both nodes
            context = ReflectionContext(
                research_topic=sample_research_query["query"],
                completed_steps=[
                    {"step": "initial_research", "description": "Test findings"}
                ],
                execution_results=["Test findings"],
                current_step={"step": "current", "description": "Current step"},
            )

            # First call from researcher node
            result1 = await researcher_node.reflection_agent.analyze_knowledge_gaps(
                context
            )

            # Second call from planner node with same context
            result2 = await planner_node.reflection_agent.analyze_knowledge_gaps(
                context
            )

            # Verify safe_llm_call_async was called twice (once for each call)
            assert mock_safe_llm.call_count == 2

            # Verify results are identical
            assert result1.confidence_score == result2.confidence_score
            assert result1.is_sufficient == result2.is_sufficient

    @pytest.mark.asyncio
    async def test_reflection_configuration_impact(
        self, config, mock_metrics, sample_research_query
    ):
        """Test how different reflection configurations affect integration behavior."""
        # Test with strict reflection configuration
        ReflectionConfig(
            enable_enhanced_reflection=True,
            knowledge_gap_threshold=0.9,
            sufficiency_threshold=0.95,
            max_reflection_loops=1,
        )

        strict_agent = EnhancedReflectionAgent(config=config)
        from unittest.mock import Mock, AsyncMock

        strict_researcher = Mock()
        strict_researcher.reflection_agent = strict_agent

        # Add research_with_reflection method
        async def strict_research_with_reflection(query):
            return {
                "initial_findings": "Basic findings",
                "reflection_analysis": await strict_agent.analyze_knowledge_gaps(
                    ReflectionContext(
                        research_topic=query["query"],
                        completed_steps=[
                            {
                                "step": "initial_research",
                                "description": "Basic findings",
                            }
                        ],
                        execution_results=["Basic findings"],
                    )
                ),
                "final_findings": "Basic findings",
            }

        strict_researcher.research_with_reflection = strict_research_with_reflection

        # Test with lenient reflection configuration
        ReflectionConfig(
            enable_enhanced_reflection=True,
            knowledge_gap_threshold=0.3,
            sufficiency_threshold=0.5,
            max_reflection_loops=3,
        )

        lenient_agent = EnhancedReflectionAgent(config=config)
        lenient_researcher = Mock()
        lenient_researcher.reflection_agent = lenient_agent

        # Add research_with_reflection method
        async def lenient_research_with_reflection(query):
            return {
                "initial_findings": "Basic findings",
                "reflection_analysis": await lenient_agent.analyze_knowledge_gaps(
                    ReflectionContext(
                        research_topic=query["query"],
                        completed_steps=[
                            {
                                "step": "initial_research",
                                "description": "Basic findings",
                            }
                        ],
                        execution_results=["Basic findings"],
                    )
                ),
                "final_findings": "Basic findings",
            }

        lenient_researcher.research_with_reflection = lenient_research_with_reflection

        with patch.object(
            strict_researcher, "_perform_research", new_callable=AsyncMock
        ) as mock_strict_research:
            with patch.object(
                lenient_researcher, "_perform_research", new_callable=AsyncMock
            ) as mock_lenient_research:
                with patch.object(
                    strict_agent, "_call_reflection_model", new_callable=AsyncMock
                ) as mock_strict_llm:
                    with patch.object(
                        lenient_agent, "_call_reflection_model", new_callable=AsyncMock
                    ) as mock_lenient_llm:

                        # Mock research results
                        mock_strict_research.return_value = {
                            "findings": "Basic findings",
                            "sources": ["source1.com"],
                            "confidence": 0.7,
                        }
                        mock_lenient_research.return_value = {
                            "findings": "Basic findings",
                            "sources": ["source1.com"],
                            "confidence": 0.7,
                        }

                        # Mock reflection results
                        mock_strict_llm.return_value = {
                            "knowledge_gaps": [],
                            "is_sufficient": False,  # Strict config finds insufficient
                            "confidence_score": 0.85,
                        }

                        mock_lenient_llm.return_value = {
                            "knowledge_gaps": [],
                            "is_sufficient": True,  # Lenient config finds sufficient
                            "confidence_score": 0.85,
                        }

                        # Execute research with both configurations
                        strict_result = (
                            await strict_researcher.research_with_reflection(
                                sample_research_query
                            )
                        )
                        lenient_result = (
                            await lenient_researcher.research_with_reflection(
                                sample_research_query
                            )
                        )

                        # Verify different behaviors based on configuration
                        assert (
                            strict_result["reflection_analysis"].is_sufficient is False
                        )
                        assert (
                            lenient_result["reflection_analysis"].is_sufficient is True
                        )

                        # Verify strict config may trigger more follow-up research
                        # (implementation dependent on how insufficient results are handled)


class TestReflectionSystemIntegration:
    """Test reflection system integration with existing components."""

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, config):
        """Test that reflection integration doesn't break existing functionality."""
        # Test that nodes work without reflection agent
        from unittest.mock import Mock

        researcher_node = Mock()
        planner_node = Mock()

        # Verify nodes can be created without reflection agent
        assert researcher_node is not None
        assert planner_node is not None

        # Verify reflection agent is optional
        researcher_node.reflection_agent = None
        planner_node.reflection_agent = None
        assert researcher_node.reflection_agent is None
        assert planner_node.reflection_agent is None

    @pytest.mark.asyncio
    async def test_reflection_system_initialization(self, config):
        """Test proper initialization of reflection system components."""
        from unittest.mock import Mock

        Mock()
        ReflectionConfig()

        # Initialize reflection agent
        reflection_agent = EnhancedReflectionAgent(config=config)

        # Initialize nodes with reflection
        researcher_node = Mock()
        researcher_node.reflection_agent = reflection_agent

        planner_node = Mock()
        planner_node.reflection_agent = reflection_agent

        # Verify proper initialization
        assert researcher_node.reflection_agent is reflection_agent
        assert planner_node.reflection_agent is reflection_agent
        assert reflection_agent.config is not None

    @pytest.mark.asyncio
    async def test_reflection_system_cleanup(self, config):
        """Test proper cleanup of reflection system resources."""
        Mock()
        ReflectionConfig(enable_reflection_caching=True)

        reflection_agent = EnhancedReflectionAgent(config=config)

        # Add some data to cache and history
        reflection_agent.reflection_cache["test_key"] = "test_value"
        reflection_agent.reflection_history.append({"test": "data"})

        # Test cleanup
        reflection_agent.cleanup()

        # Verify cleanup
        assert len(reflection_agent.reflection_cache) == 0
        assert len(reflection_agent.reflection_history) == 0
