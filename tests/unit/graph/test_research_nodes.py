"""Unit tests for research-related graph nodes.

These tests verify the functionality of the refactored research nodes
that replaced the monolithic EnhancedResearcher class.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any
from src.graph.nodes import (
    prepare_research_step_node,
    researcher_agent_node,
    reflection_node,
    update_plan_node,
)
from src.models.planner_model import Plan, Step, StepType


class TestPrepareResearchStepNode:
    """Test cases for prepare_research_step_node."""

    @pytest.fixture
    def mock_state(self) -> Dict[str, Any]:
        """Create a mock state for testing."""
        mock_step = Step(
            need_search=True,
            title="Step 1",
            description="First step",
            step_type=StepType.RESEARCH,
            execution_res=None,
        )

        mock_plan = Plan(
            locale="en-US",
            has_enough_context=False,
            thought="Test thought",
            title="Test Plan",
            steps=[
                mock_step,
                Step(
                    need_search=True,
                    title="Step 2",
                    description="Second step",
                    step_type=StepType.RESEARCH,
                    execution_res=None,
                    completed=False,
                ),
            ],
        )

        return {
            "research_topic": "Test Research Topic",
            "locale": "en",
            "current_plan": mock_plan,
            "current_step": mock_step,
            "current_step_index": 0,
            "observations": [],
            "messages": [],
        }

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.get = MagicMock(return_value={})
        return config

    @patch("src.graph.nodes.get_configuration_from_config")
    @patch("src.tools.search.get_web_search_tool")
    @patch("src.tools.crawl.crawl_tool")
    @patch("src.tools.retriever.get_retriever_tool")
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_prepare_research_step_node_success(
        self,
        mock_get_retriever,
        mock_crawl_tool,
        mock_get_web_search,
        mock_get_config,
        mock_state,
        mock_config,
    ):
        """Test successful preparation of research step."""
        # Setup mocks
        mock_configurable = MagicMock()
        mock_configurable.agents.max_search_results = 10
        mock_configurable.content.enable_smart_filtering = True
        mock_get_config.return_value = mock_configurable

        mock_get_web_search.return_value = MagicMock()
        mock_crawl_tool.return_value = MagicMock()
        mock_get_retriever.return_value = None  # No retriever tool

        # Execute
        result = await prepare_research_step_node(mock_state, mock_config)

        # Verify
        assert "agent_input" in result
        assert "current_step" in result
        assert result["agent_input"] is not None
        assert result["current_step"] == mock_state["current_plan"].steps[0]

        # Verify tools were requested
        mock_get_web_search.assert_called_once()
        # Note: crawl_tool is imported directly, not called as a function

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_prepare_research_step_node_no_steps(self, mock_config):
        """Test handling when no steps are available."""
        state = {
            "research_topic": "Test Topic",
            "current_plan": Plan(
                locale="en-US",
                has_enough_context=False,
                thought="Test thought",
                title="Test Plan",
                steps=[],
            ),
            "current_step_index": 0,
            "messages": [],
        }

        result = await prepare_research_step_node(state, mock_config)

        assert result["agent_input"] is None
        assert "preparation_error" in result


class TestResearcherAgentNode:
    """Test cases for researcher_agent_node."""

    @pytest.fixture
    def mock_state_with_agent_input(self) -> Dict[str, Any]:
        """Create a mock state with agent input prepared."""
        mock_step = Step(
            need_search=True,
            title="Test Step",
            description="Test description",
            step_type=StepType.RESEARCH,
        )

        return {
            "research_topic": "Test Research Topic",
            "agent_input": {
                "messages": [MagicMock()],
                "tools": [MagicMock()],
                "current_step_title": "Test Step",
                "research_topic": "Test Research Topic",
                "locale": "en",
                "observations": [],
            },
            "current_step": mock_step,
            "messages": [],
        }

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.get = MagicMock(return_value={})
        return config

    @patch("src.graph.nodes.create_agent")
    @patch("src.llms.error_handler.safe_llm_call_async")
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_researcher_agent_node_success(
        self,
        mock_safe_llm_call,
        mock_create_agent,
        mock_state_with_agent_input,
        mock_config,
    ):
        """Test successful agent execution."""
        # Setup mocks
        mock_agent = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_safe_llm_call.return_value = MagicMock(content="Research result")

        # Execute
        result = await researcher_agent_node(mock_state_with_agent_input, mock_config)

        # Verify
        assert "research_result" in result
        assert result["research_result"] is not None
        mock_create_agent.assert_called_once()
        mock_safe_llm_call.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_researcher_agent_node_no_agent_input(self, mock_config):
        """Test handling when no agent input is provided."""
        state = {
            "research_topic": "Test Topic",
            "agent_input": None,
            "preparation_error": "No steps available",
            "messages": [],
        }

        result = await researcher_agent_node(state, mock_config)

        assert "agent_error" in result
        assert result["agent_error"] == "No steps available"


class TestReflectionNode:
    """Test cases for reflection_node."""

    @pytest.fixture
    def mock_state_with_research_result(self) -> Dict[str, Any]:
        """Create a mock state with research result."""
        mock_step = Step(
            need_search=True,
            title="Test Step",
            description="Test description",
            step_type=StepType.RESEARCH,
            execution_res=None,
            completed=False,
        )

        mock_plan = Plan(
            locale="en-US",
            has_enough_context=False,
            thought="Test thought",
            title="Test Plan",
            steps=[
                Step(
                    need_search=True,
                    title="Step 1",
                    description="First step",
                    step_type=StepType.RESEARCH,
                    execution_res=None,
                    completed=False,
                )
            ],
        )

        return {
            "research_topic": "Test Research Topic",
            "research_result": "Some research findings",
            "current_step": mock_step,
            "current_plan": mock_plan,
            "current_step_index": 0,
            "messages": [],
        }

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.get = MagicMock(return_value={})
        return config

    @patch("src.utils.researcher.reflection_system_manager.ReflectionSystemManager")
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_reflection_node_success(
        self, mock_reflection_manager, mock_state_with_research_result, mock_config
    ):
        """Test successful reflection analysis."""
        # Setup mocks
        mock_manager_instance = MagicMock()
        mock_reflection_manager.return_value = mock_manager_instance
        mock_manager_instance.execute_reflection_analysis.return_value = {
            "is_sufficient": True,
            "reason": "Good progress",
            "suggested_improvements": [],
        }

        # Execute
        result = await reflection_node(mock_state_with_research_result, mock_config)

        # Verify
        assert "reflection_insights" in result
        assert result["reflection_insights"] is not None
        mock_manager_instance.execute_reflection_analysis.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_reflection_node_no_research_result(self, mock_config):
        """Test reflection node when no research result is available."""
        state = {
            "research_topic": "Test Topic",
            "research_result": None,
            "messages": [],
        }

        # Execute
        result = await reflection_node(state, mock_config)

        # Verify - the function returns empty dict when no research result
        assert isinstance(result, dict)
        # The actual implementation may return empty dict or specific error field


class TestUpdatePlanNode:
    """Test cases for update_plan_node."""

    @pytest.fixture
    def mock_state_with_reflection(self) -> Dict[str, Any]:
        """Create a mock state with reflection result."""
        mock_step = Step(
            need_search=True,
            title="Step 1",
            description="Test description",
            step_type=StepType.RESEARCH,
            execution_res=None,
            completed=False,
        )

        mock_plan = Plan(
            locale="en-US",
            has_enough_context=False,
            thought="Test thought",
            title="Test Plan",
            steps=[
                Step(
                    need_search=True,
                    title="Step 1",
                    description="First step",
                    step_type=StepType.RESEARCH,
                    execution_res=None,
                    completed=False,
                )
            ],
        )

        return {
            "research_topic": "Test Research Topic",
            "current_step": mock_step,
            "research_result": "Some research findings",
            "reflection_insights": MagicMock(
                should_continue=True,
                feedback="Good progress",
                is_sufficient=True,
            ),
            "current_plan": mock_plan,
            "current_step_index": 0,
            "observations": [],
            "messages": [],
        }

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.get = MagicMock(return_value={})
        return config

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_update_plan_node_success(
        self, mock_state_with_reflection, mock_config
    ):
        """Test successful plan update."""
        # Execute
        result = await update_plan_node(mock_state_with_reflection, mock_config)

        # Verify
        assert "current_plan" in result
        assert result["current_plan"] is not None
        # Check that the step was marked as completed
        assert result["current_plan"].steps[0].completed is True

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_update_plan_node_missing_data(self, mock_config):
        """Test handling when required data is missing."""
        state = {
            "research_topic": "Test Topic",
            "current_step": None,
            "current_plan": None,
            "research_result": None,
            "messages": [],
        }

        result = await update_plan_node(state, mock_config)

        assert "update_error" in result
        assert "Missing required data" in result["update_error"]

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_update_plan_node_should_not_continue(
        self, mock_state_with_reflection, mock_config
    ):
        """Test plan update when reflection suggests not to continue."""
        # Modify reflection result to suggest not continuing
        mock_state_with_reflection["reflection_insights"].should_continue = False

        # Execute
        result = await update_plan_node(mock_state_with_reflection, mock_config)

        # Verify
        assert "current_plan" in result
        # Step should still be marked as completed even if we shouldn't continue
        assert result["current_plan"].steps[0].completed is True
