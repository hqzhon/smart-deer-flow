from collections import namedtuple
import json
import pytest
from unittest.mock import patch, MagicMock
from src.graph.nodes import planner_node
from src.graph.nodes import human_feedback_node
from src.graph.nodes import coordinator_node
from src.graph.nodes import reporter_node
from src.graph.nodes import _execute_agent_step
from src.graph.nodes import _setup_and_execute_agent_step
from src.graph.nodes import researcher_node

# 在这里 mock 掉 get_llm_by_type，避免 ValueError
with patch("src.llms.llm.get_llm_by_type", return_value=MagicMock()):
    from langgraph.types import Command
    from src.graph.nodes import background_investigation_node
    from src.config import SearchEngine
    from langchain_core.messages import HumanMessage


# Mock data
MOCK_SEARCH_RESULTS = [
    {"title": "Test Title 1", "content": "Test Content 1"},
    {"title": "Test Title 2", "content": "Test Content 2"},
]


@pytest.fixture
def mock_state():
    return {
        "messages": [HumanMessage(content="test query")],
        "research_topic": "test query",
        "background_investigation_results": None,
    }


@pytest.fixture
def mock_configurable():
    mock = MagicMock()
    mock.agents.max_search_results = 5
    mock.agents.max_plan_iterations = 3
    mock.agents.enable_deep_thinking = False
    mock.content.enable_smart_filtering = True
    mock.content.enable_content_summarization = False
    mock.model_token_limits = MagicMock()
    return mock


@pytest.fixture
def mock_config():
    # 你可以根据实际需要返回一个 MagicMock 或 dict
    return MagicMock()


@pytest.fixture
def patch_config_from_runnable_config(mock_configurable):
    with patch(
        "src.graph.nodes.get_configuration_from_config",
        return_value=mock_configurable,
    ):
        yield


@pytest.fixture
def mock_tavily_search():
    with patch("src.graph.nodes.LoggedTavilySearch") as mock:
        instance = mock.return_value
        instance.invoke.return_value = [
            {"title": "Test Title 1", "content": "Test Content 1"},
            {"title": "Test Title 2", "content": "Test Content 2"},
        ]
        yield mock


@pytest.fixture
def mock_web_search_tool():
    with patch("src.graph.nodes.get_web_search_tool") as mock:
        instance = mock.return_value
        instance.invoke.return_value = [
            {"title": "Test Title 1", "content": "Test Content 1"},
            {"title": "Test Title 2", "content": "Test Content 2"},
        ]
        yield mock


@pytest.mark.parametrize("search_engine", [SearchEngine.TAVILY.value, "other"])
def test_background_investigation_node_tavily(
    mock_state,
    mock_tavily_search,
    mock_web_search_tool,
    search_engine,
    patch_config_from_runnable_config,
    mock_config,
):
    """Test background_investigation_node with Tavily search engine"""
    # Mock get_settings to return the search engine
    mock_settings = MagicMock()
    mock_settings.tools.search_engine = search_engine
    mock_settings.agent_llm_map.planner = "basic"

    with (
        patch("src.graph.nodes.get_settings", return_value=mock_settings),
        patch(
            "src.graph.nodes.get_llm_by_type",
            return_value=MagicMock(model_name="test-model"),
        ),
        patch(
            "src.graph.nodes.safe_llm_call",
            side_effect=lambda func, *args, **kwargs: func(*args),
        ),
    ):

        result = background_investigation_node(mock_state, mock_config)

        # Verify the result structure
        assert isinstance(result, dict)

        # Verify the update contains background_investigation_results
        assert "background_investigation_results" in result

        # Parse and verify the JSON content
        results = result["background_investigation_results"]

        if search_engine == SearchEngine.TAVILY.value:
            mock_tavily_search.return_value.invoke.assert_called_once_with("test query")
            assert (
                results
                == "## Test Title 1\n\nTest Content 1\n\n## Test Title 2\n\nTest Content 2"
            )
        else:
            mock_web_search_tool.return_value.invoke.assert_called_once_with(
                "test query"
            )
            assert len(json.loads(results)) == 2


def test_background_investigation_node_malformed_response(
    mock_state, mock_tavily_search, patch_config_from_runnable_config, mock_config
):
    """Test background_investigation_node with malformed Tavily response"""
    # Mock get_settings to return Tavily search engine
    mock_settings = MagicMock()
    mock_settings.tools.search_engine = SearchEngine.TAVILY.value
    mock_settings.agent_llm_map.planner = "basic"

    with (
        patch("src.graph.nodes.get_settings", return_value=mock_settings),
        patch(
            "src.graph.nodes.get_llm_by_type",
            return_value=MagicMock(model_name="test-model"),
        ),
        patch(
            "src.graph.nodes.safe_llm_call",
            side_effect=lambda func, *args, **kwargs: func(*args),
        ),
    ):

        # Mock a malformed response
        mock_tavily_search.return_value.invoke.return_value = "invalid response"

        result = background_investigation_node(mock_state, mock_config)

        # Verify the result structure
        assert isinstance(result, dict)

        # Verify the update contains background_investigation_results
        assert "background_investigation_results" in result

        # Parse and verify the JSON content
        results = result["background_investigation_results"]
        assert results == ""


@pytest.fixture
def mock_plan():
    return {
        "has_enough_context": True,
        "title": "Test Plan",
        "thought": "Test Thought",
        "steps": [],
        "locale": "en-US",
    }


@pytest.fixture
def mock_state_planner():
    return {
        "messages": [HumanMessage(content="plan this")],
        "plan_iterations": 0,
        "enable_background_investigation": True,
        "background_investigation_results": "Background info",
    }


@pytest.fixture
def mock_configurable_planner():
    mock = MagicMock()
    mock.agents.max_plan_iterations = 3
    mock.agents.enable_deep_thinking = False
    return mock


@pytest.fixture
def patch_config_from_runnable_config_planner(mock_configurable_planner):
    with patch(
        "src.graph.nodes.get_configuration_from_config",
        return_value=mock_configurable_planner,
    ):
        yield


@pytest.fixture
def patch_apply_prompt_template():
    with patch(
        "src.graph.nodes.apply_prompt_template",
        return_value=[{"role": "user", "content": "plan this"}],
    ) as mock:
        yield mock


@pytest.fixture
def patch_repair_json_output():
    with patch("src.graph.nodes.repair_json_output", side_effect=lambda x: x) as mock:
        yield mock


@pytest.fixture
def patch_plan_model_validate():
    with patch("src.graph.nodes.Plan.model_validate", side_effect=lambda x: x) as mock:
        yield mock


@pytest.fixture
def patch_ai_message():
    AIMessage = namedtuple("AIMessage", ["content", "name"])
    with patch(
        "src.graph.nodes.AIMessage",
        side_effect=lambda content, name: AIMessage(content, name),
    ) as mock:
        yield mock


def test_planner_node_basic_has_enough_context(
    mock_state_planner,
    patch_config_from_runnable_config_planner,
    patch_apply_prompt_template,
    patch_repair_json_output,
    patch_plan_model_validate,
    patch_ai_message,
    mock_plan,
):
    # AGENT_LLM_MAP["planner"] == "basic" and not thinking mode
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
        patch("src.graph.nodes.safe_llm_call") as mock_safe_llm_call,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.planner = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = json.dumps(mock_plan)
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Mock safe_llm_call to return the mock response directly
        mock_safe_llm_call.return_value = mock_response

        result = planner_node(mock_state_planner, MagicMock())
        assert isinstance(result, Command)
        assert result.goto == "reporter"
        assert "current_plan" in result.update
        assert result.update["current_plan"]["has_enough_context"] is True
        assert result.update["messages"][0].name == "planner"


def test_planner_node_basic_not_enough_context(
    mock_state_planner,
    patch_config_from_runnable_config_planner,
    patch_apply_prompt_template,
    patch_repair_json_output,
    patch_plan_model_validate,
    patch_ai_message,
):
    # AGENT_LLM_MAP["planner"] == "basic" and not thinking mode
    plan = {
        "has_enough_context": False,
        "title": "Test Plan",
        "thought": "Test Thought",
        "steps": [],
        "locale": "en-US",
    }
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
        patch("src.graph.nodes.safe_llm_call") as mock_safe_llm_call,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.planner = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = json.dumps(plan)
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Mock safe_llm_call to return the mock response directly
        mock_safe_llm_call.return_value = mock_response

        result = planner_node(mock_state_planner, MagicMock())
        assert isinstance(result, Command)
        assert result.goto == "human_feedback"
        assert "current_plan" in result.update
        assert isinstance(result.update["current_plan"], str)
        assert result.update["messages"][0].name == "planner"


def test_planner_node_stream_mode_has_enough_context(
    mock_state_planner,
    patch_config_from_runnable_config_planner,
    patch_apply_prompt_template,
    patch_repair_json_output,
    patch_plan_model_validate,
    patch_ai_message,
    mock_plan,
):
    # AGENT_LLM_MAP["planner"] != "basic"
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
        patch("src.graph.nodes.safe_llm_call") as mock_safe_llm_call,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.planner = "gpt-4o"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        # Simulate streaming chunks
        chunk = MagicMock()
        chunk.content = json.dumps(mock_plan)
        mock_llm.stream.return_value = [chunk]
        mock_get_llm.return_value = mock_llm

        # Mock safe_llm_call to return a mock stream response
        mock_stream_response = MagicMock()
        mock_stream_response.content = json.dumps(mock_plan)
        mock_safe_llm_call.return_value = mock_stream_response

        result = planner_node(mock_state_planner, MagicMock())
        assert isinstance(result, Command)
        assert result.goto == "reporter"
        assert "current_plan" in result.update
        assert result.update["current_plan"]["has_enough_context"] is True


def test_planner_node_stream_mode_not_enough_context(
    mock_state_planner,
    patch_config_from_runnable_config_planner,
    patch_apply_prompt_template,
    patch_repair_json_output,
    patch_plan_model_validate,
    patch_ai_message,
):
    # AGENT_LLM_MAP["planner"] != "basic"
    plan = {
        "has_enough_context": False,
        "title": "Test Plan",
        "thought": "Test Thought",
        "steps": [],
        "locale": "en-US",
    }
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
        patch("src.graph.nodes.safe_llm_call") as mock_safe_llm_call,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.planner = "gpt-4o"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        chunk = MagicMock()
        chunk.content = json.dumps(plan)
        mock_llm.stream.return_value = [chunk]
        mock_get_llm.return_value = mock_llm

        # Mock safe_llm_call to return a mock stream response
        mock_stream_response = MagicMock()
        mock_stream_response.content = json.dumps(plan)
        mock_safe_llm_call.return_value = mock_stream_response

        result = planner_node(mock_state_planner, MagicMock())
        assert isinstance(result, Command)
        assert result.goto == "human_feedback"
        assert "current_plan" in result.update
        assert isinstance(result.update["current_plan"], str)


def test_planner_node_plan_iterations_exceeded(mock_state_planner):
    # plan_iterations >= max_plan_iterations
    state = dict(mock_state_planner)
    state["plan_iterations"] = 5
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type", return_value=MagicMock()),
        patch("src.graph.nodes.safe_llm_call") as mock_safe_llm_call,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.planner = "basic"
        mock_get_settings.return_value = mock_settings

        # Mock safe_llm_call to return a mock response
        mock_response = MagicMock()
        mock_safe_llm_call.return_value = mock_response

        result = planner_node(state, MagicMock())
        assert isinstance(result, Command)
        assert result.goto == "reporter"


def test_planner_node_json_decode_error_first_iteration(mock_state_planner):
    # Simulate JSONDecodeError on first iteration
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
        patch("src.graph.nodes.safe_llm_call") as mock_safe_llm_call,
        patch(
            "src.graph.nodes.json.loads",
            side_effect=json.JSONDecodeError("err", "doc", 0),
        ),
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.planner = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"bad": "json"'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Mock safe_llm_call to return the mock response directly
        mock_safe_llm_call.return_value = mock_response

        result = planner_node(mock_state_planner, MagicMock())
        assert isinstance(result, Command)
        assert result.goto == "__end__"


def test_planner_node_json_decode_error_second_iteration(mock_state_planner):
    # Simulate JSONDecodeError on second iteration
    state = dict(mock_state_planner)
    state["plan_iterations"] = 1
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
        patch("src.graph.nodes.safe_llm_call") as mock_safe_llm_call,
        patch(
            "src.graph.nodes.json.loads",
            side_effect=json.JSONDecodeError("err", "doc", 0),
        ),
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.planner = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"bad": "json"'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Mock safe_llm_call to return the mock response directly
        mock_safe_llm_call.return_value = mock_response

        result = planner_node(state, MagicMock())
        assert isinstance(result, Command)
        assert result.goto == "reporter"


# Patch Plan.model_validate and repair_json_output globally for these tests
@pytest.fixture(autouse=True)
def patch_plan_and_repair(monkeypatch):
    monkeypatch.setattr("src.graph.nodes.Plan.model_validate", lambda x: x)
    monkeypatch.setattr("src.graph.nodes.repair_json_output", lambda x: x)
    yield


@pytest.fixture
def mock_state_base():
    return {
        "current_plan": json.dumps(
            {
                "has_enough_context": True,
                "title": "Test Plan",
                "thought": "Test Thought",
                "steps": [],
                "locale": "en-US",
            }
        ),
        "plan_iterations": 0,
    }


def test_human_feedback_node_auto_accepted(monkeypatch, mock_state_base):
    # auto_accepted_plan True, should skip interrupt and parse plan
    state = dict(mock_state_base)
    state["auto_accepted_plan"] = True
    result = human_feedback_node(state)
    assert isinstance(result, Command)
    assert result.goto == "reporter"
    assert result.update["plan_iterations"] == 1
    assert result.update["current_plan"]["has_enough_context"] is True


def test_human_feedback_node_edit_plan(monkeypatch, mock_state_base):
    # interrupt returns [EDIT_PLAN]..., should return Command to planner
    state = dict(mock_state_base)
    state["auto_accepted_plan"] = False
    with patch("src.graph.nodes.interrupt", return_value="[EDIT_PLAN] Please revise"):
        result = human_feedback_node(state)
        assert isinstance(result, Command)
        assert result.goto == "planner"
        assert result.update["messages"][0].name == "feedback"
        assert "[EDIT_PLAN]" in result.update["messages"][0].content


def test_human_feedback_node_accepted(monkeypatch, mock_state_base):
    # interrupt returns [ACCEPTED]..., should proceed to parse plan
    state = dict(mock_state_base)
    state["auto_accepted_plan"] = False
    with patch("src.graph.nodes.interrupt", return_value="[ACCEPTED] Looks good!"):
        result = human_feedback_node(state)
        assert isinstance(result, Command)
        assert result.goto == "reporter"
        assert result.update["plan_iterations"] == 1
        assert result.update["current_plan"]["has_enough_context"] is True


def test_human_feedback_node_invalid_interrupt(monkeypatch, mock_state_base):
    # interrupt returns something else, should raise TypeError
    state = dict(mock_state_base)
    state["auto_accepted_plan"] = False
    with patch("src.graph.nodes.interrupt", return_value="RANDOM_FEEDBACK"):
        with pytest.raises(TypeError):
            human_feedback_node(state)


def test_human_feedback_node_json_decode_error_first_iteration(
    monkeypatch, mock_state_base
):
    # repair_json_output returns bad json, json.loads raises JSONDecodeError, plan_iterations=0
    state = dict(mock_state_base)
    state["auto_accepted_plan"] = True
    state["plan_iterations"] = 0
    with patch(
        "src.graph.nodes.json.loads", side_effect=json.JSONDecodeError("err", "doc", 0)
    ):
        result = human_feedback_node(state)
        assert isinstance(result, Command)
        assert result.goto == "__end__"


def test_human_feedback_node_json_decode_error_second_iteration(
    monkeypatch, mock_state_base
):
    # repair_json_output returns bad json, json.loads raises JSONDecodeError, plan_iterations>0
    state = dict(mock_state_base)
    state["auto_accepted_plan"] = True
    state["plan_iterations"] = 2
    with patch(
        "src.graph.nodes.json.loads", side_effect=json.JSONDecodeError("err", "doc", 0)
    ):
        result = human_feedback_node(state)
        assert isinstance(result, Command)
        assert result.goto == "reporter"


def test_human_feedback_node_not_enough_context(monkeypatch, mock_state_base):
    # Plan does not have enough context, should goto research_team
    plan = {
        "has_enough_context": False,
        "title": "Test Plan",
        "thought": "Test Thought",
        "steps": [],
        "locale": "en-US",
    }
    state = dict(mock_state_base)
    state["current_plan"] = json.dumps(plan)
    state["auto_accepted_plan"] = True
    result = human_feedback_node(state)
    assert isinstance(result, Command)
    assert result.goto == "research_team"
    assert result.update["plan_iterations"] == 1
    assert result.update["current_plan"]["has_enough_context"] is False


@pytest.fixture
def mock_state_coordinator():
    return {
        "messages": [{"role": "user", "content": "test"}],
        "locale": "en-US",
    }


@pytest.fixture
def mock_configurable_coordinator():
    mock = MagicMock()
    mock.resources = ["resource1", "resource2"]
    return mock


@pytest.fixture
def patch_config_from_runnable_config_coordinator(mock_configurable_coordinator):
    with patch(
        "src.graph.nodes.get_configuration_from_config",
        return_value=mock_configurable_coordinator,
    ):
        yield


@pytest.fixture
def patch_apply_prompt_template_coordinator():
    with patch(
        "src.graph.nodes.apply_prompt_template",
        return_value=[{"role": "user", "content": "test"}],
    ) as mock:
        yield mock


@pytest.fixture
def patch_handoff_to_planner():
    with patch("src.graph.nodes.handoff_to_planner", MagicMock()):
        yield


@pytest.fixture
def patch_logger():
    with patch("src.graph.nodes.logger") as mock_logger:
        yield mock_logger


def make_mock_llm_response(tool_calls=None):
    resp = MagicMock()
    resp.tool_calls = tool_calls or []
    return resp


def test_coordinator_node_no_tool_calls(
    mock_state_coordinator,
    patch_config_from_runnable_config_coordinator,
    patch_apply_prompt_template_coordinator,
    patch_handoff_to_planner,
    patch_logger,
):
    # No tool calls, should goto __end__
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.coordinator = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = make_mock_llm_response([])
        mock_get_llm.return_value = mock_llm

        result = coordinator_node(mock_state_coordinator, MagicMock())
        assert result.goto == "__end__"
        assert result.update["locale"] == "en-US"
        assert result.update["resources"] == ["resource1", "resource2"]


def test_coordinator_node_with_tool_calls_planner(
    mock_state_coordinator,
    patch_config_from_runnable_config_coordinator,
    patch_apply_prompt_template_coordinator,
    patch_handoff_to_planner,
    patch_logger,
):
    # tool_calls present, should goto planner
    tool_calls = [{"name": "handoff_to_planner", "args": {}}]
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.coordinator = "basic"
        mock_get_settings.return_value = mock_settings

        mock_settings = MagicMock()
        mock_settings.agent_llm_map.coordinator = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = make_mock_llm_response(tool_calls)
        mock_get_llm.return_value = mock_llm

        result = coordinator_node(mock_state_coordinator, MagicMock())
        assert result.goto == "planner"
        assert result.update["locale"] == "en-US"
        assert result.update["resources"] == ["resource1", "resource2"]


def test_coordinator_node_with_tool_calls_background_investigator(
    mock_state_coordinator,
    patch_config_from_runnable_config_coordinator,
    patch_apply_prompt_template_coordinator,
    patch_handoff_to_planner,
    patch_logger,
):
    # enable_background_investigation True, should goto background_investigator
    state = dict(mock_state_coordinator)
    state["enable_background_investigation"] = True
    tool_calls = [{"name": "handoff_to_planner", "args": {}}]
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.coordinator = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = make_mock_llm_response(tool_calls)
        mock_get_llm.return_value = mock_llm

        result = coordinator_node(state, MagicMock())
        assert result.goto == "background_investigator"
        assert result.update["locale"] == "en-US"
        assert result.update["resources"] == ["resource1", "resource2"]


def test_coordinator_node_with_tool_calls_locale_override(
    mock_state_coordinator,
    patch_config_from_runnable_config_coordinator,
    patch_apply_prompt_template_coordinator,
    patch_handoff_to_planner,
    patch_logger,
):
    # tool_calls with locale in args should override locale
    tool_calls = [
        {
            "name": "handoff_to_planner",
            "args": {"locale": "zh-CN", "research_topic": "test topic"},
        }
    ]
    with (patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = make_mock_llm_response(tool_calls)
        mock_get_llm.return_value = mock_llm

        result = coordinator_node(mock_state_coordinator, MagicMock())
        assert result.goto == "planner"
        assert result.update["locale"] == "zh-CN"
        assert result.update["research_topic"] == "test topic"
        assert result.update["resources"] == ["resource1", "resource2"]
        assert result.update["resources"] == ["resource1", "resource2"]


def test_coordinator_node_tool_calls_exception_handling(
    mock_state_coordinator,
    patch_config_from_runnable_config_coordinator,
    patch_apply_prompt_template_coordinator,
    patch_handoff_to_planner,
    patch_logger,
):
    # tool_calls raises exception in processing
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.coordinator = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Simulate tool_call.get("args", {}) raising AttributeError
        class BadToolCall(dict):
            def get(self, key, default=None):
                if key == "args":
                    raise Exception("bad args")
                return super().get(key, default)

        mock_llm.invoke.return_value = make_mock_llm_response(
            [BadToolCall({"name": "handoff_to_planner"})]
        )
        mock_get_llm.return_value = mock_llm

        # Should not raise, just log error and continue
        result = coordinator_node(mock_state_coordinator, MagicMock())
        assert result.goto == "planner"
        assert result.update["locale"] == "en-US"
        assert result.update["resources"] == ["resource1", "resource2"]


@pytest.fixture
def mock_state_reporter():
    # Simulate a plan object with title and thought attributes
    Plan = namedtuple("Plan", ["title", "thought"])
    return {
        "current_plan": Plan(title="Test Title", thought="Test Thought"),
        "locale": "en-US",
        "observations": [],
    }


@pytest.fixture
def mock_state_reporter_with_observations():
    Plan = namedtuple("Plan", ["title", "thought"])
    return {
        "current_plan": Plan(title="Test Title", thought="Test Thought"),
        "locale": "en-US",
        "observations": ["Observation 1", "Observation 2"],
    }


@pytest.fixture
def mock_configurable_reporter():
    mock = MagicMock()
    return mock


@pytest.fixture
def patch_config_from_runnable_config_reporter(mock_configurable_reporter):
    with patch(
        "src.graph.nodes.get_configuration_from_config",
        return_value=mock_configurable_reporter,
    ):
        yield


@pytest.fixture
def patch_apply_prompt_template_reporter():
    with patch(
        "src.graph.nodes.apply_prompt_template",
        side_effect=lambda *args, **kwargs: [MagicMock()],
    ) as mock:
        yield mock


@pytest.fixture
def patch_human_message():
    HumanMessage = MagicMock()
    with patch("src.graph.nodes.HumanMessage", HumanMessage):
        yield HumanMessage


@pytest.fixture
def patch_logger_reporter():
    with patch("src.graph.nodes.logger") as mock_logger:
        yield mock_logger


def make_mock_llm_response_reporter(content):
    resp = MagicMock()
    resp.content = content
    return resp


def test_reporter_node_basic(
    mock_state_reporter,
    patch_config_from_runnable_config_reporter,
    patch_apply_prompt_template_reporter,
    patch_human_message,
    patch_logger_reporter,
):
    # Patch get_llm_by_type and AGENT_LLM_MAP
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.reporter = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_llm_response_reporter(
            "Final Report Content"
        )
        mock_get_llm.return_value = mock_llm

        result = reporter_node(mock_state_reporter, MagicMock())
        assert isinstance(result, dict)
        assert "final_report" in result
        assert result["final_report"] == "Final Report Content"
        # Should call apply_prompt_template with correct arguments
        patch_apply_prompt_template_reporter.assert_called()
        # Should call invoke on the LLM
        mock_llm.invoke.assert_called()


def test_reporter_node_with_observations(
    mock_state_reporter_with_observations,
    patch_config_from_runnable_config_reporter,
    patch_apply_prompt_template_reporter,
    patch_human_message,
    patch_logger_reporter,
):
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.reporter = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_llm_response_reporter(
            "Report with Observations"
        )
        mock_get_llm.return_value = mock_llm

        result = reporter_node(mock_state_reporter_with_observations, MagicMock())
        assert isinstance(result, dict)
        assert "final_report" in result
        assert result["final_report"] == "Report with Observations"
        # Should call apply_prompt_template with correct arguments
        patch_apply_prompt_template_reporter.assert_called()
        # Should call invoke on the LLM
        mock_llm.invoke.assert_called()


def test_reporter_node_locale_default(
    patch_config_from_runnable_config_reporter,
    patch_apply_prompt_template_reporter,
    patch_human_message,
    patch_logger_reporter,
):
    # If locale is missing, should default to "en-US"
    Plan = namedtuple("Plan", ["title", "thought"])
    state = {
        "current_plan": Plan(title="Test Title", thought="Test Thought"),
        # "locale" omitted
        "observations": [],
    }
    with (
        patch("src.graph.nodes.get_settings") as mock_get_settings,
        patch("src.graph.nodes.get_llm_by_type") as mock_get_llm,
    ):
        mock_settings = MagicMock()
        mock_settings.agent_llm_map.reporter = "basic"
        mock_get_settings.return_value = mock_settings

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_llm_response_reporter(
            "Default Locale Report"
        )
        mock_get_llm.return_value = mock_llm

        result = reporter_node(state, MagicMock())
        assert isinstance(result, dict)
        assert "final_report" in result
        assert result["final_report"] == "Default Locale Report"


# Create the real Step class for the tests
class Step:
    def __init__(self, title, description, execution_res=None):
        self.title = title
        self.description = description
        self.execution_res = execution_res

    def copy(self, update=None):
        """Create a copy of the step with optional updates."""
        new_step = Step(
            title=self.title,
            description=self.description,
            execution_res=self.execution_res,
        )
        if update:
            for key, value in update.items():
                setattr(new_step, key, value)
        return new_step


@pytest.fixture
def mock_step():
    return Step(title="Step 1", description="Desc 1", execution_res=None)


@pytest.fixture
def mock_completed_step():
    return Step(title="Step 0", description="Desc 0", execution_res="Done")


class Plan:
    def __init__(self, steps):
        self.steps = steps

    def copy(self, update=None):
        """Create a copy of the plan with optional updates."""
        new_plan = Plan(steps=self.steps[:])
        if update:
            for key, value in update.items():
                setattr(new_plan, key, value)
        return new_plan


@pytest.fixture
def mock_state_with_steps(mock_step, mock_completed_step):
    # Simulate a plan with one completed and one unexecuted step
    plan = Plan(steps=[mock_completed_step, mock_step])
    return {
        "current_plan": plan,
        "observations": ["obs1"],
        "locale": "en-US",
        "resources": [],
    }


@pytest.fixture
def mock_state_no_unexecuted():
    Step = namedtuple("Step", ["title", "description", "execution_res"])
    Plan = MagicMock()
    Plan.steps = [
        Step(title="Step 1", description="Desc 1", execution_res="done"),
        Step(title="Step 2", description="Desc 2", execution_res="done"),
    ]
    return {
        "current_plan": Plan,
        "observations": [],
        "locale": "en-US",
        "resources": [],
    }


@pytest.fixture
def mock_agent():
    agent = MagicMock()

    async def ainvoke(input, config):
        # Simulate agent returning a message list
        return {"messages": [MagicMock(content="result content")]}

    agent.ainvoke = ainvoke
    return agent


@pytest.mark.asyncio
async def test_execute_agent_step_basic(mock_state_with_steps, mock_agent):
    # Should execute the first unexecuted step and update execution_res
    with patch(
        "src.graph.nodes.HumanMessage",
        side_effect=lambda content, name=None: MagicMock(content=content, name=name),
    ):
        result = await _execute_agent_step(
            mock_state_with_steps, MagicMock(), mock_agent, "researcher"
        )
        assert isinstance(result, Command)
        assert result.goto == "research_team"
        assert "messages" in result.update
        assert "observations" in result.update
        # The new observation should be appended
        print(f"Actual observation: {repr(result.update['observations'][-1])}")
        assert "Step: Step 1" in result.update["observations"][-1]
        assert "Agent: researcher" in result.update["observations"][-1]
        # Content may be compressed, so check for either full content or compressed version
        observation = result.update["observations"][-1]
        assert "result content" in observation or "Result" in observation
        # The step's execution_res should be updated in the returned plan
        updated_plan = result.update["current_plan"]
        assert updated_plan.steps[1].execution_res == "result content"


@pytest.mark.asyncio
async def test_execute_agent_step_no_unexecuted_step(
    mock_state_no_unexecuted, mock_agent
):
    # Should return Command with goto="research_team" and not fail
    with patch("src.graph.nodes.logger") as mock_logger:
        result = await _execute_agent_step(
            mock_state_no_unexecuted, MagicMock(), mock_agent, "researcher"
        )
        assert isinstance(result, Command)
        assert result.goto == "research_team"
        mock_logger.warning.assert_called_with("No unexecuted step found")


@pytest.mark.asyncio
async def test_execute_agent_step_with_resources_and_researcher(mock_step):
    # Should add resource info and citation reminder for researcher
    Resource = namedtuple("Resource", ["title", "description"])
    resources = [Resource(title="file1.txt", description="desc1")]
    plan = Plan(steps=[mock_step])
    state = {
        "current_plan": plan,
        "observations": [],
        "locale": "en-US",
        "resources": resources,
    }
    agent = MagicMock()

    # Mock safe_llm_call_async to capture the agent_state
    async def mock_safe_llm_call(func, input, **kwargs):
        # Check that resource info and citation reminder are present in agent_state
        resources_info = input.get("resources_info", "")
        citation_reminder = input.get("citation_reminder", "")
        assert (
            "local_search_tool" in resources_info
        ), f"Expected local_search_tool in resources_info: {resources_info}"
        assert (
            "DO NOT include inline citations" in citation_reminder
        ), f"Expected citation reminder: {citation_reminder}"
        return MagicMock(content="resource result")

    with patch("src.graph.nodes.safe_llm_call_async", side_effect=mock_safe_llm_call):
        result = await _execute_agent_step(state, MagicMock(), agent, "researcher")
        assert isinstance(result, Command)
        assert result.goto == "research_team"
        assert "Step: Step 1" in result.update["observations"][-1]
        assert "Agent: researcher" in result.update["observations"][-1]
        # Check for either full content or compressed content
        observation = result.update["observations"][-1]
        assert "resource result" in observation or "Result" in observation


@pytest.mark.asyncio
async def test_execute_agent_step_recursion_limit_env(
    monkeypatch, mock_state_with_steps, mock_agent
):
    # Should respect AGENT_RECURSION_LIMIT env variable if set and valid
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "42")
    with (
        patch("src.graph.nodes.logger") as mock_logger,
        patch(
            "src.graph.nodes.HumanMessage",
            side_effect=lambda content, name=None: MagicMock(
                content=content, name=name
            ),
        ),
    ):
        result = await _execute_agent_step(
            mock_state_with_steps, MagicMock(), mock_agent, "coder"
        )
        assert isinstance(result, Command)
        mock_logger.info.assert_any_call("Recursion limit set to: 42")


@pytest.mark.asyncio
async def test_execute_agent_step_recursion_limit_env_invalid(
    monkeypatch, mock_state_with_steps, mock_agent
):
    # Should fallback to default if env variable is invalid
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "notanint")
    with (
        patch("src.graph.nodes.logger") as mock_logger,
        patch(
            "src.graph.nodes.HumanMessage",
            side_effect=lambda content, name=None: MagicMock(
                content=content, name=name
            ),
        ),
    ):
        result = await _execute_agent_step(
            mock_state_with_steps, MagicMock(), mock_agent, "coder"
        )
        assert isinstance(result, Command)
        mock_logger.warning.assert_any_call(
            "Invalid AGENT_RECURSION_LIMIT value: 'notanint'. Using default value 10."
        )


@pytest.mark.asyncio
async def test_execute_agent_step_recursion_limit_env_negative(
    monkeypatch, mock_state_with_steps, mock_agent
):
    # Should fallback to default if env variable is negative or zero
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "-5")
    with (
        patch("src.graph.nodes.logger") as mock_logger,
        patch(
            "src.graph.nodes.HumanMessage",
            side_effect=lambda content, name=None: MagicMock(
                content=content, name=name
            ),
        ),
    ):
        result = await _execute_agent_step(
            mock_state_with_steps, MagicMock(), mock_agent, "coder"
        )
        assert isinstance(result, Command)
        mock_logger.warning.assert_any_call(
            "AGENT_RECURSION_LIMIT value '-5' (parsed as -5) is not positive. Using default value 10."
        )


# Removed duplicate fixtures - already defined earlier in the file


@pytest.fixture
def mock_configurable_with_mcp():
    mock = MagicMock()
    mock.mcp = MagicMock()
    mock.mcp.enabled = True
    mock.mcp.servers = [
        {
            "enabled_tools": ["toolA", "toolB"],
            "add_to_agents": ["researcher"],
            "transport": "http",
            "command": "run",
            "args": {},
            "url": "http://localhost",
            "env": {},
            "other": "ignore",
        }
    ]
    return mock


@pytest.fixture
def mock_configurable_without_mcp():
    mock = MagicMock()
    mock.mcp = MagicMock()
    mock.mcp.enabled = False
    mock.mcp.servers = []
    return mock


@pytest.fixture
def patch_config_from_runnable_config_with_mcp(mock_configurable_with_mcp):
    with patch(
        "src.graph.nodes.get_configuration_from_config",
        return_value=mock_configurable_with_mcp,
    ):
        yield


@pytest.fixture
def patch_config_from_runnable_config_without_mcp(mock_configurable_without_mcp):
    with patch(
        "src.graph.nodes.get_configuration_from_config",
        return_value=mock_configurable_without_mcp,
    ):
        yield


@pytest.fixture
def patch_create_agent():
    with patch("src.agents.create_agent") as mock:
        yield mock


@pytest.fixture
def patch_execute_agent_step():
    async def fake_execute_agent_step(state, config, agent, agent_name):
        return "EXECUTED"

    with patch(
        "src.graph.nodes._execute_agent_step", side_effect=fake_execute_agent_step
    ) as mock:
        yield mock


@pytest.fixture
def patch_multiserver_mcp_client():
    # Patch MultiServerMCPClient as async context manager
    class FakeTool:
        def __init__(self, name, description="desc"):
            self.name = name
            self.description = description

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get_tools(self):
            return [
                FakeTool("toolA", "descA"),
                FakeTool("toolB", "descB"),
                FakeTool("toolC", "descC"),
            ]

    with patch(
        "src.graph.nodes.MultiServerMCPClient", return_value=FakeClient()
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_setup_and_execute_agent_step_with_mcp(
    mock_state_with_steps,
    mock_config,
    patch_config_from_runnable_config_with_mcp,
    patch_create_agent,
    patch_execute_agent_step,
    patch_multiserver_mcp_client,
):
    # Should use MCP client, load tools, and call create_agent with correct tools
    from langchain_core.tools import tool

    @tool
    def default_tool(query: str) -> str:
        """A default tool for testing."""
        return "default result"

    default_tools = [default_tool]
    agent_type = "researcher"

    result = await _setup_and_execute_agent_step(
        mock_state_with_steps,
        mock_config,
        agent_type,
        default_tools,
    )
    # Should call create_agent with loaded_tools including toolA and toolB
    args, kwargs = patch_create_agent.call_args
    loaded_tools = args[2]
    tool_names = [t.name for t in loaded_tools if hasattr(t, "name")]
    assert "toolA" in tool_names
    assert "toolB" in tool_names
    # Should call _execute_agent_step
    patch_execute_agent_step.assert_called_once()
    assert result == "EXECUTED"


@pytest.mark.asyncio
async def test_setup_and_execute_agent_step_without_mcp(
    mock_state_with_steps,
    mock_config,
    patch_config_from_runnable_config_without_mcp,
    patch_create_agent,
    patch_execute_agent_step,
):
    # Should use default tools and not use MCP client
    from langchain_core.tools import tool

    @tool
    def default_tool(query: str) -> str:
        """A default tool for testing."""
        return "default result"

    default_tools = [default_tool]
    agent_type = "coder"

    result = await _setup_and_execute_agent_step(
        mock_state_with_steps,
        mock_config,
        agent_type,
        default_tools,
    )
    # Should call create_agent with default_tools
    args, kwargs = patch_create_agent.call_args
    assert args[2] == default_tools
    patch_execute_agent_step.assert_called_once()
    assert result == "EXECUTED"


@pytest.mark.asyncio
async def test_setup_and_execute_agent_step_with_mcp_no_enabled_tools(
    mock_state_with_steps,
    mock_config,
    patch_create_agent,
    patch_execute_agent_step,
):
    # If mcp_settings present but no enabled_tools for agent_type, should fallback to default_tools
    mcp_settings = {
        "servers": {
            "server1": {
                "enabled_tools": ["toolA"],
                "add_to_agents": ["other_agent"],
                "transport": "http",
                "command": "run",
                "args": {},
                "url": "http://localhost",
                "env": {},
            }
        }
    }
    configurable = MagicMock()
    configurable.mcp = MagicMock()
    configurable.mcp.enabled = True
    configurable.mcp.servers = [mcp_settings["servers"]["server1"]]
    with patch(
        "src.graph.nodes.get_configuration_from_config",
        return_value=configurable,
    ):
        from langchain_core.tools import tool

        @tool
        def default_tool(query: str) -> str:
            """A default tool for testing."""
            return "default result"

        default_tools = [default_tool]
        agent_type = "researcher"
        result = await _setup_and_execute_agent_step(
            mock_state_with_steps,
            mock_config,
            agent_type,
            default_tools,
        )
        args, kwargs = patch_create_agent.call_args
        assert args[2] == default_tools
        patch_execute_agent_step.assert_called_once()
        assert result == "EXECUTED"


@pytest.mark.asyncio
async def test_setup_and_execute_agent_step_with_mcp_tools_description_update(
    mock_state_with_steps,
    mock_config,
    patch_config_from_runnable_config_with_mcp,
    patch_create_agent,
    patch_execute_agent_step,
):
    # Should update tool.description with Powered by info
    from langchain_core.tools import tool

    @tool
    def default_tool(query: str) -> str:
        """A default tool for testing."""
        return "default result"

    default_tools = [default_tool]
    agent_type = "researcher"

    # Patch MultiServerMCPClient to check description update
    class FakeTool:
        def __init__(self, name, description="desc"):
            self.name = name
            self.description = description

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get_tools(self):
            return [FakeTool("toolA", "descA")]

    with patch(
        "langchain_mcp_adapters.client.MultiServerMCPClient", return_value=FakeClient()
    ):
        await _setup_and_execute_agent_step(
            mock_state_with_steps,
            mock_config,
            agent_type,
            default_tools,
        )
        # The tool description should be updated
        args, kwargs = patch_create_agent.call_args
        loaded_tools = args[2]
        found = False
        for t in loaded_tools:
            if hasattr(t, "name") and t.name == "toolA":
                assert t.description.startswith("Powered by 'server1'.\n")
                found = True
        assert found


@pytest.fixture
def mock_state_with_resources():
    return {"resources": ["resource1", "resource2"], "other": "value"}


@pytest.fixture
def mock_state_without_resources():
    return {"other": "value"}


# Removed duplicate fixtures - they are already defined at the top of the file


@pytest.fixture
def patch_get_web_search_tool():
    with patch("src.graph.nodes.get_web_search_tool") as mock:
        from langchain_core.tools import tool

        @tool
        def web_search_tool(query: str) -> str:
            """A web search tool for testing."""
            return "search result"

        mock.return_value = web_search_tool
        yield mock


@pytest.fixture
def patch_crawl_tool():
    from langchain_core.tools import tool

    @tool
    def crawl_tool(url: str) -> str:
        """A crawl tool for testing."""
        return "crawl result"

    with patch("src.graph.nodes.crawl_tool", crawl_tool):
        yield


@pytest.fixture
def patch_get_retriever_tool():
    with patch("src.graph.nodes.get_retriever_tool") as mock:
        from langchain_core.tools import tool

        @tool
        def retriever_tool(query: str) -> str:
            """A retriever tool for testing."""
            return "retriever result"

        mock.return_value = retriever_tool
        yield mock


@pytest.fixture
def patch_setup_and_execute_agent_step():
    async def fake_setup_and_execute_agent_step(state, config, agent_type, tools):
        return "RESEARCHER_RESULT"

    with patch(
        "src.graph.nodes._setup_and_execute_agent_step",
        side_effect=fake_setup_and_execute_agent_step,
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_researcher_node_with_retriever_tool(
    mock_state_with_resources,
    mock_config,
    patch_config_from_runnable_config,
    patch_get_web_search_tool,
    patch_crawl_tool,
    patch_get_retriever_tool,
    patch_setup_and_execute_agent_step,
):
    # Simulate retriever_tool is returned
    retriever_tool = MagicMock(name="retriever_tool")
    patch_get_retriever_tool.return_value = retriever_tool

    result = await researcher_node(mock_state_with_resources, mock_config)

    # Should call get_web_search_tool with correct max_search_results and enable_smart_filtering
    patch_get_web_search_tool.assert_called_once_with(5, True)
    # Should call get_retriever_tool with resources
    patch_get_retriever_tool.assert_called_once_with(["resource1", "resource2"])
    # Should call _setup_and_execute_agent_step with retriever_tool first
    args, kwargs = patch_setup_and_execute_agent_step.call_args
    tools = args[3]
    assert tools[0] == retriever_tool
    assert patch_get_web_search_tool.return_value in tools
    assert result == "RESEARCHER_RESULT"


@pytest.mark.asyncio
async def test_researcher_node_without_retriever_tool(
    mock_state_with_resources,
    mock_config,
    patch_config_from_runnable_config,
    patch_get_web_search_tool,
    patch_crawl_tool,
    patch_setup_and_execute_agent_step,
):
    # Simulate retriever_tool is None
    with patch("src.graph.nodes.get_retriever_tool") as mock_get_retriever_tool:
        mock_get_retriever_tool.return_value = None

        result = await researcher_node(mock_state_with_resources, mock_config)

        patch_get_web_search_tool.assert_called_once_with(5, True)
        mock_get_retriever_tool.assert_called_once_with(["resource1", "resource2"])
        args, kwargs = patch_setup_and_execute_agent_step.call_args
        tools = args[3]
        # Should not include retriever_tool
        assert all(getattr(t, "name", None) != "retriever_tool" for t in tools)
        assert patch_get_web_search_tool.return_value in tools
        assert result == "RESEARCHER_RESULT"


@pytest.mark.asyncio
async def test_researcher_node_without_resources(
    mock_state_without_resources,
    mock_config,
    patch_config_from_runnable_config,
    patch_get_web_search_tool,
    patch_crawl_tool,
    patch_setup_and_execute_agent_step,
):
    with patch("src.graph.nodes.get_retriever_tool") as mock_get_retriever_tool:
        mock_get_retriever_tool.return_value = None

        result = await researcher_node(mock_state_without_resources, mock_config)

        patch_get_web_search_tool.assert_called_once_with(5, True)
        mock_get_retriever_tool.assert_called_once_with([])
        args, kwargs = patch_setup_and_execute_agent_step.call_args
        tools = args[3]
        assert patch_get_web_search_tool.return_value in tools
        assert result == "RESEARCHER_RESULT"
