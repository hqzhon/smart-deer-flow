import pytest
from src.tools.tool_result import ToolResult, CLIResult, ToolFailure
from src.tools.tool_collection import ToolCollection
from src.tools.base_tool import BaseTool, ToolOutput


class MockTool(BaseTool):
    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input string"},
            },
            "required": ["input"],
        }

    def execute(self, **kwargs) -> ToolOutput:
        input_val = kwargs.get("input", "")
        if input_val == "error":
            raise ValueError("Test error")
        return ToolOutput(success=True, message="Executed", data=f"Result: {input_val}")


class AsyncMockTool(BaseTool):
    @property
    def name(self) -> str:
        return "async_mock_tool"

    @property
    def description(self) -> str:
        return "An async mock tool"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **kwargs):
        return ToolOutput(success=True, message="Async executed")


class TestToolResult:
    def test_tool_result_creation(self):
        result = ToolResult(output="test output")
        assert result.output == "test output"
        assert result.error is None
        assert result.base64_image is None
        assert result.system is None

    def test_tool_result_with_error(self):
        result = ToolResult(error="test error")
        assert result.error == "test error"
        assert result.output is None

    def test_tool_result_bool(self):
        assert bool(ToolResult(output="test")) is True
        assert bool(ToolResult(error="error")) is True
        assert bool(ToolResult()) is False

    def test_tool_result_str(self):
        assert str(ToolResult(output="test")) == "test"
        assert str(ToolResult(error="error")) == "Error: error"

    def test_tool_result_add(self):
        result1 = ToolResult(output="output1")
        result2 = ToolResult(output="output2")
        combined = result1 + result2
        assert combined.output == "output1output2"

    def test_tool_result_add_with_error(self):
        result1 = ToolResult(output="output")
        result2 = ToolResult(error="error")
        combined = result1 + result2
        assert combined.output == "output"
        assert combined.error == "error"

    def test_tool_result_replace(self):
        result = ToolResult(output="original")
        new_result = result.replace(output="replaced")
        assert new_result.output == "replaced"
        assert result.output == "original"

    def test_cli_result(self):
        result = CLIResult(output="cli output")
        assert isinstance(result, ToolResult)
        assert result.output == "cli output"

    def test_tool_failure(self):
        result = ToolFailure(error="failure")
        assert isinstance(result, ToolResult)
        assert result.error == "failure"


class TestToolCollection:
    def test_empty_collection(self):
        collection = ToolCollection()
        assert len(collection) == 0
        assert len(collection.tool_map) == 0

    def test_collection_with_tools(self):
        tool1 = MockTool()
        tool2 = AsyncMockTool()
        collection = ToolCollection(tool1, tool2)
        assert len(collection) == 2
        assert "mock_tool" in collection.tool_map
        assert "async_mock_tool" in collection.tool_map

    def test_to_params(self):
        tool = MockTool()
        collection = ToolCollection(tool)
        params = collection.to_params()
        assert len(params) == 1
        assert params[0]["type"] == "function"
        assert params[0]["function"]["name"] == "mock_tool"

    def test_get_tool(self):
        tool = MockTool()
        collection = ToolCollection(tool)
        retrieved = collection.get_tool("mock_tool")
        assert retrieved is not None
        assert retrieved.name == "mock_tool"

    def test_get_tool_not_found(self):
        collection = ToolCollection()
        assert collection.get_tool("nonexistent") is None

    def test_add_tool(self):
        tool = MockTool()
        collection = ToolCollection()
        result = collection.add_tool(tool)
        assert result is collection
        assert len(collection) == 1
        assert "mock_tool" in collection.tool_map

    def test_add_duplicate_tool(self):
        tool1 = MockTool()
        tool2 = MockTool()
        collection = ToolCollection(tool1)
        collection.add_tool(tool2)
        assert len(collection) == 1

    def test_add_tools(self):
        tool1 = MockTool()
        tool2 = AsyncMockTool()
        collection = ToolCollection()
        result = collection.add_tools(tool1, tool2)
        assert result is collection
        assert len(collection) == 2

    def test_remove_tool(self):
        tool = MockTool()
        collection = ToolCollection(tool)
        collection.remove_tool("mock_tool")
        assert len(collection) == 0
        assert "mock_tool" not in collection.tool_map

    def test_remove_nonexistent_tool(self):
        collection = ToolCollection()
        collection.remove_tool("nonexistent")
        assert len(collection) == 0

    def test_has_tool(self):
        tool = MockTool()
        collection = ToolCollection(tool)
        assert collection.has_tool("mock_tool") is True
        assert collection.has_tool("nonexistent") is False

    def test_get_tool_names(self):
        tool1 = MockTool()
        tool2 = AsyncMockTool()
        collection = ToolCollection(tool1, tool2)
        names = collection.get_tool_names()
        assert "mock_tool" in names
        assert "async_mock_tool" in names

    def test_iter(self):
        tool1 = MockTool()
        tool2 = AsyncMockTool()
        collection = ToolCollection(tool1, tool2)
        tools = list(collection)
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_execute(self):
        tool = MockTool()
        collection = ToolCollection(tool)
        result = await collection.execute(
            name="mock_tool", tool_input={"input": "test"}
        )
        assert result.output is not None or result.error is None

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        collection = ToolCollection()
        result = await collection.execute(name="nonexistent", tool_input={})
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_all(self):
        tool1 = MockTool()
        tool2 = AsyncMockTool()
        collection = ToolCollection(tool1, tool2)
        results = await collection.execute_all()
        assert len(results) == 2
