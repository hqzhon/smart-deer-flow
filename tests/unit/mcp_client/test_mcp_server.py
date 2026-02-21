import pytest
from unittest.mock import MagicMock, AsyncMock

from src.mcp_integration.server import MCPServer, MCPServerSettings


class TestMCPServerSettings:
    def test_default_values(self):
        settings = MCPServerSettings()
        assert settings.name == "deerflow"
        assert settings.transport == "stdio"
        assert settings.auto_cleanup is True

    def test_custom_values(self):
        settings = MCPServerSettings(
            name="custom-server",
            transport="sse",
            auto_cleanup=False,
        )
        assert settings.name == "custom-server"
        assert settings.transport == "sse"
        assert settings.auto_cleanup is False


class TestMCPServer:
    def test_initialization(self):
        server = MCPServer()
        assert server.name == "deerflow"
        assert server._tools == {}
        assert server._tool_handlers == {}
        assert server._initialized is False

    def test_register_tool(self):
        server = MCPServer()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {"type": "object", "properties": {}}

        server.register_tool(mock_tool)

        assert "test_tool" in server._tools
        assert server._tools["test_tool"] == mock_tool

    def test_register_multiple_tools(self):
        server = MCPServer()

        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.parameters = {}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.parameters = {}

        server.register_tools([tool1, tool2])

        assert "tool1" in server._tools
        assert "tool2" in server._tools

    def test_unregister_tool(self):
        server = MCPServer()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {}

        server.register_tool(mock_tool)
        result = server.unregister_tool("test_tool")

        assert result is True
        assert "test_tool" not in server._tools

    def test_unregister_nonexistent_tool(self):
        server = MCPServer()
        result = server.unregister_tool("nonexistent")
        assert result is False

    def test_get_registered_tools(self):
        server = MCPServer()

        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.parameters = {}

        server.register_tool(tool1)

        tools = server.get_registered_tools()
        assert "tool1" in tools

    def test_build_docstring(self):
        server = MCPServer()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First param"},
                "param2": {"type": "integer", "description": "Second param"},
            },
            "required": ["param1"],
        }

        docstring = server._build_docstring(mock_tool)

        assert "A test tool" in docstring
        assert "param1" in docstring
        assert "param2" in docstring
        assert "(required)" in docstring
        assert "(optional)" in docstring

    def test_build_signature(self):
        server = MCPServer()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"},
            },
            "required": ["param1"],
        }

        signature = server._build_signature(mock_tool)

        param_names = [p.name for p in signature.parameters.values()]
        assert "param1" in param_names
        assert "param2" in param_names

    def test_create_tool_handler(self):
        server = MCPServer()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {}
        mock_tool.async_execute = AsyncMock(return_value=MagicMock(output="Success"))

        handler = server._create_tool_handler(mock_tool)

        assert handler.__name__ == "test_tool"
        assert "A test tool" in handler.__doc__

    @pytest.mark.asyncio
    async def test_cleanup(self):
        server = MCPServer()

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {}
        mock_tool.cleanup = AsyncMock()

        server.register_tool(mock_tool)
        await server.cleanup()

        assert server._tools == {}
        assert server._tool_handlers == {}
