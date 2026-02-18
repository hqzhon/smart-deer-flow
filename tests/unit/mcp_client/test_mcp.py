import pytest
from unittest.mock import Mock, AsyncMock

from src.mcp.client import MCPServerConfig, MCPClientToolProxy, MCPClients


class TestMCPServerConfig:
    def test_sse_config(self):
        config = MCPServerConfig(transport="sse", url="http://localhost:8080/mcp")
        assert config.transport == "sse"
        assert config.url == "http://localhost:8080/mcp"
        assert config.command is None

    def test_stdio_config(self):
        config = MCPServerConfig(
            transport="stdio", command="mcp-filesystem", args=["/workspace"]
        )
        assert config.transport == "stdio"
        assert config.command == "mcp-filesystem"
        assert config.args == ["/workspace"]

    def test_default_args(self):
        config = MCPServerConfig(transport="sse", url="http://localhost:8080")
        assert config.args == []


class TestMCPClientToolProxy:
    def test_proxy_creation(self):
        proxy = MCPClientToolProxy(
            name="mcp_server_test_tool",
            description="Test tool",
            parameters={"type": "object"},
            session=None,
            server_id="server",
            original_name="test_tool",
        )
        assert proxy.name == "mcp_server_test_tool"
        assert proxy.description == "Test tool"
        assert proxy.original_name == "test_tool"

    def test_get_schema(self):
        proxy = MCPClientToolProxy(
            name="test_tool",
            description="Test",
            parameters={"type": "object", "properties": {}},
            session=None,
            server_id="server",
            original_name="test",
        )
        schema = proxy.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_no_session(self):
        proxy = MCPClientToolProxy(
            name="test",
            description="Test",
            parameters={},
            session=None,
            server_id="server",
            original_name="test",
        )
        result = await proxy.execute()
        assert result.error is not None
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_session(self):
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.content = [Mock(text="Tool output")]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        proxy = MCPClientToolProxy(
            name="test",
            description="Test",
            parameters={},
            session=mock_session,
            server_id="server",
            original_name="test_tool",
        )
        result = await proxy.execute(arg1="value1")
        assert result.output == "Tool output"
        mock_session.call_tool.assert_called_once_with("test_tool", {"arg1": "value1"})

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("Connection error"))

        proxy = MCPClientToolProxy(
            name="test",
            description="Test",
            parameters={},
            session=mock_session,
            server_id="server",
            original_name="test",
        )
        result = await proxy.execute()
        assert result.error is not None
        assert "Error executing tool" in result.error


class TestMCPClients:
    def test_clients_creation(self):
        clients = MCPClients()
        assert len(clients.sessions) == 0
        assert len(clients.exit_stacks) == 0
        assert len(clients._tool_proxies) == 0

    def test_sanitize_tool_name(self):
        clients = MCPClients()
        assert clients._sanitize_tool_name("valid_name") == "valid_name"
        assert clients._sanitize_tool_name("invalid name!") == "invalid_name"
        assert clients._sanitize_tool_name("a" * 100) == "a" * 64

    def test_get_tool_names(self):
        clients = MCPClients()
        clients._tool_proxies["tool1"] = Mock()
        clients._tool_proxies["tool2"] = Mock()
        names = clients.get_tool_names()
        assert "tool1" in names
        assert "tool2" in names

    def test_get_tool(self):
        clients = MCPClients()
        proxy = MCPClientToolProxy(
            name="test",
            description="Test",
            parameters={},
            session=None,
            server_id="server",
            original_name="test",
        )
        clients._tool_proxies["test"] = proxy
        retrieved = clients.get_tool("test")
        assert retrieved is proxy

    def test_get_tool_not_found(self):
        clients = MCPClients()
        assert clients.get_tool("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        clients = MCPClients()
        result = await clients.execute(name="nonexistent", tool_input={})
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        clients = MCPClients()
        proxy = MCPClientToolProxy(
            name="test",
            description="Test",
            parameters={},
            session=None,
            server_id="server",
            original_name="test",
        )
        clients._tool_proxies["test"] = proxy
        result = await clients.execute(name="test", tool_input={})
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_connect_sse_no_url(self):
        clients = MCPClients()
        with pytest.raises(ValueError, match="URL is required"):
            await clients.connect_sse("")

    @pytest.mark.asyncio
    async def test_connect_stdio_no_command(self):
        clients = MCPClients()
        with pytest.raises(ValueError, match="command is required"):
            await clients.connect_stdio("", [])

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self):
        clients = MCPClients()
        await clients.disconnect("nonexistent")

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        clients = MCPClients()
        await clients.disconnect()
        assert len(clients._tool_proxies) == 0
