import pytest
from unittest.mock import MagicMock

from src.mcp.enhanced_client import (
    MCPClients,
    MCPServerConfig,
    MCPConnectionInfo,
    MCPClientToolProxy,
    ConnectionState,
)


class TestConnectionState:
    def test_state_values(self):
        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"
        assert ConnectionState.ERROR == "error"


class TestMCPServerConfig:
    def test_sse_config(self):
        config = MCPServerConfig(transport="sse", url="http://localhost:8080")
        assert config.transport == "sse"
        assert config.url == "http://localhost:8080"
        assert config.reconnect_enabled is True
        assert config.timeout == 30

    def test_stdio_config(self):
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            args=["-m", "mcp_server"],
        )
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["-m", "mcp_server"]

    def test_custom_reconnect_settings(self):
        config = MCPServerConfig(
            transport="sse",
            url="http://localhost:8080",
            reconnect_enabled=False,
            reconnect_interval=10,
            max_reconnect_attempts=5,
        )
        assert config.reconnect_enabled is False
        assert config.reconnect_interval == 10
        assert config.max_reconnect_attempts == 5


class TestMCPConnectionInfo:
    def test_default_values(self):
        info = MCPConnectionInfo(server_id="test-server")
        assert info.server_id == "test-server"
        assert info.state == ConnectionState.DISCONNECTED
        assert info.connected_at is None
        assert info.tools_count == 0
        assert info.reconnect_attempts == 0

    def test_with_values(self):
        info = MCPConnectionInfo(
            server_id="test-server",
            state=ConnectionState.CONNECTED,
            tools_count=5,
            connected_at=12345.0,
        )
        assert info.state == ConnectionState.CONNECTED
        assert info.tools_count == 5
        assert info.connected_at == 12345.0


class TestMCPClientToolProxy:
    def test_proxy_creation(self):
        proxy = MCPClientToolProxy(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
            session=None,
            server_id="test-server",
            original_name="test_tool",
        )

        assert proxy.name == "test_tool"
        assert proxy.description == "A test tool"
        assert proxy.server_id == "test-server"

    def test_get_schema(self):
        proxy = MCPClientToolProxy(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            session=None,
            server_id="test-server",
            original_name="test_tool",
        )

        schema = proxy.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"


class TestMCPClients:
    def test_initialization(self):
        clients = MCPClients()
        assert clients.sessions == {}
        assert clients._tool_proxies == {}
        assert clients._connection_info == {}

    def test_get_tool_names_empty(self):
        clients = MCPClients()
        assert clients.get_tool_names() == []

    def test_get_connection_state_nonexistent(self):
        clients = MCPClients()
        state = clients.get_connection_state("nonexistent")
        assert state == ConnectionState.DISCONNECTED

    def test_is_connected_empty(self):
        clients = MCPClients()
        assert clients.is_connected() is True
        assert clients.is_connected("nonexistent") is False

    def test_get_stats_empty(self):
        clients = MCPClients()
        stats = clients.get_stats()
        assert stats["total_tools"] == 0
        assert stats["total_servers"] == 0
        assert stats["servers"] == {}

    def test_sanitize_tool_name(self):
        clients = MCPClients()

        assert clients._sanitize_tool_name("simple_name") == "simple_name"
        assert clients._sanitize_tool_name("name-with-dashes") == "name-with-dashes"
        assert clients._sanitize_tool_name("name with spaces") == "name_with_spaces"
        assert clients._sanitize_tool_name("name@#$%special") == "name_special"

        long_name = "a" * 100
        sanitized = clients._sanitize_tool_name(long_name)
        assert len(sanitized) == 64

    def test_extract_server_id(self):
        clients = MCPClients()

        server_id = clients._extract_server_id("http://localhost:8080")
        assert "localhost" in server_id

        server_id = clients._extract_server_id("https://api.example.com/mcp")
        assert "example_com" in server_id

    def test_set_callbacks(self):
        clients = MCPClients()

        tools_callback = MagicMock()
        state_callback = MagicMock()

        clients.set_on_tools_changed(tools_callback)
        clients.set_on_state_changed(state_callback)

        assert clients._on_tools_changed == tools_callback
        assert clients._on_state_changed == state_callback

    @pytest.mark.asyncio
    async def test_connect_sse_missing_url(self):
        clients = MCPClients()

        with pytest.raises(ValueError, match="Server URL is required"):
            await clients.connect_sse(server_url="")

    @pytest.mark.asyncio
    async def test_connect_stdio_missing_command(self):
        clients = MCPClients()

        with pytest.raises(ValueError, match="Server command is required"):
            await clients.connect_stdio(command="", args=[])

    @pytest.mark.asyncio
    async def test_disconnect_empty(self):
        clients = MCPClients()

        await clients.disconnect()
        assert clients.sessions == {}

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        clients = MCPClients()

        await clients.disconnect()
        assert clients._tool_proxies == {}

    def test_get_connection_info_empty(self):
        clients = MCPClients()

        info = clients.get_connection_info()
        assert info == {}

        info = clients.get_connection_info("nonexistent")
        assert info == {}
