import asyncio
import logging
import re
import time
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.mcp_integration.client import (
    MCPClients as BaseMCPClients,
    MCPClientToolProxy,
)
from src.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """MCP connection state enumeration"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MCPServerConfig(BaseModel):
    """MCP server configuration with enhanced features"""

    transport: str = Field(..., description="Transport type: sse or stdio")
    url: Optional[str] = Field(default=None, description="SSE URL")
    command: Optional[str] = Field(default=None, description="stdio command")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    reconnect_enabled: bool = Field(default=True, description="Enable auto reconnect")
    reconnect_interval: int = Field(
        default=5, description="Reconnect interval in seconds"
    )
    max_reconnect_attempts: int = Field(default=3, description="Max reconnect attempts")
    timeout: int = Field(default=30, description="Connection timeout in seconds")


class MCPConnectionInfo(BaseModel):
    """MCP connection information"""

    server_id: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    connected_at: Optional[float] = None
    last_activity: Optional[float] = None
    reconnect_attempts: int = 0
    error_message: Optional[str] = None
    tools_count: int = 0


class EnhancedMCPClientToolProxy(MCPClientToolProxy):
    """Enhanced proxy for MCP tools with connection state tracking"""

    async def execute(self, **kwargs) -> ToolResult:
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            result = await self.session.call_tool(self.original_name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if hasattr(item, "text")
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")


class MCPClients(BaseMCPClients):
    """Enhanced MCP client with connection management and auto-reconnect"""

    def __init__(self):
        super().__init__()
        self._connection_info: dict[str, MCPConnectionInfo] = {}
        self._configs: dict[str, MCPServerConfig] = {}
        self._on_tools_changed: Optional[Callable[[list[str]], None]] = None
        self._on_state_changed: Optional[Callable[[str, ConnectionState], None]] = None
        self._reconnect_tasks: dict[str, asyncio.Task] = {}

    def set_on_tools_changed(self, callback: Callable[[list[str]], None]) -> None:
        self._on_tools_changed = callback

    def set_on_state_changed(
        self, callback: Callable[[str, ConnectionState], None]
    ) -> None:
        self._on_state_changed = callback

    def _update_state(
        self, server_id: str, state: ConnectionState, error: Optional[str] = None
    ) -> None:
        if server_id in self._connection_info:
            self._connection_info[server_id].state = state
            self._connection_info[server_id].error_message = error
            if state == ConnectionState.CONNECTED:
                self._connection_info[server_id].connected_at = time.time()
                self._connection_info[server_id].reconnect_attempts = 0

        if self._on_state_changed:
            self._on_state_changed(server_id, state)

    async def connect_sse(
        self,
        server_url: str,
        server_id: str = "",
        reconnect_enabled: bool = True,
        timeout: int = 30,
    ) -> None:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        if not server_url:
            raise ValueError("Server URL is required.")

        server_id = server_id or self._extract_server_id(server_url)

        if server_id in self.sessions:
            await self.disconnect(server_id)

        self._connection_info[server_id] = MCPConnectionInfo(server_id=server_id)
        self._configs[server_id] = MCPServerConfig(
            transport="sse",
            url=server_url,
            reconnect_enabled=reconnect_enabled,
            timeout=timeout,
        )

        self._update_state(server_id, ConnectionState.CONNECTING)

        try:
            from contextlib import AsyncExitStack

            exit_stack = AsyncExitStack()
            self.exit_stacks[server_id] = exit_stack

            streams_context = sse_client(url=server_url)
            streams = await asyncio.wait_for(
                exit_stack.enter_async_context(streams_context),
                timeout=timeout,
            )
            session = await exit_stack.enter_async_context(ClientSession(*streams))
            self.sessions[server_id] = session

            await self._initialize_and_list_tools(server_id)
            self._update_state(server_id, ConnectionState.CONNECTED)

        except asyncio.TimeoutError:
            self._update_state(server_id, ConnectionState.ERROR, "Connection timeout")
            raise TimeoutError(
                f"Connection to {server_url} timed out after {timeout} seconds"
            )
        except Exception as e:
            self._update_state(server_id, ConnectionState.ERROR, str(e))
            raise

    async def connect_stdio(
        self,
        command: str,
        args: list[str],
        server_id: str = "",
        reconnect_enabled: bool = True,
        timeout: int = 30,
    ) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not command:
            raise ValueError("Server command is required.")

        server_id = server_id or command

        if server_id in self.sessions:
            await self.disconnect(server_id)

        self._connection_info[server_id] = MCPConnectionInfo(server_id=server_id)
        self._configs[server_id] = MCPServerConfig(
            transport="stdio",
            command=command,
            args=args,
            reconnect_enabled=reconnect_enabled,
            timeout=timeout,
        )

        self._update_state(server_id, ConnectionState.CONNECTING)

        try:
            from contextlib import AsyncExitStack

            exit_stack = AsyncExitStack()
            self.exit_stacks[server_id] = exit_stack

            server_params = StdioServerParameters(command=command, args=args)
            stdio_transport = await asyncio.wait_for(
                exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=timeout,
            )
            read, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            self.sessions[server_id] = session

            await self._initialize_and_list_tools(server_id)
            self._update_state(server_id, ConnectionState.CONNECTED)

        except asyncio.TimeoutError:
            self._update_state(server_id, ConnectionState.ERROR, "Connection timeout")
            raise TimeoutError(
                f"Connection to {command} timed out after {timeout} seconds"
            )
        except Exception as e:
            self._update_state(server_id, ConnectionState.ERROR, str(e))
            raise

    async def _initialize_and_list_tools(self, server_id: str) -> None:
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Session not initialized for server {server_id}")

        await session.initialize()
        response = await session.list_tools()

        server_tools = []
        for tool in response.tools:
            original_name = tool.name
            tool_name = f"mcp_{server_id}_{original_name}"
            tool_name = self._sanitize_tool_name(tool_name)

            proxy = EnhancedMCPClientToolProxy(
                name=tool_name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
                original_name=original_name,
            )
            self._tool_proxies[tool_name] = proxy
            server_tools.append(tool_name)

        if server_id in self._connection_info:
            self._connection_info[server_id].tools_count = len(server_tools)
            self._connection_info[server_id].last_activity = time.time()

        logger.info(
            f"Connected to server {server_id} with {len(server_tools)} tools: {server_tools}"
        )

        if self._on_tools_changed:
            self._on_tools_changed(list(self._tool_proxies.keys()))

    def _extract_server_id(self, url: str) -> str:
        match = re.search(r"://([^/:]+)", url)
        if match:
            return match.group(1).replace(".", "_").replace("-", "_")
        return f"server_{int(time.time())}"

    async def reconnect(self, server_id: str) -> bool:
        if server_id not in self._configs:
            logger.warning(f"No config found for server {server_id}")
            return False

        config = self._configs[server_id]
        self._update_state(server_id, ConnectionState.RECONNECTING)

        try:
            if config.transport == "sse":
                await self.connect_sse(
                    server_url=config.url,
                    server_id=server_id,
                    reconnect_enabled=config.reconnect_enabled,
                    timeout=config.timeout,
                )
            else:
                await self.connect_stdio(
                    command=config.command,
                    args=config.args,
                    server_id=server_id,
                    reconnect_enabled=config.reconnect_enabled,
                    timeout=config.timeout,
                )
            return True
        except Exception as e:
            logger.error(f"Reconnect failed for {server_id}: {e}")
            self._update_state(server_id, ConnectionState.ERROR, str(e))
            return False

    async def _auto_reconnect_loop(self, server_id: str) -> None:
        if server_id not in self._configs:
            return

        config = self._configs[server_id]
        max_attempts = config.max_reconnect_attempts

        for attempt in range(max_attempts):
            if server_id not in self._connection_info:
                break

            info = self._connection_info[server_id]
            if info.state == ConnectionState.CONNECTED:
                break

            info.reconnect_attempts = attempt + 1
            logger.info(
                f"Auto-reconnect attempt {attempt + 1}/{max_attempts} for {server_id}"
            )

            success = await self.reconnect(server_id)
            if success:
                logger.info(f"Auto-reconnect successful for {server_id}")
                break

            await asyncio.sleep(config.reconnect_interval)

    async def disconnect(self, server_id: str = "") -> None:
        if server_id:
            if server_id in self._reconnect_tasks:
                self._reconnect_tasks[server_id].cancel()
                del self._reconnect_tasks[server_id]

            if server_id in self._connection_info:
                self._connection_info[server_id].state = ConnectionState.DISCONNECTED

        await super().disconnect(server_id)

    async def refresh_tools(self, server_id: str = "") -> list[str]:
        if server_id:
            if server_id not in self.sessions:
                raise RuntimeError(f"Not connected to server {server_id}")
            await self._initialize_and_list_tools(server_id)
        else:
            for sid in self.sessions:
                await self._initialize_and_list_tools(sid)

        return list(self._tool_proxies.keys())

    async def execute(
        self, *, name: str, tool_input: dict[str, Any] | None = None
    ) -> ToolResult:
        proxy = self._tool_proxies.get(name)
        if not proxy:
            return ToolResult(error=f"Tool {name} not found")

        server_id = proxy.server_id
        if server_id in self._connection_info:
            self._connection_info[server_id].last_activity = time.time()

        result = await proxy.execute(**(tool_input or {}))

        if result.error and server_id in self._configs:
            config = self._configs[server_id]
            if config.reconnect_enabled and server_id in self._connection_info:
                info = self._connection_info[server_id]
                if info.state != ConnectionState.CONNECTED:
                    if server_id not in self._reconnect_tasks:
                        self._reconnect_tasks[server_id] = asyncio.create_task(
                            self._auto_reconnect_loop(server_id)
                        )

        return result

    def get_connection_info(self, server_id: str = "") -> dict[str, MCPConnectionInfo]:
        if server_id:
            if server_id in self._connection_info:
                return {server_id: self._connection_info[server_id]}
            return {}
        return self._connection_info.copy()

    def get_connection_state(self, server_id: str) -> ConnectionState:
        if server_id in self._connection_info:
            return self._connection_info[server_id].state
        return ConnectionState.DISCONNECTED

    def is_connected(self, server_id: str = "") -> bool:
        if server_id:
            return self.get_connection_state(server_id) == ConnectionState.CONNECTED
        return all(
            info.state == ConnectionState.CONNECTED
            for info in self._connection_info.values()
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "servers": {
                server_id: {
                    "state": info.state.value,
                    "tools_count": info.tools_count,
                    "connected_at": info.connected_at,
                    "last_activity": info.last_activity,
                    "reconnect_attempts": info.reconnect_attempts,
                    "error": info.error_message,
                }
                for server_id, info in self._connection_info.items()
            },
            "total_tools": len(self._tool_proxies),
            "total_servers": len(self.sessions),
        }
