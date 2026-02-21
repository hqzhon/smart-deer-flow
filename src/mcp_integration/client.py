import logging
import re
from contextlib import AsyncExitStack
from typing import Any

from pydantic import BaseModel, Field

from src.tools.tool_collection import ToolCollection
from src.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    transport: str = Field(..., description="Transport type: sse or stdio")
    url: str | None = Field(default=None, description="SSE URL")
    command: str | None = Field(default=None, description="stdio command")
    args: list[str] = Field(default_factory=list, description="Command arguments")


class MCPClientToolProxy:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        session: Any,
        server_id: str,
        original_name: str,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.session = session
        self.server_id = server_id
        self.original_name = original_name

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

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


class MCPClients(ToolCollection):
    sessions: dict[str, Any] = {}
    exit_stacks: dict[str, AsyncExitStack] = {}

    def __init__(self):
        super().__init__()
        self._tool_proxies: dict[str, MCPClientToolProxy] = {}

    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        if not server_url:
            raise ValueError("Server URL is required.")

        server_id = server_id or server_url

        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session

        await self._initialize_and_list_tools(server_id)

    async def connect_stdio(
        self, command: str, args: list[str], server_id: str = ""
    ) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not command:
            raise ValueError("Server command is required.")

        server_id = server_id or command

        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        self.sessions[server_id] = session

        await self._initialize_and_list_tools(server_id)

    async def _initialize_and_list_tools(self, server_id: str) -> None:
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Session not initialized for server {server_id}")

        await session.initialize()
        response = await session.list_tools()

        for tool in response.tools:
            original_name = tool.name
            tool_name = f"mcp_{server_id}_{original_name}"
            tool_name = self._sanitize_tool_name(tool_name)

            proxy = MCPClientToolProxy(
                name=tool_name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
                original_name=original_name,
            )
            self._tool_proxies[tool_name] = proxy

        logger.info(
            f"Connected to server {server_id} with tools: {[tool.name for tool in response.tools]}"
        )

    def _sanitize_tool_name(self, name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_")

        if len(sanitized) > 64:
            sanitized = sanitized[:64]

        return sanitized

    async def disconnect(self, server_id: str = "") -> None:
        if server_id:
            if server_id in self.sessions:
                try:
                    exit_stack = self.exit_stacks.get(server_id)

                    if exit_stack:
                        try:
                            await exit_stack.aclose()
                        except RuntimeError:
                            pass

                    self.sessions.pop(server_id, None)
                    self.exit_stacks.pop(server_id, None)

                    self._tool_proxies = {
                        k: v
                        for k, v in self._tool_proxies.items()
                        if v.server_id != server_id
                    }
                    logger.info(f"Disconnected from MCP server {server_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting from server {server_id}: {e}")
        else:
            for sid in sorted(list(self.sessions.keys())):
                await self.disconnect(sid)
            self._tool_proxies = {}
            logger.info("Disconnected from all MCP servers")

    async def execute(
        self, *, name: str, tool_input: dict[str, Any] | None = None
    ) -> ToolResult:
        proxy = self._tool_proxies.get(name)
        if not proxy:
            return ToolResult(error=f"Tool {name} not found")
        return await proxy.execute(**(tool_input or {}))

    def get_tool_names(self) -> list[str]:
        return list(self._tool_proxies.keys())

    def get_tool(self, name: str) -> MCPClientToolProxy | None:
        return self._tool_proxies.get(name)
