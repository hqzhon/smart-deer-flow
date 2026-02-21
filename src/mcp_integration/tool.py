import logging
from typing import Any

from pydantic import Field

from src.tools.base_tool import BaseTool
from src.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)


class MCPClientTool(BaseTool):
    _name: str = Field(default="mcp_tool")
    _description: str = Field(default="MCP tool proxy")
    _parameters: dict[str, Any] | None = Field(default=None)
    session: Any = Field(default=None)
    server_id: str = Field(default="")
    original_name: str = Field(default="")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters or {}

    def execute(self, **kwargs) -> ToolResult:
        import asyncio

        return asyncio.run(self._async_execute(**kwargs))

    async def _async_execute(self, **kwargs) -> ToolResult:
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
