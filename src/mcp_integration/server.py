import argparse
import asyncio
import atexit
import json
import logging
import sys
from inspect import Parameter, Signature
from typing import Any, Callable, Optional

from pydantic import BaseModel

from src.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class MCPServerSettings(BaseModel):
    """MCP Server configuration settings."""

    name: str = "deerflow"
    transport: str = "stdio"
    auto_cleanup: bool = True


class MCPServer:
    """MCP Server implementation for exposing DeerFlow tools as MCP services.

    This server allows other applications to discover and use DeerFlow tools
    through the Model Context Protocol (MCP).
    """

    def __init__(
        self, name: str = "deerflow", settings: Optional[MCPServerSettings] = None
    ):
        self.name = name
        self.settings = settings or MCPServerSettings(name=name)
        self._tools: dict[str, BaseTool] = {}
        self._tool_handlers: dict[str, Callable] = {}
        self._mcp_server: Optional[Any] = None
        self._initialized = False

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool with the MCP server.

        Args:
            tool: The tool instance to register
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

        if self._initialized and self._mcp_server:
            self._register_tool_with_mcp(tool)

    def register_tools(self, tools: list[BaseTool]) -> None:
        """Register multiple tools with the MCP server.

        Args:
            tools: List of tool instances to register
        """
        for tool in tools:
            self.register_tool(tool)

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the MCP server.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            if tool_name in self._tool_handlers:
                del self._tool_handlers[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False

    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def _build_docstring(self, tool: BaseTool) -> str:
        """Build a formatted docstring from tool metadata."""
        description = tool.description or f"Execute {tool.name} tool"
        params = tool.parameters or {}
        properties = params.get("properties", {})
        required = params.get("required", [])

        docstring = description
        if properties:
            docstring += "\n\nParameters:\n"
            for param_name, param_details in properties.items():
                required_str = "(required)" if param_name in required else "(optional)"
                param_type = param_details.get("type", "any")
                param_desc = param_details.get("description", "")
                docstring += (
                    f"    {param_name} ({param_type}) {required_str}: {param_desc}\n"
                )

        return docstring

    def _build_signature(self, tool: BaseTool) -> Signature:
        """Build a function signature from tool parameters."""
        params = tool.parameters or {}
        properties = params.get("properties", {})
        required = params.get("required", [])

        parameters = []
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        for param_name, param_details in properties.items():
            param_type = param_details.get("type", "")
            default = Parameter.empty if param_name in required else None
            annotation = type_mapping.get(param_type, Any)

            param = Parameter(
                name=param_name,
                kind=Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
            parameters.append(param)

        return Signature(parameters=parameters)

    def _create_tool_handler(self, tool: BaseTool) -> Callable:
        """Create an async handler function for a tool."""

        async def handler(**kwargs) -> str:
            logger.info(f"Executing tool '{tool.name}' with args: {kwargs}")
            try:
                result = await tool.async_execute(**kwargs)

                if hasattr(result, "model_dump"):
                    return json.dumps(result.model_dump())
                elif isinstance(result, dict):
                    return json.dumps(result)
                elif hasattr(result, "message"):
                    return result.message
                return str(result)
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg, "success": False})

        handler.__name__ = tool.name
        handler.__doc__ = self._build_docstring(tool)
        handler.__signature__ = self._build_signature(tool)

        return handler

    def _register_tool_with_mcp(self, tool: BaseTool) -> None:
        """Register a tool with the underlying MCP server."""
        if not self._mcp_server:
            return

        handler = self._create_tool_handler(tool)
        self._tool_handlers[tool.name] = handler

        try:
            self._mcp_server.tool()(handler)
            logger.info(f"Registered tool '{tool.name}' with MCP server")
        except Exception as e:
            logger.error(f"Failed to register tool '{tool.name}': {e}")

    def _initialize_mcp_server(self) -> None:
        """Initialize the underlying FastMCP server."""
        try:
            from mcp.server.fastmcp import FastMCP

            self._mcp_server = FastMCP(self.name)
            self._initialized = True
            logger.info(f"MCP server '{self.name}' initialized")

            for tool in self._tools.values():
                self._register_tool_with_mcp(tool)

        except ImportError:
            raise ImportError(
                "FastMCP is not installed. Install it with: pip install mcp"
            )

    async def cleanup(self) -> None:
        """Clean up server resources."""
        logger.info("Cleaning up MCP server resources")

        cleanup_tasks = []
        for tool in self._tools.values():
            if hasattr(tool, "cleanup") and callable(getattr(tool, "cleanup")):
                cleanup_tasks.append(tool.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._tools.clear()
        self._tool_handlers.clear()

    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server.

        Args:
            transport: Transport type ('stdio' or 'sse')
        """
        self._initialize_mcp_server()

        if self.settings.auto_cleanup:
            atexit.register(lambda: asyncio.run(self.cleanup()))

        logger.info(f"Starting MCP server '{self.name}' ({transport} mode)")
        self._mcp_server.run(transport=transport)

    async def run_async(self, transport: str = "stdio") -> None:
        """Run the MCP server asynchronously.

        Args:
            transport: Transport type ('stdio' or 'sse')
        """
        self._initialize_mcp_server()

        if self.settings.auto_cleanup:
            atexit.register(lambda: asyncio.run(self.cleanup()))

        logger.info(f"Starting MCP server '{self.name}' ({transport} mode)")
        await self._mcp_server.run_async(transport=transport)


def create_default_server() -> MCPServer:
    """Create a default MCP server with standard tools."""
    from src.tools.browser.browser_use import BrowserUseTool
    from src.tools.python_repl_tool import PythonREPLTool

    server = MCPServer(name="deerflow")

    server.register_tool(PythonREPLTool())
    server.register_tool(BrowserUseTool())

    return server


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeerFlow MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Communication transport (default: stdio)",
    )
    parser.add_argument(
        "--name",
        default="deerflow",
        help="Server name (default: deerflow)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)]
    )

    args = parse_args()
    server = create_default_server()
    server.run(transport=args.transport)
