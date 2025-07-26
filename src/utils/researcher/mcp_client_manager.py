"""MCP client management module - Unified management of MCP server connections and tools"""

import logging
from typing import Dict, List, Any, Tuple
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MCPClientManager:
    """MCP client manager - Unified management of MCP server connections and tool configurations"""

    def __init__(self, configurable: Any):
        """Initialize MCP client manager

        Args:
            configurable: Configuration object
        """
        self.configurable = configurable
        self._mcp_servers = None
        self._enabled_tools = None

        # Initialize unified configuration manager
        from .isolation_config_manager import IsolationConfigManager

        config_manager = IsolationConfigManager(configurable)
        self.unified_config = config_manager.get_unified_config()

        logger.info("MCPClientManager initialized with unified config")

    def is_mcp_enabled(self) -> bool:
        """Check if MCP is enabled

        Returns:
            Whether MCP is enabled
        """
        try:
            return self.unified_config.mcp_enabled and self.unified_config.mcp_servers
        except AttributeError:
            logger.debug("MCP configuration not found")
            return False

    def _extract_mcp_servers_config(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Extract MCP server configuration

        Returns:
            (Server configuration dictionary, enabled tools mapping)
        """
        if not self.is_mcp_enabled():
            return {}, {}

        mcp_servers = {}
        enabled_tools = {}

        try:
            for server_name, server_config in enumerate(
                self.unified_config.mcp_servers
            ):
                # Check if server is enabled for researcher
                if server_config.get(
                    "enabled_tools"
                ) and "researcher" in server_config.get("add_to_agents", []):
                    server_key = f"server_{server_name}"

                    # Extract server connection configuration
                    mcp_servers[server_key] = {
                        k: v
                        for k, v in server_config.items()
                        if k in ("transport", "command", "args", "url", "env")
                    }

                    # Map enabled tools to server
                    for tool_name in server_config["enabled_tools"]:
                        enabled_tools[tool_name] = server_key

            logger.info(
                f"Extracted {len(mcp_servers)} MCP servers with {len(enabled_tools)} tools"
            )

        except Exception as e:
            logger.error(f"Failed to extract MCP server config: {e}")
            return {}, {}

        return mcp_servers, enabled_tools

    def get_mcp_config(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Get MCP configuration

        Returns:
            (Server configuration dictionary, enabled tools mapping)
        """
        if self._mcp_servers is None or self._enabled_tools is None:
            self._mcp_servers, self._enabled_tools = self._extract_mcp_servers_config()

        return self._mcp_servers.copy(), self._enabled_tools.copy()

    @asynccontextmanager
    async def get_mcp_client(self):
        """Get MCP client context manager

        Yields:
            MCP client instance and enabled tools mapping
        """
        mcp_servers, enabled_tools = self.get_mcp_config()

        if not mcp_servers:
            logger.debug("No MCP servers configured, yielding None")
            yield None, {}
            return

        try:
            # Dynamically import MCP client
            from src.mcp.multi_server_client import MultiServerMCPClient

            logger.info(f"Creating MCP client with {len(mcp_servers)} servers")

            async with MultiServerMCPClient(mcp_servers) as client:
                logger.info("MCP client created successfully")
                yield client, enabled_tools

        except ImportError as e:
            logger.error(f"Failed to import MCP client: {e}")
            yield None, {}
        except Exception as e:
            logger.error(f"Failed to create MCP client: {e}")
            yield None, {}

    def get_mcp_tools_with_client(
        self, client: Any, enabled_tools: Dict[str, str]
    ) -> List[Any]:
        """Get tool list from MCP client

        Args:
            client: MCP client instance
            enabled_tools: Enabled tools mapping

        Returns:
            MCP tools list
        """
        if not client:
            return []

        try:
            mcp_tools = []
            for tool in client.get_tools():
                if tool.name in enabled_tools:
                    # Add server information to tool
                    server_name = enabled_tools[tool.name]
                    tool.description = (
                        f"Powered by '{server_name}'.\n{tool.description}"
                    )
                    mcp_tools.append(tool)

            logger.info(f"Retrieved {len(mcp_tools)} MCP tools")
            return mcp_tools

        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            return []

    async def cleanup_clients(self):
        """Clean up MCP client resources

        Note: Actual cleanup is handled automatically by context manager
        """
        logger.debug("MCP client cleanup requested (handled by context manager)")

    def get_mcp_summary(self) -> Dict[str, Any]:
        """Get MCP configuration summary

        Returns:
            MCP configuration summary
        """
        mcp_servers, enabled_tools = self.get_mcp_config()

        return {
            "mcp_enabled": self.is_mcp_enabled(),
            "server_count": len(mcp_servers),
            "enabled_tool_count": len(enabled_tools),
            "server_keys": list(mcp_servers.keys()),
            "enabled_tool_names": list(enabled_tools.keys()),
        }
