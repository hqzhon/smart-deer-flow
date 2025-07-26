"""Tool management module - Unified management of research tool creation and configuration"""

import logging
from typing import List, Any, Optional

from src.graph.types import State
from src.tools.search import get_web_search_tool
from src.tools.crawl import crawl_tool
from src.tools.retriever import get_retriever_tool

logger = logging.getLogger(__name__)


class ResearchToolManager:
    """Research tool manager - Unified management of all research-related tools"""

    def __init__(self, configurable: Any, state: State):
        """Initialize tool manager

        Args:
            configurable: Configuration object
            state: Current state
        """
        self.configurable = configurable
        self.state = state
        self._tools = None

        # Initialize unified configuration manager
        from .isolation_config_manager import IsolationConfigManager

        config_manager = IsolationConfigManager(configurable)
        self.unified_config = config_manager.get_unified_config()

        logger.info("ResearchToolManager initialized with unified config")

    def _create_web_search_tool(self) -> Any:
        """Create web search tool

        Returns:
            Web search tool instance
        """
        try:
            max_search_results = self.unified_config.max_search_results
            enable_smart_filtering = self.unified_config.enable_smart_filtering

            tool = get_web_search_tool(max_search_results, enable_smart_filtering)
            logger.debug(
                f"Web search tool created with max_results={max_search_results}, "
                f"smart_filtering={enable_smart_filtering}"
            )
            return tool
        except Exception as e:
            logger.error(f"Failed to create web search tool: {e}")
            # Return tool with default configuration
            return get_web_search_tool(10, True)

    def _create_crawl_tool(self) -> Any:
        """Create crawl tool

        Returns:
            Crawl tool instance
        """
        logger.debug("Creating crawl tool")
        return crawl_tool

    def _create_retriever_tool(self) -> Optional[Any]:
        """Create retriever tool

        Returns:
            Retriever tool instance, or None if no resources
        """
        resources = self.state.get("resources", [])
        if not resources:
            logger.debug("No resources found, skipping retriever tool")
            return None

        try:
            tool = get_retriever_tool(resources)
            logger.debug(f"Retriever tool created with {len(resources)} resources")
            return tool
        except Exception as e:
            logger.error(f"Failed to create retriever tool: {e}")
            return None

    def get_basic_tools(self) -> List[Any]:
        """Get basic tools list

        Returns:
            Basic tools list
        """
        logger.debug("Creating basic tools")

        tools = []

        # Add web search tool
        web_search_tool = self._create_web_search_tool()
        tools.append(web_search_tool)

        # Add crawl tool
        crawl_tool_instance = self._create_crawl_tool()
        tools.append(crawl_tool_instance)

        # Add retriever tool (if resources available)
        retriever_tool = self._create_retriever_tool()
        if retriever_tool:
            # Retriever tool should be placed first
            tools.insert(0, retriever_tool)

        logger.info(f"Created {len(tools)} basic tools")
        return tools

    def get_all_tools(self) -> List[Any]:
        """Get all tools (including basic tools and MCP tools)

        Note: MCP tools need to be added externally as they require async context

        Returns:
            All tools list
        """
        if self._tools is None:
            self._tools = self.get_basic_tools()

        return self._tools.copy()

    def add_mcp_tools(self, mcp_tools: List[Any], enabled_tools: dict) -> List[Any]:
        """Add MCP tools to tool list

        Args:
            mcp_tools: MCP tools list
            enabled_tools: Enabled tools mapping

        Returns:
            Complete tool list including MCP tools
        """
        base_tools = self.get_all_tools()

        for tool in mcp_tools:
            if tool.name in enabled_tools:
                # Add server information for MCP tools
                server_name = enabled_tools[tool.name]
                tool.description = f"Powered by '{server_name}'.\n{tool.description}"
                base_tools.append(tool)

        logger.info(f"Added {len(mcp_tools)} MCP tools, total tools: {len(base_tools)}")
        return base_tools

    def get_tool_summary(self) -> dict:
        """Get tool summary information

        Returns:
            Tool summary dictionary
        """
        tools = self.get_all_tools()

        summary = {
            "total_tools": len(tools),
            "tool_names": [getattr(tool, "name", str(tool)) for tool in tools],
            "has_retriever": any("retriever" in str(tool).lower() for tool in tools),
            "has_web_search": any("search" in str(tool).lower() for tool in tools),
            "has_crawl": any("crawl" in str(tool).lower() for tool in tools),
        }

        return summary
