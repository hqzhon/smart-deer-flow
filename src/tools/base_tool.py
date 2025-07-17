"""
Base tool interface for the tool registry system.
All tools must inherit from BaseTool to be automatically discovered.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Base input model for tools."""

    pass


class ToolOutput(BaseModel):
    """Base output model for tools."""

    success: bool = Field(default=True)
    message: str = Field(default="")
    data: Optional[Any] = None


class BaseTool(ABC):
    """Abstract base class for all tools in the system.

    All tools must inherit from this class to be automatically discovered
    by the ToolRegistry.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name - must be unique across all tools."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for agent understanding."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for tool parameters."""
        return {}

    @property
    def required_parameters(self) -> List[str]:
        """List of required parameter names."""
        return []

    @abstractmethod
    def execute(self, **kwargs) -> ToolOutput:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters based on the schema.

        Returns:
            ToolOutput containing execution results.
        """
        pass

    @property
    def category(self) -> str:
        """Tool category for organization."""
        return "general"

    @property
    def tags(self) -> List[str]:
        """Tool tags for filtering and discovery."""
        return []

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            True if valid, False otherwise.
        """
        required = set(self.required_parameters)
        provided = set(kwargs.keys())

        # Check if all required parameters are provided
        if not required.issubset(provided):
            missing = required - provided
            raise ValueError(f"Missing required parameters: {missing}")

        return True

    def get_schema(self) -> Dict[str, Any]:
        """Get complete tool schema for agent consumption."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_parameters,
                },
            },
        }

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class ToolRegistry:
    """Registry for managing and discovering tools automatically."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._loaded = False

    def discover_tools(self, tools_dir: Optional[str] = None) -> None:
        """Automatically discover and register all tools.

        Args:
            tools_dir: Directory to scan for tools. Defaults to src/tools/.
        """
        if tools_dir is None:
            tools_dir = Path(__file__).parent
        else:
            tools_dir = Path(tools_dir)

        if not tools_dir.exists():
            logger.warning(f"Tools directory does not exist: {tools_dir}")
            return

        # Import all Python files in the tools directory
        import importlib.util
        import sys

        for py_file in tools_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"tools.{py_file.stem}", py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find all BaseTool subclasses
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, BaseTool)
                            and attr != BaseTool
                        ):

                            try:
                                tool_instance = attr()
                                self.register_tool(tool_instance)
                            except Exception as e:
                                logger.error(
                                    f"Failed to instantiate tool {attr_name}: {e}"
                                )

            except Exception as e:
                logger.error(f"Failed to load tools from {py_file}: {e}")

        self._loaded = True
        logger.info(f"Discovered {len(self._tools)} tools")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: Tool instance to register.
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self._tools[tool.name] = tool

        # Add to category
        if tool.category not in self._categories:
            self._categories[tool.category] = []

        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)

        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve.

        Returns:
            Tool instance or None if not found.
        """
        if not self._loaded:
            self.discover_tools()

        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        if not self._loaded:
            self.discover_tools()

        return list(self._tools.keys())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a category.

        Args:
            category: Tool category.

        Returns:
            List of tool instances in the category.
        """
        if not self._loaded:
            self.discover_tools()

        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools.

        Returns:
            List of tool schemas for agent consumption.
        """
        if not self._loaded:
            self.discover_tools()

        return [tool.get_schema() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """Get formatted description of all tools for prompts.

        Returns:
            Formatted string describing all available tools.
        """
        if not self._loaded:
            self.discover_tools()

        if not self._tools:
            return "No tools available."

        descriptions = []
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        return "\n".join(descriptions)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if tool exists, False otherwise.
        """
        if not self._loaded:
            self.discover_tools()

        return tool_name in self._tools

    def reload_tools(self) -> None:
        """Reload all tools from disk."""
        self._tools.clear()
        self._categories.clear()
        self._loaded = False
        self.discover_tools()

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            tool_name: Name of the tool to remove.

        Returns:
            True if tool was removed, False if it didn't exist.
        """
        if tool_name not in self._tools:
            return False

        tool = self._tools[tool_name]

        # Remove from category
        if tool.category in self._categories:
            if tool_name in self._categories[tool.category]:
                self._categories[tool.category].remove(tool_name)
                if not self._categories[tool.category]:
                    del self._categories[tool.category]

        # Remove from tools
        del self._tools[tool_name]
        logger.info(f"Removed tool: {tool_name}")

        return True


# Global tool registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance.

    Returns:
        Tool registry instance.
    """
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_available_tools() -> List[str]:
    """Convenience function to list available tools.

    Returns:
        List of available tool names.
    """
    registry = get_tool_registry()
    return registry.list_tools()


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Convenience function to get all tool schemas.

    Returns:
        List of tool schemas.
    """
    registry = get_tool_registry()
    return registry.get_all_schemas()


# Import logging for the module
import logging

logger = logging.getLogger(__name__)
