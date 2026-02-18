import logging
from typing import Any, Dict, List

from src.tools.base_tool import BaseTool
from src.tools.tool_result import ToolFailure, ToolResult

logger = logging.getLogger(__name__)


class ToolCollection:
    def __init__(self, *tools: BaseTool):
        self.tools: tuple[BaseTool, ...] = tools
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.get_schema() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] | None = None
    ) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} not found")
        try:
            tool_input = tool_input or {}
            result = tool.execute(**tool_input)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(output=result)
        except Exception as e:
            return ToolFailure(error=str(e))

    async def execute_all(self) -> List[ToolResult]:
        results = []
        for tool in self.tools:
            try:
                result = tool.execute()
                if isinstance(result, ToolResult):
                    results.append(result)
                else:
                    results.append(ToolResult(output=result))
            except Exception as e:
                results.append(ToolFailure(error=str(e)))
        return results

    def get_tool(self, name: str) -> BaseTool | None:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool) -> "ToolCollection":
        if tool.name in self.tool_map:
            logger.warning(f"Tool {tool.name} already exists in collection, skipping")
            return self
        self.tools = self.tools + (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool) -> "ToolCollection":
        for tool in tools:
            self.add_tool(tool)
        return self

    def remove_tool(self, name: str) -> "ToolCollection":
        if name in self.tool_map:
            del self.tool_map[name]
            self.tools = tuple(t for t in self.tools if t.name != name)
        return self

    def has_tool(self, name: str) -> bool:
        return name in self.tool_map

    def get_tool_names(self) -> List[str]:
        return list(self.tool_map.keys())
