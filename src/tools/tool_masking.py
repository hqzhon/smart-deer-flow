"""Tool masking mechanism for action space constraint.

This module implements the "mask rather than remove" principle from Manus AI,
constraining tool selection through logit masking instead of removing
tool definitions from context.

Key principles:
1. Don't remove tools from context (breaks KV cache)
2. Use logit masking to constrain action selection
3. Support prefill-based constraints for supported models
"""

from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ToolGroup(Enum):
    """Tool groups for masking."""

    SEARCH = "search"
    BROWSER = "browser"
    SHELL = "shell"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    RAG = "rag"
    TTS = "tts"
    CRAWL = "crawl"


@dataclass
class ToolMaskingRule:
    """Rule for tool masking."""

    allowed_groups: Set[ToolGroup] = field(default_factory=set)
    allowed_tools: Set[str] = field(default_factory=set)
    blocked_tools: Set[str] = field(default_factory=set)

    def is_tool_allowed(
        self, tool_name: str, tool_group: Optional[ToolGroup] = None
    ) -> bool:
        """Check if a tool is allowed by this rule.

        Args:
            tool_name: Name of the tool
            tool_group: Group the tool belongs to

        Returns:
            True if the tool is allowed
        """
        if tool_name in self.blocked_tools:
            return False

        if tool_name in self.allowed_tools:
            return True

        if tool_group and tool_group in self.allowed_groups:
            return True

        if not self.allowed_groups and not self.allowed_tools:
            return True

        return False


class ToolMaskingManager:
    """Tool masking manager - implements "mask rather than remove" principle.

    This class manages tool availability through masking rather than
    removal, preserving KV cache efficiency.
    """

    def __init__(self):
        self.tool_groups: Dict[str, ToolGroup] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self._initialize_default_tools()

    def _initialize_default_tools(self) -> None:
        """Initialize default tool groupings."""
        search_tools = {
            "web_search": "Web search for information",
            "arxiv_search": "Search arXiv for academic papers",
            "tavily_search": "Search using Tavily API",
            "duckduckgo_search": "Search using DuckDuckGo",
            "brave_search": "Search using Brave Search",
        }

        browser_tools = {
            "browser_navigate": "Navigate to a URL",
            "browser_click": "Click an element",
            "browser_scroll": "Scroll the page",
            "browser_read": "Read page content",
        }

        shell_tools = {
            "shell_execute": "Execute shell command",
            "shell_read": "Read file content",
            "python_repl": "Execute Python code",
        }

        analysis_tools = {
            "data_analysis": "Analyze data",
            "python_repl": "Execute Python code for analysis",
        }

        rag_tools = {
            "rag_retriever": "Retrieve from RAG knowledge base",
            "document_search": "Search documents",
        }

        crawl_tools = {
            "crawl_tool": "Crawl web pages",
            "jina_reader": "Read web content via Jina",
        }

        tts_tools = {
            "text_to_speech": "Convert text to speech",
        }

        for name, desc in search_tools.items():
            self.register_tool(name, ToolGroup.SEARCH, desc)

        for name, desc in browser_tools.items():
            self.register_tool(name, ToolGroup.BROWSER, desc)

        for name, desc in shell_tools.items():
            self.register_tool(name, ToolGroup.SHELL, desc)

        for name, desc in analysis_tools.items():
            self.register_tool(name, ToolGroup.ANALYSIS, desc)

        for name, desc in rag_tools.items():
            self.register_tool(name, ToolGroup.RAG, desc)

        for name, desc in crawl_tools.items():
            self.register_tool(name, ToolGroup.CRAWL, desc)

        for name, desc in tts_tools.items():
            self.register_tool(name, ToolGroup.TTS, desc)

    def register_tool(self, name: str, group: ToolGroup, description: str = "") -> None:
        """Register a tool with its group.

        Args:
            name: Tool name
            group: Tool group
            description: Tool description
        """
        self.tool_groups[name] = group
        self.tool_descriptions[name] = description

    def get_tool_group(self, name: str) -> Optional[ToolGroup]:
        """Get the group for a tool.

        Args:
            name: Tool name

        Returns:
            Tool group or None
        """
        return self.tool_groups.get(name)

    def get_masking_rule_for_state(self, state: Dict[str, Any]) -> ToolMaskingRule:
        """Get masking rule based on current state.

        Implements the "mask rather than remove" principle:
        - Don't remove tools from context
        - Constrain action selection through masking

        Args:
            state: Current workflow state

        Returns:
            ToolMaskingRule for the current state
        """
        current_phase = state.get("current_phase", "research")
        node_type = state.get("node_type", "")

        if node_type == "planner" or current_phase == "planning":
            return ToolMaskingRule(
                allowed_groups={ToolGroup.SEARCH, ToolGroup.RAG},
                allowed_tools=set(),
                blocked_tools=set(),
            )

        elif node_type == "researcher" or current_phase == "research":
            return ToolMaskingRule(
                allowed_groups={
                    ToolGroup.SEARCH,
                    ToolGroup.BROWSER,
                    ToolGroup.ANALYSIS,
                    ToolGroup.CRAWL,
                    ToolGroup.RAG,
                },
                allowed_tools=set(),
                blocked_tools=set(),
            )

        elif node_type == "coder" or current_phase == "coding":
            return ToolMaskingRule(
                allowed_groups={ToolGroup.SHELL, ToolGroup.ANALYSIS},
                allowed_tools=set(),
                blocked_tools=set(),
            )

        elif node_type == "reporter" or current_phase == "reporting":
            return ToolMaskingRule(
                allowed_groups={ToolGroup.ANALYSIS},
                allowed_tools={"python_repl"},
                blocked_tools={"browser_navigate", "browser_click", "shell_execute"},
            )

        else:
            return ToolMaskingRule(
                allowed_groups=set(ToolGroup), allowed_tools=set(), blocked_tools=set()
            )

    def create_prefill_prefix(self, rule: ToolMaskingRule) -> str:
        """Create prefill prefix for tool selection constraint.

        For models that support response prefilling, this creates
        a prefix that constrains tool selection.

        Args:
            rule: Masking rule to apply

        Returns:
            Prefill prefix string
        """
        allowed_prefixes = []

        for tool_name, group in self.tool_groups.items():
            if rule.is_tool_allowed(tool_name, group):
                allowed_prefixes.append(tool_name)

        if not allowed_prefixes:
            return ""

        if len(allowed_prefixes) == 1:
            return f'{{"name": "{allowed_prefixes[0]}"'
        else:
            return '{"name": "'

    def get_allowed_tools(self, rule: ToolMaskingRule) -> List[str]:
        """Get list of allowed tools for a rule.

        Args:
            rule: Masking rule

        Returns:
            List of allowed tool names
        """
        allowed = []
        for tool_name, group in self.tool_groups.items():
            if rule.is_tool_allowed(tool_name, group):
                allowed.append(tool_name)
        return allowed

    def filter_tool_calls(
        self, tool_calls: List[Dict[str, Any]], rule: ToolMaskingRule
    ) -> List[Dict[str, Any]]:
        """Filter tool calls based on masking rule.

        Args:
            tool_calls: List of tool calls
            rule: Masking rule to apply

        Returns:
            Filtered list of tool calls
        """
        filtered = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_group = self.tool_groups.get(tool_name)

            if rule.is_tool_allowed(tool_name, tool_group):
                filtered.append(call)

        return filtered


def apply_tool_masking(
    state: Dict[str, Any], tool_calls: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Apply tool masking to tool calls.

    Convenience function for applying masking in workflow nodes.

    Args:
        state: Current workflow state
        tool_calls: List of tool calls to filter

    Returns:
        Filtered tool calls
    """
    manager = ToolMaskingManager()
    rule = manager.get_masking_rule_for_state(state)
    return manager.filter_tool_calls(tool_calls, rule)


def get_tool_masking_prefill(state: Dict[str, Any]) -> str:
    """Get prefill prefix for tool selection.

    Args:
        state: Current workflow state

    Returns:
        Prefill prefix string
    """
    manager = ToolMaskingManager()
    rule = manager.get_masking_rule_for_state(state)
    return manager.create_prefill_prefix(rule)


class ToolAvailabilityChecker:
    """Checker for tool availability based on state."""

    def __init__(self, manager: Optional[ToolMaskingManager] = None):
        self.manager = manager or ToolMaskingManager()

    def is_tool_available(self, tool_name: str, state: Dict[str, Any]) -> bool:
        """Check if a tool is available in current state.

        Args:
            tool_name: Tool name to check
            state: Current workflow state

        Returns:
            True if tool is available
        """
        rule = self.manager.get_masking_rule_for_state(state)
        tool_group = self.manager.get_tool_group(tool_name)
        return rule.is_tool_allowed(tool_name, tool_group)

    def get_available_tools(self, state: Dict[str, Any]) -> List[str]:
        """Get all available tools for current state.

        Args:
            state: Current workflow state

        Returns:
            List of available tool names
        """
        rule = self.manager.get_masking_rule_for_state(state)
        return self.manager.get_allowed_tools(rule)

    def get_tool_descriptions(self, state: Dict[str, Any]) -> Dict[str, str]:
        """Get descriptions of available tools.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of tool name to description
        """
        available = self.get_available_tools(state)
        return {
            name: self.manager.tool_descriptions.get(name, "") for name in available
        }
