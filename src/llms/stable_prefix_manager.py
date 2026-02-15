"""Stable prefix manager for KV cache optimization.

This module implements the "KV cache first" principle from Manus AI,
optimizing LLM API calls for maximum cache hit rate.

Key principles:
1. Keep prompt prefix stable
2. Make context append-only
3. Use deterministic JSON serialization
4. Insert cache breakpoints where supported
5. Time context is appended as dynamic suffix (not in stable prefix)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class StablePrefixConfig:
    """Configuration for stable prefix management."""

    include_timestamp: bool = False
    deterministic_json: bool = True
    cache_breakpoints: List[int] = field(default_factory=list)
    max_prefix_messages: int = 10
    enable_cache_tracking: bool = True

    def __post_init__(self):
        if not self.cache_breakpoints:
            self.cache_breakpoints = []


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    total_tokens_cached: int = 0
    total_tokens_uncached: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def cost_savings_ratio(self) -> float:
        """Estimate cost savings from caching (cached tokens cost ~10x less)."""
        total = self.total_tokens_cached + self.total_tokens_uncached
        if total == 0:
            return 0.0
        return self.total_tokens_cached / total


class StablePrefixManager:
    """Stable prefix manager for KV cache optimization.

    Implements the "KV cache first" principle from Manus AI:
    - Keep prompt prefix stable (no timestamps at start)
    - Use deterministic JSON serialization
    - Make context append-only
    - Insert cache breakpoints for supported models
    """

    def __init__(self, config: Optional[StablePrefixConfig] = None):
        self.config = config or StablePrefixConfig()
        self._prefix_cache: Dict[str, str] = {}
        self._stats = CacheStats()

    def create_stable_system_prompt(
        self,
        agent_name: str,
        base_prompt: str,
        tools: Optional[List[Dict]] = None,
        dynamic_context: Optional[str] = None,
        current_time: Optional[str] = None,
    ) -> str:
        """Create a stable system prompt prefix.

        Key principles:
        1. System prompt start does not include timestamp
        2. Tool definitions use deterministic serialization
        3. Dynamic content (including time) goes at the end

        Args:
            agent_name: Name of the agent
            base_prompt: Base system prompt
            tools: List of tool definitions
            dynamic_context: Dynamic context to append
            current_time: Current time string to append as dynamic suffix

        Returns:
            Stable system prompt string
        """
        stable_parts = [base_prompt]

        if tools and self.config.deterministic_json:
            sorted_tools = sorted(tools, key=lambda t: t.get("name", ""))
            tools_json = json.dumps(
                sorted_tools, sort_keys=True, ensure_ascii=False, indent=2
            )
            stable_parts.append(f"\n\n## 可用工具\n\n```json\n{tools_json}\n```")

        if current_time:
            stable_parts.append(f"\n\n**Current Date**: {current_time}")

        if dynamic_context:
            stable_parts.append(f"\n\n## 当前上下文\n\n{dynamic_context}")

        return "\n".join(stable_parts)

    def create_append_only_messages(
        self, existing_messages: List[Dict], new_action: Dict, new_observation: str
    ) -> List[Dict]:
        """Create append-only message sequence.

        Key principles:
        1. Never modify previous messages
        2. Only append new actions and observations
        3. Maintain stable message order

        Args:
            existing_messages: Existing message list
            new_action: New action to append
            new_observation: Observation from the action

        Returns:
            New message list with appended content
        """
        messages = [m.copy() for m in existing_messages]

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": new_action.get("id", ""),
                        "name": new_action.get("name", ""),
                        "args": new_action.get("args", {}),
                    }
                ],
            }
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": new_action.get("id", ""),
                "content": new_observation,
            }
        )

        return messages

    def insert_cache_breakpoints(
        self, messages: List[Dict], model_supports_manual_cache: bool = False
    ) -> List[Dict]:
        """Insert cache breakpoints for supported models.

        For models that support manual cache breakpoints (like Claude),
        insert cache markers at key positions.

        Args:
            messages: Message list
            model_supports_manual_cache: Whether model supports manual caching

        Returns:
            Messages with cache breakpoints
        """
        if not model_supports_manual_cache:
            return messages

        result = []
        for i, msg in enumerate(messages):
            result.append(msg)

            if i == 0 and msg.get("role") == "system":
                result.append(
                    {
                        "role": "system",
                        "content": "<cache_breakpoint/>",
                        "_cache_marker": True,
                    }
                )

            if msg.get("role") == "system" and "可用工具" in msg.get("content", ""):
                result.append(
                    {
                        "role": "system",
                        "content": "<cache_breakpoint/>",
                        "_cache_marker": True,
                    }
                )

        return result

    def calculate_prefix_hash(self, messages: List[Dict]) -> str:
        """Calculate prefix hash for cache key.

        Args:
            messages: Message list

        Returns:
            Hash string for the prefix
        """
        prefix_messages = messages[: self.config.max_prefix_messages]

        def serialize_message(msg: Dict) -> str:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                return f"{role}:{content}"
            return f"{role}:{json.dumps(content, sort_keys=True)}"

        content = "|".join(serialize_message(m) for m in prefix_messages)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_cache_hit(self, messages: List[Dict]) -> tuple[bool, str]:
        """Check if prefix is in cache.

        Args:
            messages: Message list

        Returns:
            Tuple of (is_hit, prefix_hash)
        """
        if not self.config.enable_cache_tracking:
            return False, ""

        prefix_hash = self.calculate_prefix_hash(messages)
        self._stats.total_requests += 1

        if prefix_hash in self._prefix_cache:
            self._stats.hits += 1
            return True, prefix_hash

        self._stats.misses += 1
        self._prefix_cache[prefix_hash] = "cached"
        return False, prefix_hash

    def record_token_usage(
        self, cached_tokens: int = 0, uncached_tokens: int = 0
    ) -> None:
        """Record token usage for statistics.

        Args:
            cached_tokens: Number of cached tokens
            uncached_tokens: Number of uncached tokens
        """
        self._stats.total_tokens_cached += cached_tokens
        self._stats.total_tokens_uncached += uncached_tokens

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = CacheStats()
        self._prefix_cache.clear()


class CachedLLMCaller:
    """LLM caller with KV cache optimization.

    Wraps LLM calls with cache optimization logic.
    """

    def __init__(self, llm: Any, prefix_manager: Optional[StablePrefixManager] = None):
        self.llm = llm
        self.prefix_manager = prefix_manager or StablePrefixManager()

    async def invoke_with_cache_optimization(
        self, messages: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Invoke LLM with cache optimization.

        Args:
            messages: Message list
            config: Optional configuration

        Returns:
            LLM response
        """
        config = config or {}

        is_hit, prefix_hash = self.prefix_manager.check_cache_hit(messages)

        if config.get("model_supports_manual_cache"):
            messages = self.prefix_manager.insert_cache_breakpoints(
                messages, model_supports_manual_cache=True
            )

        response = await self.llm.ainvoke(messages)

        if hasattr(response, "usage_metadata"):
            metadata = response.usage_metadata
            if is_hit:
                self.prefix_manager.record_token_usage(
                    cached_tokens=metadata.get("input_tokens", 0)
                )
            else:
                self.prefix_manager.record_token_usage(
                    uncached_tokens=metadata.get("input_tokens", 0)
                )

        return response

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.prefix_manager.get_stats()


def create_deterministic_json(data: Any) -> str:
    """Create deterministic JSON string.

    Ensures consistent key ordering for cache stability.

    Args:
        data: Data to serialize

    Returns:
        Deterministic JSON string
    """
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def normalize_tool_definition(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a tool definition for deterministic serialization.

    Args:
        tool: Tool definition dictionary

    Returns:
        Normalized tool definition
    """
    normalized = {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
    }

    if "parameters" in tool:
        params = tool["parameters"]
        if isinstance(params, dict):
            normalized["parameters"] = {
                "type": params.get("type", "object"),
                "properties": dict(sorted(params.get("properties", {}).items())),
                "required": sorted(params.get("required", [])),
            }

    return normalized


def create_stable_tool_prefix(tools: List[Dict[str, Any]]) -> str:
    """Create stable tool definition prefix.

    Args:
        tools: List of tool definitions

    Returns:
        Stable JSON string of tools
    """
    normalized = [normalize_tool_definition(t) for t in tools]
    sorted_tools = sorted(normalized, key=lambda t: t.get("name", ""))
    return create_deterministic_json(sorted_tools)
