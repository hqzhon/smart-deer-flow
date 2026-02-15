# -*- coding: utf-8 -*-
"""
KV-Cache Optimizer for LLM Token Efficiency.

Implements the Manus AI principles:
1. Keep prompt prefix stable
2. Append-only context (never modify existing content)
3. Track cache hit rate as key metric
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CacheState(Enum):
    """State of the KV-Cache."""

    EMPTY = "empty"
    PREFIX_SET = "prefix_set"
    ACTIVE = "active"
    OPTIMIZED = "optimized"


@dataclass
class CacheStats:
    """Statistics for KV-Cache performance."""

    cache_hits: int = 0
    cache_misses: int = 0
    prefix_length: int = 0
    buffer_length: int = 0
    total_tokens_saved: int = 0
    last_access_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "prefix_length": self.prefix_length,
            "buffer_length": self.buffer_length,
            "total_tokens_saved": self.total_tokens_saved,
        }


class KVCacheOptimizer:
    """KV-Cache optimizer for efficient LLM context management.

    Key principles from Manus AI:
    1. Keep prompt prefix stable - set once, never change
    2. Append-only context - add new content, never modify existing
    3. Track cache hit rate - key metric for optimization
    4. Mask tools instead of removing - keep KV-cache stable

    Usage:
        optimizer = KVCacheOptimizer()
        optimizer.set_stable_prefix(system_prompt)
        optimizer.append_context(user_message)
        context = optimizer.get_optimized_context()
    """

    def __init__(self, max_buffer_size: int = 100):
        self._stable_prefix: str = ""
        self._append_buffer: List[str] = []
        self._state: CacheState = CacheState.EMPTY
        self._stats = CacheStats()
        self._max_buffer_size = max_buffer_size

        self._tool_registry: List[str] = []
        self._tool_mask: Dict[str, bool] = {}

    def set_stable_prefix(self, prefix: str) -> bool:
        """Set the stable prefix for KV-Cache optimization.

        This should only be called once per session.
        The prefix typically contains:
        - System instructions
        - Role definition
        - Available tools description

        Args:
            prefix: The stable prefix content

        Returns:
            True if set successfully, False if already set
        """
        if self._state != CacheState.EMPTY:
            logger.warning("Stable prefix already set, cannot modify")
            return False

        self._stable_prefix = prefix
        self._stats.prefix_length = len(prefix)
        self._state = CacheState.PREFIX_SET

        logger.info(
            f"KV-Cache stable prefix set: {len(prefix)} chars, "
            f"~{len(prefix) // 4} tokens"
        )
        return True

    def append_context(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Append content to the context buffer.

        This follows the append-only principle - content is never modified
        once added, ensuring KV-Cache stability.

        Args:
            content: Content to append
            metadata: Optional metadata for tracking
        """
        if self._state == CacheState.EMPTY:
            logger.warning("Cannot append context before setting stable prefix")
            return

        if len(self._append_buffer) >= self._max_buffer_size:
            self._evict_oldest()

        self._append_buffer.append(content)
        self._stats.buffer_length = len(self._append_buffer)
        self._state = CacheState.ACTIVE

        logger.debug(
            f"Appended context: {len(content)} chars, buffer size: {len(self._append_buffer)}"
        )

    def get_optimized_context(self) -> str:
        """Get the optimized context for LLM input.

        Returns:
            The full context with stable prefix and appended content
        """
        if self._state == CacheState.EMPTY:
            logger.warning("KV-Cache not initialized, returning empty string")
            return ""

        self._stats.cache_hits += 1
        self._stats.last_access_time = time.time()

        if not self._append_buffer:
            return self._stable_prefix

        return self._stable_prefix + "\n\n" + "\n\n".join(self._append_buffer)

    def get_incremental_context(self, since_index: int = 0) -> Tuple[str, int]:
        """Get only the new content since a given index.

        This is useful for streaming scenarios where only new content
        needs to be processed.

        Args:
            since_index: Index to start from

        Returns:
            Tuple of (new_content, new_index)
        """
        if since_index >= len(self._append_buffer):
            return "", len(self._append_buffer)

        new_content = "\n\n".join(self._append_buffer[since_index:])
        return new_content, len(self._append_buffer)

    def register_tools(self, tools: List[str]) -> None:
        """Register all available tools for masking.

        Tools are registered once and then masked/unmasked
        rather than added/removed, keeping KV-Cache stable.

        Args:
            tools: List of tool names
        """
        self._tool_registry = tools.copy()
        self._tool_mask = {tool: True for tool in tools}
        logger.info(f"Registered {len(tools)} tools for KV-Cache optimization")

    def mask_tools(self, unavailable_tools: List[str]) -> Dict[str, bool]:
        """Mask tools as unavailable without removing them.

        This follows the Manus AI principle: "Mask instead of remove"
        to maintain KV-Cache stability.

        Args:
            unavailable_tools: Tools to mask as unavailable

        Returns:
            Current tool mask state
        """
        for tool in unavailable_tools:
            if tool in self._tool_mask:
                self._tool_mask[tool] = False

        available_count = sum(1 for v in self._tool_mask.values() if v)
        logger.info(
            f"Tool masking updated: {available_count}/{len(self._tool_registry)} available"
        )

        return self._tool_mask.copy()

    def unmask_tools(self, tools: List[str]) -> Dict[str, bool]:
        """Unmask tools to make them available again.

        Args:
            tools: Tools to unmask

        Returns:
            Current tool mask state
        """
        for tool in tools:
            if tool in self._tool_mask:
                self._tool_mask[tool] = True

        return self._tool_mask.copy()

    def get_available_tools(self) -> List[str]:
        """Get list of currently available (unmasked) tools."""
        return [tool for tool, available in self._tool_mask.items() if available]

    def get_tool_mask_info(self) -> Dict[str, Any]:
        """Get information about tool masking state."""
        return {
            "total_tools": len(self._tool_registry),
            "available_tools": len(self.get_available_tools()),
            "masked_tools": len(self._tool_registry) - len(self.get_available_tools()),
            "mask_state": self._tool_mask.copy(),
        }

    def _evict_oldest(self) -> None:
        """Evict the oldest content from the buffer.

        This is called when the buffer exceeds max size.
        The evicted content can be stored elsewhere if needed.
        """
        if self._append_buffer:
            evicted = self._append_buffer.pop(0)
            self._stats.cache_misses += 1
            logger.debug(f"Evicted oldest content: {len(evicted)} chars")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get_state(self) -> CacheState:
        """Get current cache state."""
        return self._state

    def reset(self, keep_prefix: bool = False) -> None:
        """Reset the optimizer.

        Args:
            keep_prefix: If True, keep the stable prefix
        """
        self._append_buffer.clear()
        self._stats.buffer_length = 0

        if not keep_prefix:
            self._stable_prefix = ""
            self._stats.prefix_length = 0
            self._state = CacheState.EMPTY
            self._tool_registry.clear()
            self._tool_mask.clear()
        else:
            self._state = CacheState.PREFIX_SET

        logger.info(f"KV-Cache optimizer reset (keep_prefix={keep_prefix})")


class KVCacheManager:
    """Manager for multiple KV-Cache optimizers.

    Supports multiple sessions/threads with independent caches.
    """

    def __init__(self):
        self._caches: Dict[str, KVCacheOptimizer] = {}

    def get_cache(self, session_id: str) -> KVCacheOptimizer:
        """Get or create a cache for a session."""
        if session_id not in self._caches:
            self._caches[session_id] = KVCacheOptimizer()
        return self._caches[session_id]

    def remove_cache(self, session_id: str) -> bool:
        """Remove a cache for a session."""
        if session_id in self._caches:
            del self._caches[session_id]
            return True
        return False

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            session_id: cache.get_stats().to_dict()
            for session_id, cache in self._caches.items()
        }

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all caches."""
        total_hits = sum(c.get_stats().cache_hits for c in self._caches.values())
        total_misses = sum(c.get_stats().cache_misses for c in self._caches.values())
        total_saved = sum(
            c.get_stats().total_tokens_saved for c in self._caches.values()
        )

        return {
            "total_sessions": len(self._caches),
            "total_cache_hits": total_hits,
            "total_cache_misses": total_misses,
            "aggregate_hit_rate": (
                f"{total_hits / (total_hits + total_misses):.2%}"
                if (total_hits + total_misses) > 0
                else "0%"
            ),
            "total_tokens_saved": total_saved,
        }


_global_cache_manager: Optional[KVCacheManager] = None


def get_global_cache_manager() -> KVCacheManager:
    """Get the global KV-Cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = KVCacheManager()
    return _global_cache_manager
