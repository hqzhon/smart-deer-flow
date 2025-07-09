# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Advanced memory management and caching for DeerFlow

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
import hashlib
import pickle
import threading

from src.utils.decorators import safe_background_task

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheLevel(Enum):
    """Cache levels for hierarchical memory architecture."""

    L1_MEMORY = 1  # In-memory cache (fastest)
    L2_DISK = 2  # Disk-based cache (medium)
    L3_REMOTE = 3  # Remote cache (slowest)


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    priority: int = 1  # Higher number = higher priority

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()

    def _calculate_size(self) -> int:
        """Calculate approximate size of the cached value."""
        try:
            return len(pickle.dumps(self.value))
        except Exception:
            # Fallback for non-picklable objects
            return len(str(self.value).encode("utf-8"))

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class HierarchicalMemoryManager(Generic[T]):
    """Hierarchical memory manager with L1/L2/L3 cache levels."""

    def __init__(
        self,
        l1_max_size: int = 100 * 1024 * 1024,  # 100MB
        l2_max_size: int = 1024 * 1024 * 1024,  # 1GB
        l3_max_size: int = 10 * 1024 * 1024 * 1024,  # 10GB
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
        enable_compression: bool = True,
    ):
        self.l1_max_size = l1_max_size
        self.l2_max_size = l2_max_size
        self.l3_max_size = l3_max_size
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression

        # L1 Cache (Memory)
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_current_size = 0
        self.l1_lock = threading.RLock()

        # L2 Cache (Disk) - simplified for demo
        self.l2_cache: Dict[str, CacheEntry] = {}
        self.l2_current_size = 0
        self.l2_lock = threading.RLock()

        # L3 Cache (Remote) - placeholder
        self.l3_cache: Dict[str, CacheEntry] = {}
        self.l3_current_size = 0
        self.l3_lock = threading.RLock()

        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "evictions": 0,
            "promotions": 0,
        }

        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300  # 5 minutes

    async def start(self):
        """Start the memory manager."""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
        logger.info("Hierarchical memory manager started")

    async def stop(self):
        """Stop the memory manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Hierarchical memory manager stopped")

    def _generate_key(self, key: Union[str, Dict[str, Any]]) -> str:
        """Generate a consistent cache key."""
        if isinstance(key, str):
            return key
        elif isinstance(key, dict):
            # Sort dict for consistent hashing
            sorted_items = sorted(key.items())
            key_str = json.dumps(sorted_items, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()

    async def get(
        self, key: Union[str, Dict[str, Any]], default: T = None
    ) -> Optional[T]:
        """Get value from hierarchical cache."""
        cache_key = self._generate_key(key)

        # Try L1 cache first
        entry = self._get_from_l1(cache_key)
        if entry:
            self.stats["l1_hits"] += 1
            return entry.value

        # Try L2 cache
        entry = await self._get_from_l2(cache_key)
        if entry:
            self.stats["l2_hits"] += 1
            # Promote to L1
            await self._promote_to_l1(entry)
            return entry.value

        # Try L3 cache
        entry = await self._get_from_l3(cache_key)
        if entry:
            self.stats["l3_hits"] += 1
            # Promote to L1
            await self._promote_to_l1(entry)
            return entry.value

        # Cache miss
        self.stats["misses"] += 1
        return default

    async def set(
        self,
        key: Union[str, Dict[str, Any]],
        value: T,
        ttl: Optional[float] = None,
        priority: int = 1,
    ) -> bool:
        """Set value in hierarchical cache."""
        cache_key = self._generate_key(key)

        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            ttl=ttl,
            priority=priority,
        )

        # Always try to store in L1 first
        success = await self._set_in_l1(entry)
        if not success:
            # If L1 is full, try L2
            success = await self._set_in_l2(entry)
            if not success:
                # If L2 is full, try L3
                success = await self._set_in_l3(entry)

        return success

    def _get_from_l1(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L1 cache."""
        with self.l1_lock:
            entry = self.l1_cache.get(key)
            if entry and not entry.is_expired():
                entry.update_access()
                # Move to end (most recently used)
                self.l1_cache.move_to_end(key)
                return entry
            elif entry:
                # Remove expired entry
                self._remove_from_l1(key)
        return None

    async def _get_from_l2(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L2 cache."""
        with self.l2_lock:
            entry = self.l2_cache.get(key)
            if entry and not entry.is_expired():
                entry.update_access()
                return entry
            elif entry:
                # Remove expired entry
                self._remove_from_l2(key)
        return None

    async def _get_from_l3(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L3 cache."""
        with self.l3_lock:
            entry = self.l3_cache.get(key)
            if entry and not entry.is_expired():
                entry.update_access()
                return entry
            elif entry:
                # Remove expired entry
                self._remove_from_l3(key)
        return None

    async def _set_in_l1(self, entry: CacheEntry) -> bool:
        """Set entry in L1 cache."""
        with self.l1_lock:
            # Check if we need to evict
            while (
                self.l1_current_size + entry.size_bytes > self.l1_max_size
                and len(self.l1_cache) > 0
            ):
                await self._evict_from_l1()

            # Check if we have space now
            if self.l1_current_size + entry.size_bytes <= self.l1_max_size:
                self.l1_cache[entry.key] = entry
                self.l1_current_size += entry.size_bytes
                return True

        return False

    async def _set_in_l2(self, entry: CacheEntry) -> bool:
        """Set entry in L2 cache."""
        with self.l2_lock:
            # Check if we need to evict
            while (
                self.l2_current_size + entry.size_bytes > self.l2_max_size
                and len(self.l2_cache) > 0
            ):
                await self._evict_from_l2()

            # Check if we have space now
            if self.l2_current_size + entry.size_bytes <= self.l2_max_size:
                self.l2_cache[entry.key] = entry
                self.l2_current_size += entry.size_bytes
                return True

        return False

    async def _set_in_l3(self, entry: CacheEntry) -> bool:
        """Set entry in L3 cache."""
        with self.l3_lock:
            # Check if we need to evict
            while (
                self.l3_current_size + entry.size_bytes > self.l3_max_size
                and len(self.l3_cache) > 0
            ):
                await self._evict_from_l3()

            # Check if we have space now
            if self.l3_current_size + entry.size_bytes <= self.l3_max_size:
                self.l3_cache[entry.key] = entry
                self.l3_current_size += entry.size_bytes
                return True

        return False

    async def _promote_to_l1(self, entry: CacheEntry):
        """Promote entry to L1 cache."""
        success = await self._set_in_l1(entry)
        if success:
            self.stats["promotions"] += 1

    async def _evict_from_l1(self):
        """Evict entry from L1 cache based on eviction policy."""
        if not self.l1_cache:
            return

        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first item in OrderedDict)
            key, entry = self.l1_cache.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            key = min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k].access_count)
            entry = self.l1_cache.pop(key)
        else:
            # Default to LRU
            key, entry = self.l1_cache.popitem(last=False)

        self.l1_current_size -= entry.size_bytes
        self.stats["evictions"] += 1

        # Try to demote to L2
        await self._set_in_l2(entry)

    async def _evict_from_l2(self):
        """Evict entry from L2 cache."""
        if not self.l2_cache:
            return

        # Simple LRU for L2
        key = min(self.l2_cache.keys(), key=lambda k: self.l2_cache[k].last_accessed)
        entry = self.l2_cache.pop(key)
        self.l2_current_size -= entry.size_bytes
        self.stats["evictions"] += 1

        # Try to demote to L3
        await self._set_in_l3(entry)

    async def _evict_from_l3(self):
        """Evict entry from L3 cache."""
        if not self.l3_cache:
            return

        # Simple LRU for L3
        key = min(self.l3_cache.keys(), key=lambda k: self.l3_cache[k].last_accessed)
        entry = self.l3_cache.pop(key)
        self.l3_current_size -= entry.size_bytes
        self.stats["evictions"] += 1

    def _remove_from_l1(self, key: str):
        """Remove entry from L1 cache."""
        entry = self.l1_cache.pop(key, None)
        if entry:
            self.l1_current_size -= entry.size_bytes

    def _remove_from_l2(self, key: str):
        """Remove entry from L2 cache."""
        entry = self.l2_cache.pop(key, None)
        if entry:
            self.l2_current_size -= entry.size_bytes

    def _remove_from_l3(self, key: str):
        """Remove entry from L3 cache."""
        entry = self.l3_cache.pop(key, None)
        if entry:
            self.l3_current_size -= entry.size_bytes

    @safe_background_task
    async def _background_cleanup(self):
        """Background task for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")

    async def _cleanup_expired_entries(self):
        """Clean up expired entries from all cache levels."""
        current_time = time.time()

        # Clean L1
        with self.l1_lock:
            expired_keys = [
                key for key, entry in self.l1_cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_from_l1(key)

        # Clean L2
        with self.l2_lock:
            expired_keys = [
                key for key, entry in self.l2_cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_from_l2(key)

        # Clean L3
        with self.l3_lock:
            expired_keys = [
                key for key, entry in self.l3_cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_from_l3(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum(self.stats.values())
        hit_rate = 0
        if total_requests > 0:
            hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
            hit_rate = hits / total_requests

        return {
            "hit_rate": hit_rate,
            "l1_entries": len(self.l1_cache),
            "l2_entries": len(self.l2_cache),
            "l3_entries": len(self.l3_cache),
            "l1_size_mb": self.l1_current_size / (1024 * 1024),
            "l2_size_mb": self.l2_current_size / (1024 * 1024),
            "l3_size_mb": self.l3_current_size / (1024 * 1024),
            "l1_utilization": self.l1_current_size / self.l1_max_size,
            "l2_utilization": self.l2_current_size / self.l2_max_size,
            "l3_utilization": self.l3_current_size / self.l3_max_size,
            **self.stats,
        }


class SmartCacheDecorator:
    """Smart caching decorator with automatic key generation."""

    def __init__(
        self,
        memory_manager: HierarchicalMemoryManager,
        ttl: Optional[float] = None,
        priority: int = 1,
        key_func: Optional[Callable] = None,
    ):
        self.memory_manager = memory_manager
        self.ttl = ttl
        self.priority = priority
        self.key_func = key_func

    def __call__(self, func: Callable) -> Callable:
        import asyncio

        if asyncio.iscoroutinefunction(func):
            # Async function handling
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if self.key_func:
                    cache_key = self.key_func(*args, **kwargs)
                else:
                    cache_key = {"func": func.__name__, "args": args, "kwargs": kwargs}

                # Try to get from cache
                result = await self.memory_manager.get(cache_key)
                if result is not None:
                    return result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.memory_manager.set(
                    cache_key, result, ttl=self.ttl, priority=self.priority
                )

                return result

            return async_wrapper
        else:
            # Sync function handling - use simple memory cache to avoid async complexity
            import threading
            import time
            import hashlib
            import json

            # Create simple thread-safe cache for sync functions
            if not hasattr(self, "_sync_cache"):
                self._sync_cache = {}
                self._sync_cache_lock = threading.Lock()

            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                if self.key_func:
                    cache_key_data = self.key_func(*args, **kwargs)
                else:
                    cache_key_data = {
                        "func": func.__name__,
                        "args": args,
                        "kwargs": kwargs,
                    }

                # Create string cache key
                try:
                    cache_key_str = json.dumps(
                        cache_key_data, sort_keys=True, default=str
                    )
                    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
                except Exception:
                    # If serialization fails, use string representation of function name and parameters
                    cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"

                current_time = time.time()

                # Try to get from cache
                with self._sync_cache_lock:
                    if cache_key in self._sync_cache:
                        cached_item = self._sync_cache[cache_key]
                        # Check if expired
                        if (
                            self.ttl is None
                            or (current_time - cached_item["timestamp"]) < self.ttl
                        ):
                            return cached_item["value"]
                        else:
                            # Expired, delete cache item
                            del self._sync_cache[cache_key]

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                with self._sync_cache_lock:
                    self._sync_cache[cache_key] = {
                        "value": result,
                        "timestamp": current_time,
                    }

                    # Simple cache size control (keep recent 100 items)
                    if len(self._sync_cache) > 100:
                        # Delete oldest item
                        oldest_key = min(
                            self._sync_cache.keys(),
                            key=lambda k: self._sync_cache[k]["timestamp"],
                        )
                        del self._sync_cache[oldest_key]

                return result

            return sync_wrapper

    async def _sync_cache_get(self, cache_key):
        """Helper method for sync cache retrieval"""
        return await self.memory_manager.get(cache_key)


# Global memory manager instance
global_memory_manager = HierarchicalMemoryManager()


def cached(
    ttl: Optional[float] = None, priority: int = 1, key_func: Optional[Callable] = None
):
    """Decorator for caching function results."""
    return SmartCacheDecorator(
        global_memory_manager, ttl=ttl, priority=priority, key_func=key_func
    )
