"""Configuration Cache Optimization Module - Implements intelligent configuration caching and tool reuse mechanisms"""

import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from threading import RLock
from weakref import WeakValueDictionary
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry data class"""

    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache is expired"""
        if self.ttl_seconds is None:
            return False

        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def update_access(self):
        """Update access information"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """Get cache age (seconds)"""
        return (datetime.now() - self.created_at).total_seconds()

    def get_idle_seconds(self) -> float:
        """Get idle time (seconds)"""
        return (datetime.now() - self.last_accessed).total_seconds()


class SmartCache:
    """Smart cache system - supports TTL, LRU and intelligent invalidation strategies"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

        logger.debug(
            f"Smart cache initialized with max_size={max_size}, default_ttl={default_ttl}"
        )

    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                logger.debug(f"Cache entry expired: {key}")
                return None

            entry.update_access()
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set cache value"""
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            # Check if space cleanup is needed
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            # Create cache entry
            entry = CacheEntry(value=value, ttl_seconds=ttl, metadata=metadata or {})

            self._cache[key] = entry
            logger.debug(f"Cache entry set: {key} (TTL: {ttl}s)")
            return True

    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache entry deleted: {key}")
                return True
            return False

    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            logger.debug("Cache cleared")

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        with self._lock:
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._stats["expirations"] += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if not self._cache:
            return False

        # Find least recently used entry
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)

        del self._cache[lru_key]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted LRU entry: {lru_key}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / max(total_requests, 1)

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "stats": self._stats.copy(),
                "memory_usage_estimate": self._estimate_memory_usage(),
            }

    def _estimate_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage"""
        # Simple memory usage estimation
        total_entries = len(self._cache)
        avg_key_size = sum(len(k.encode("utf-8")) for k in self._cache.keys()) // max(
            total_entries, 1
        )

        return {
            "total_entries": total_entries,
            "estimated_avg_key_size_bytes": avg_key_size,
            "estimated_total_size_kb": (
                (total_entries * (avg_key_size + 200)) // 1024
            ),  # Rough estimation
        }


class ConfigCacheOptimizer:
    """Configuration cache optimizer - manages intelligent caching of configurations and tools"""

    def __init__(self, config_ttl: int = 3600, tool_ttl: int = 1800):
        self.config_cache = SmartCache(max_size=500, default_ttl=config_ttl)
        self.tool_cache = SmartCache(max_size=200, default_ttl=tool_ttl)
        self.weak_tool_refs = WeakValueDictionary()  # Weak reference tool instances

        # Configuration hash cache for fast detection of configuration changes
        self.config_hashes: Dict[str, str] = {}

        logger.info(
            f"Config cache optimizer initialized (config_ttl={config_ttl}s, tool_ttl={tool_ttl}s)"
        )

    def get_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate configuration hash value"""
        # Create standardized JSON representation of configuration
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def cache_config(
        self, config_key: str, config: Dict[str, Any], processed_config: Any
    ) -> str:
        """Cache processed configuration"""
        config_hash = self.get_config_hash(config)
        cache_key = f"config_{config_key}_{config_hash}"

        self.config_cache.set(
            cache_key,
            processed_config,
            metadata={
                "config_key": config_key,
                "config_hash": config_hash,
                "original_config_size": len(str(config)),
            },
        )

        self.config_hashes[config_key] = config_hash
        logger.debug(f"Cached config: {config_key} (hash: {config_hash})")
        return cache_key

    def get_cached_config(
        self, config_key: str, config: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached configuration"""
        config_hash = self.get_config_hash(config)
        cache_key = f"config_{config_key}_{config_hash}"

        cached_config = self.config_cache.get(cache_key)
        if cached_config is not None:
            logger.debug(f"Config cache hit: {config_key}")
            return cached_config

        logger.debug(f"Config cache miss: {config_key}")
        return None

    def is_config_changed(self, config_key: str, config: Dict[str, Any]) -> bool:
        """Check if configuration has changed"""
        current_hash = self.get_config_hash(config)
        cached_hash = self.config_hashes.get(config_key)

        if cached_hash is None:
            return True  # No cache, consider as changed

        return current_hash != cached_hash

    def cache_tool(
        self, tool_key: str, tool_config: Dict[str, Any], tool_instance: Any
    ) -> str:
        """Cache tool instance"""
        config_hash = self.get_config_hash(tool_config)
        cache_key = f"tool_{tool_key}_{config_hash}"

        # Use weak references to store tool instances to avoid memory leaks
        self.weak_tool_refs[cache_key] = tool_instance

        # Store tool metadata in regular cache
        tool_metadata = {
            "tool_key": tool_key,
            "config_hash": config_hash,
            "tool_type": type(tool_instance).__name__,
            "created_at": datetime.now().isoformat(),
        }

        self.tool_cache.set(
            cache_key,
            tool_metadata,
            metadata={
                "tool_key": tool_key,
                "config_hash": config_hash,
            },
        )

        logger.debug(f"Cached tool: {tool_key} (hash: {config_hash})")
        return cache_key

    def get_cached_tool(
        self, tool_key: str, tool_config: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached tool instance"""
        config_hash = self.get_config_hash(tool_config)
        cache_key = f"tool_{tool_key}_{config_hash}"

        # First check weak references
        tool_instance = self.weak_tool_refs.get(cache_key)
        if tool_instance is not None:
            # Update cache access statistics
            self.tool_cache.get(cache_key)
            logger.debug(f"Tool cache hit: {tool_key}")
            return tool_instance

        # If not in weak references, clean up metadata in regular cache
        self.tool_cache.delete(cache_key)
        logger.debug(f"Tool cache miss: {tool_key}")
        return None

    def invalidate_config_cache(self, config_key: Optional[str] = None):
        """Invalidate configuration cache"""
        if config_key is None:
            # Clear all configuration cache
            self.config_cache.clear()
            self.config_hashes.clear()
            logger.info("All config cache invalidated")
        else:
            # Clear cache for specific configuration
            if config_key in self.config_hashes:
                config_hash = self.config_hashes[config_key]
                cache_key = f"config_{config_key}_{config_hash}"
                self.config_cache.delete(cache_key)
                del self.config_hashes[config_key]
                logger.debug(f"Config cache invalidated: {config_key}")

    def invalidate_tool_cache(self, tool_key: Optional[str] = None):
        """Invalidate tool cache"""
        if tool_key is None:
            # Clear all tool cache
            self.tool_cache.clear()
            self.weak_tool_refs.clear()
            logger.info("All tool cache invalidated")
        else:
            # Clear cache for specific tool
            keys_to_remove = []
            for cache_key in list(self.weak_tool_refs.keys()):
                if cache_key.startswith(f"tool_{tool_key}_"):
                    keys_to_remove.append(cache_key)

            for cache_key in keys_to_remove:
                self.weak_tool_refs.pop(cache_key, None)
                self.tool_cache.delete(cache_key)

            if keys_to_remove:
                logger.debug(
                    f"Tool cache invalidated: {tool_key} ({len(keys_to_remove)} entries)"
                )

    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired cache"""
        config_cleaned = self.config_cache.cleanup_expired()
        tool_cleaned = self.tool_cache.cleanup_expired()

        # Clean up invalid entries in weak references
        weak_cleaned = 0
        invalid_keys = []
        for key in list(self.weak_tool_refs.keys()):
            if self.weak_tool_refs.get(key) is None:
                invalid_keys.append(key)

        for key in invalid_keys:
            self.weak_tool_refs.pop(key, None)
            weak_cleaned += 1

        cleanup_stats = {
            "config_expired": config_cleaned,
            "tool_expired": tool_cleaned,
            "weak_refs_cleaned": weak_cleaned,
        }

        if sum(cleanup_stats.values()) > 0:
            logger.info(f"Cache cleanup completed: {cleanup_stats}")

        return cleanup_stats

    def get_cache_summary(self) -> Dict[str, Any]:
        """Get cache summary information"""
        return {
            "config_cache": self.config_cache.get_stats(),
            "tool_cache": self.tool_cache.get_stats(),
            "weak_tool_refs_count": len(self.weak_tool_refs),
            "config_hashes_count": len(self.config_hashes),
        }

    def optimize_cache_settings(self) -> Dict[str, Any]:
        """Optimize cache settings"""
        config_stats = self.config_cache.get_stats()
        tool_stats = self.tool_cache.get_stats()

        recommendations = []

        # Analyze configuration cache hit rate
        if config_stats["hit_rate"] < 0.5:
            recommendations.append(
                {
                    "type": "config_cache",
                    "issue": "low_hit_rate",
                    "current_rate": config_stats["hit_rate"],
                    "suggestion": (
                        "Consider increasing config cache TTL or reviewing config generation patterns"
                    ),
                }
            )

        # Analyze tool cache hit rate
        if tool_stats["hit_rate"] < 0.3:
            recommendations.append(
                {
                    "type": "tool_cache",
                    "issue": "low_hit_rate",
                    "current_rate": tool_stats["hit_rate"],
                    "suggestion": (
                        "Consider increasing tool cache TTL or optimizing tool creation patterns"
                    ),
                }
            )

        # Analyze cache size utilization
        config_utilization = config_stats["size"] / config_stats["max_size"]
        if config_utilization > 0.9:
            recommendations.append(
                {
                    "type": "config_cache",
                    "issue": "high_utilization",
                    "current_utilization": config_utilization,
                    "suggestion": "Consider increasing config cache max_size",
                }
            )

        tool_utilization = tool_stats["size"] / tool_stats["max_size"]
        if tool_utilization > 0.9:
            recommendations.append(
                {
                    "type": "tool_cache",
                    "issue": "high_utilization",
                    "current_utilization": tool_utilization,
                    "suggestion": "Consider increasing tool cache max_size",
                }
            )

        return {
            "cache_summary": self.get_cache_summary(),
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score(
                config_stats, tool_stats
            ),
        }

    def _calculate_optimization_score(
        self, config_stats: Dict, tool_stats: Dict
    ) -> float:
        """Calculate cache optimization score (0-100)"""
        # Calculate score based on hit rate and utilization
        config_score = config_stats["hit_rate"] * 50
        tool_score = tool_stats["hit_rate"] * 50

        # Consider cache utilization (moderate utilization is better)
        config_util = config_stats["size"] / config_stats["max_size"]
        tool_util = tool_stats["size"] / tool_stats["max_size"]

        # Ideal utilization is between 0.3-0.8
        config_util_score = max(0, 1 - abs(config_util - 0.55) / 0.45) * 25
        tool_util_score = max(0, 1 - abs(tool_util - 0.55) / 0.45) * 25

        total_score = (
            config_score + tool_score + config_util_score + tool_util_score
        ) / 1.5
        return min(100, max(0, total_score))


# Global configuration cache optimizer instance
global_config_cache_optimizer = ConfigCacheOptimizer()
