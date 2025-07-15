# -*- coding: utf-8 -*-
"""
Configuration Cache
Provides caching mechanisms for configuration data
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Set
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with timestamp and data"""
    data: Any
    timestamp: float
    ttl: Optional[float] = None  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class ConfigCache:
    """Thread-safe configuration cache with TTL support"""
    
    def __init__(self, default_ttl: Optional[float] = None):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._dependencies: Dict[str, Set[str]] = {}  # key -> set of dependent keys
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                logger.debug(f"Cache entry expired for key: {key}")
                del self._cache[key]
                return None
            
            return entry.data
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set cached value with optional TTL"""
        with self._lock:
            if ttl is None:
                ttl = self._default_ttl
            
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl
            )
            self._cache[key] = entry
            logger.debug(f"Cached value for key: {key} (TTL: {ttl})")
    
    def delete(self, key: str) -> bool:
        """Delete cached value by key"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Deleted cache entry for key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached values"""
        with self._lock:
            self._cache.clear()
            self._dependencies.clear()
            logger.debug("Cleared all cache entries")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern (simple prefix matching)"""
        with self._lock:
            keys_to_delete = [key for key in self._cache.keys() if key.startswith(pattern)]
            for key in keys_to_delete:
                del self._cache[key]
            
            logger.debug(f"Invalidated {len(keys_to_delete)} cache entries matching pattern: {pattern}")
            return len(keys_to_delete)
    
    def add_dependency(self, key: str, dependent_key: str) -> None:
        """Add dependency relationship between keys"""
        with self._lock:
            if key not in self._dependencies:
                self._dependencies[key] = set()
            self._dependencies[key].add(dependent_key)
    
    def invalidate_dependents(self, key: str) -> int:
        """Invalidate all keys that depend on the given key"""
        with self._lock:
            dependents = self._dependencies.get(key, set())
            count = 0
            
            for dependent_key in dependents:
                if self.delete(dependent_key):
                    count += 1
                # Recursively invalidate dependents of dependents
                count += self.invalidate_dependents(dependent_key)
            
            logger.debug(f"Invalidated {count} dependent cache entries for key: {key}")
            return count
    
    def cleanup_expired(self) -> int:
        """Remove all expired cache entries"""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'active_entries': total_entries - expired_entries,
                'dependencies': len(self._dependencies)
            }


# Global cache instance
_config_cache = ConfigCache(default_ttl=300)  # 5 minutes default TTL


def get_config_cache() -> ConfigCache:
    """Get global configuration cache instance"""
    return _config_cache


def cached_config(key: str, ttl: Optional[float] = None):
    """Decorator for caching configuration function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_config_cache()
            
            # Try to get from cache first
            cached_value = cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {key}")
                return cached_value
            
            # Cache miss, compute value
            logger.debug(f"Cache miss for key: {key}, computing value")
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator


def invalidate_config_cache(pattern: str = None, key: str = None) -> int:
    """Invalidate configuration cache entries"""
    cache = get_config_cache()
    
    if key:
        # Invalidate specific key and its dependents
        count = 1 if cache.delete(key) else 0
        count += cache.invalidate_dependents(key)
        return count
    elif pattern:
        # Invalidate by pattern
        return cache.invalidate_pattern(pattern)
    else:
        # Clear all cache
        cache.clear()
        return 0


def cleanup_config_cache() -> int:
    """Clean up expired configuration cache entries"""
    return get_config_cache().cleanup_expired()


def get_cache_stats() -> Dict[str, Any]:
    """Get configuration cache statistics"""
    return get_config_cache().get_stats()