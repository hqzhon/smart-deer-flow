#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rate limiter
Used to control the frequency of parallel requests to avoid triggering API "Too Many Requests" errors
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    requests_per_minute: int = 60  # Maximum requests per minute
    requests_per_second: int = 10  # Maximum requests per second
    burst_size: int = 5  # Burst request size
    backoff_factor: float = 1.5  # Backoff factor
    max_backoff: float = 60.0  # Maximum backoff time (seconds)
    min_interval: float = 0.1  # Minimum request interval (seconds)


class TokenBucket:
    """Token bucket algorithm implementation"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity  # Bucket capacity
        self.tokens = capacity  # Current token count
        self.refill_rate = refill_rate  # Token refill rate (per second)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens"""
        async with self._lock:
            now = time.time()
            # Refill tokens
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: int = 1) -> float:
        """Wait until there are enough tokens, return wait time"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time needed
            needed_tokens = tokens - self.tokens
            wait_time = needed_tokens / self.refill_rate
            return wait_time


class RateLimiter:
    """Rate limiter"""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # Create token buckets
        self.per_second_bucket = TokenBucket(
            capacity=self.config.requests_per_second,
            refill_rate=self.config.requests_per_second,
        )

        self.per_minute_bucket = TokenBucket(
            capacity=self.config.requests_per_minute,
            refill_rate=self.config.requests_per_minute / 60.0,
        )

        # Request history
        self.request_history = deque(maxlen=1000)
        self.failure_count = 0
        self.last_failure_time = 0
        self.current_backoff = 0

        # Statistics
        self.total_requests = 0
        self.total_delays = 0
        self.total_delay_time = 0.0

        logger.info(
            f"Rate limiter initialized: {self.config.requests_per_minute}/min, {self.config.requests_per_second}/sec"
        )

    async def acquire(self) -> float:
        """Acquire request permission, return wait time"""
        start_time = time.time()

        # Check if in backoff period
        if self.current_backoff > 0:
            backoff_remaining = self.current_backoff - (
                time.time() - self.last_failure_time
            )
            if backoff_remaining > 0:
                logger.info(
                    f"Rate limiter in backoff, waiting {backoff_remaining:.2f}s"
                )
                await asyncio.sleep(backoff_remaining)
                self.total_delay_time += backoff_remaining
                self.total_delays += 1

        # Wait for token buckets
        wait_time_per_second = await self.per_second_bucket.wait_for_tokens()
        wait_time_per_minute = await self.per_minute_bucket.wait_for_tokens()

        max_wait_time = max(
            wait_time_per_second, wait_time_per_minute, self.config.min_interval
        )

        if max_wait_time > 0:
            logger.debug(f"Rate limiter waiting {max_wait_time:.2f}s")
            await asyncio.sleep(max_wait_time)
            self.total_delay_time += max_wait_time
            self.total_delays += 1

        # Record request
        self.request_history.append(time.time())
        self.total_requests += 1

        total_wait_time = time.time() - start_time
        return total_wait_time

    def record_success(self):
        """Record successful request"""
        # Reset failure count and backoff
        if self.failure_count > 0:
            logger.info(
                f"Request succeeded, resetting failure count from {self.failure_count}"
            )
            self.failure_count = 0
            self.current_backoff = 0

    def record_failure(self, error_message: str = ""):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        # Calculate backoff time
        self.current_backoff = min(
            self.config.max_backoff, (self.config.backoff_factor**self.failure_count)
        )

        logger.warning(
            f"Rate limit failure #{self.failure_count}, "
            f"backoff: {self.current_backoff:.2f}s, "
            f"error: {error_message}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        now = time.time()
        recent_requests = sum(1 for t in self.request_history if now - t <= 60)

        # Ensure type-safe average delay calculation
        try:
            average_delay = float(self.total_delay_time) / max(
                1, int(self.total_delays)
            )
        except (TypeError, ValueError, ZeroDivisionError):
            average_delay = 0.0

        return {
            "total_requests": int(self.total_requests),
            "total_delays": int(self.total_delays),
            "total_delay_time": float(self.total_delay_time),
            "average_delay": average_delay,
            "recent_requests_per_minute": int(recent_requests),
            "failure_count": int(self.failure_count),
            "current_backoff": float(self.current_backoff),
            "per_second_tokens": float(self.per_second_bucket.tokens),
            "per_minute_tokens": float(self.per_minute_bucket.tokens),
        }

    def adjust_limits(
        self,
        requests_per_minute: Optional[int] = None,
        requests_per_second: Optional[int] = None,
    ):
        """Dynamically adjust rate limits"""
        if requests_per_minute is not None:
            self.config.requests_per_minute = requests_per_minute
            self.per_minute_bucket = TokenBucket(
                capacity=requests_per_minute, refill_rate=requests_per_minute / 60.0
            )
            logger.info(f"Adjusted rate limit to {requests_per_minute}/min")

        if requests_per_second is not None:
            self.config.requests_per_second = requests_per_second
            self.per_second_bucket = TokenBucket(
                capacity=requests_per_second, refill_rate=requests_per_second
            )
            logger.info(f"Adjusted rate limit to {requests_per_second}/sec")


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter"""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        super().__init__(config)
        self.success_streak = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 30  # Adjustment interval (seconds)
        self.max_requests_per_second = self.config.requests_per_second * 2
        self.min_requests_per_second = max(1, self.config.requests_per_second // 2)

    def record_success(self):
        """Record successful request and adaptive adjustment"""
        super().record_success()
        self.success_streak += 1

        # If consecutive successes and time since last adjustment exceeds interval, try to increase rate
        now = time.time()
        if (
            self.success_streak >= 10
            and now - self.last_adjustment > self.adjustment_interval
            and self.config.requests_per_second < self.max_requests_per_second
        ):

            new_rate = min(
                self.max_requests_per_second, int(self.config.requests_per_second * 1.2)
            )
            self.adjust_limits(requests_per_second=new_rate)
            self.last_adjustment = now
            self.success_streak = 0
            logger.info(f"Adaptive rate limiter increased rate to {new_rate}/sec")

    def record_failure(self, error_message: str = ""):
        """Record failed request and adaptive adjustment"""
        super().record_failure(error_message)
        self.success_streak = 0

        # If it's a rate limit error, reduce request rate
        if (
            "rate limit" in error_message.lower()
            or "too many requests" in error_message.lower()
        ):
            new_rate = max(
                self.min_requests_per_second, int(self.config.requests_per_second * 0.7)
            )
            self.adjust_limits(requests_per_second=new_rate)
            self.last_adjustment = time.time()
            logger.info(
                f"Adaptive rate limiter decreased rate to {new_rate}/sec due to rate limit error"
            )


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_global_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get global rate limiter instance"""
    global _global_rate_limiter

    if _global_rate_limiter is None:
        if config is None:
            # Try to load from config
            try:
                from src.config import get_settings

                settings = get_settings()

                # Extract rate limiting parameters from config
                config = RateLimitConfig(
                    requests_per_minute=getattr(settings, "requests_per_minute", 60),
                    requests_per_second=getattr(settings, "requests_per_second", 10),
                    burst_size=getattr(settings, "burst_size", 5),
                    backoff_factor=getattr(settings, "backoff_factor", 1.5),
                    max_backoff=getattr(settings, "max_backoff", 60.0),
                    min_interval=getattr(settings, "min_interval", 0.1),
                )
            except Exception as e:
                logger.warning(f"Failed to load rate limit config: {e}, using defaults")
                config = RateLimitConfig()

        # Choose rate limiter type based on configuration
        try:
            from src.config import get_settings

            settings = get_settings()
            adaptive = getattr(settings, "adaptive_rate_limiting", True)
            if adaptive:
                _global_rate_limiter = AdaptiveRateLimiter(config)
            else:
                _global_rate_limiter = RateLimiter(config)
        except Exception:
            _global_rate_limiter = AdaptiveRateLimiter(config)
    elif config is not None:
        # If new configuration is provided, recreate instance
        try:
            from src.config import get_settings

            settings = get_settings()
            adaptive = getattr(settings, "adaptive_rate_limiting", True)
            if adaptive:
                _global_rate_limiter = AdaptiveRateLimiter(config)
            else:
                _global_rate_limiter = RateLimiter(config)
        except Exception:
            _global_rate_limiter = AdaptiveRateLimiter(config)

    return _global_rate_limiter


def reset_global_rate_limiter():
    """Reset global rate limiter"""
    global _global_rate_limiter
    _global_rate_limiter = None
