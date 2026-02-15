# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict


def get_current_time_context() -> Dict[str, Any]:
    """Get the current time context with multiple format options.

    Returns:
        Dictionary containing various time format representations.
    """
    now = datetime.now(timezone.utc)
    return {
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M:%S"),
        "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "current_datetime_iso": now.isoformat(),
        "current_year": now.year,
        "current_month": now.month,
        "current_day": now.day,
        "current_hour": now.hour,
        "current_minute": now.minute,
        "current_weekday": now.strftime("%A"),
        "timezone": "UTC",
    }


def get_time_aware_search_suffix() -> str:
    """Generate a time-aware suffix for search queries.

    Returns:
        A string containing the current year for time-sensitive searches.
    """
    now = datetime.now(timezone.utc)
    return f"after:{now.year - 1}"


def with_time_context(func: Callable) -> Callable:
    """Decorator that adds time context to tool function calls.

    This decorator automatically injects current time information into
    the function's kwargs, allowing tools to make time-aware decisions.

    Args:
        func: The function to decorate.

    Returns:
        Wrapped function with time context injection.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "time_context" not in kwargs:
            kwargs["time_context"] = get_current_time_context()
        return func(*args, **kwargs)

    return wrapper


def is_time_sensitive_query(query: str) -> bool:
    """Check if a query is likely time-sensitive.

    Args:
        query: The search query to check.

    Returns:
        True if the query appears to be time-sensitive.
    """
    time_sensitive_keywords = [
        "latest",
        "recent",
        "current",
        "new",
        "today",
        "yesterday",
        "this week",
        "this month",
        "this year",
        "2024",
        "2025",
        "2026",
        "最新",
        "最近",
        "今日",
        "今年",
        "当前",
        "最新消息",
        "最新动态",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in time_sensitive_keywords)


def enhance_query_with_time(
    query: str, time_context: Dict[str, Any] | None = None
) -> str:
    """Enhance a search query with time context if appropriate.

    Args:
        query: The original search query.
        time_context: Optional time context dictionary. If not provided,
                      current time context will be used.

    Returns:
        Enhanced query with time context if the query is time-sensitive.
    """
    if not is_time_sensitive_query(query):
        return query

    if time_context is None:
        time_context = get_current_time_context()

    current_year = time_context.get("current_year", datetime.now(timezone.utc).year)

    if "after:" not in query.lower() and "before:" not in query.lower():
        return f"{query} after:{current_year - 1}"

    return query
