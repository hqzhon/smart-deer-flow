# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


class SystemMetrics(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cpu_usage: float = Field(description="CPU usage percentage")
    memory_usage: float = Field(description="Memory usage percentage")
    memory_used: str = Field(description="Memory used")
    memory_total: str = Field(description="Total memory")
    disk_usage: float = Field(description="Disk usage percentage")
    disk_used: str = Field(description="Disk used")
    disk_total: str = Field(description="Total disk")
    network_connections: int = Field(
        default=0, description="Active network connections"
    )
    process_count: int = Field(default=0, description="Number of processes")
    load_average: List[float] = Field(default_factory=list, description="Load average")


class LLMMetrics(BaseModel):
    total_requests: int = Field(default=0, description="Total LLM requests")
    total_tokens: int = Field(default=0, description="Total tokens used")
    input_tokens: int = Field(default=0, description="Input tokens")
    output_tokens: int = Field(default=0, description="Output tokens")
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    average_latency: float = Field(default=0.0, description="Average latency in ms")
    requests_per_minute: float = Field(default=0.0, description="Requests per minute")
    tokens_per_minute: float = Field(default=0.0, description="Tokens per minute")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    models_used: Dict[str, int] = Field(
        default_factory=dict, description="Usage by model"
    )


class ContextMetrics(BaseModel):
    total_sessions: int = Field(default=0, description="Total sessions")
    active_sessions: int = Field(default=0, description="Active sessions")
    total_messages: int = Field(default=0, description="Total messages")
    total_context_size: int = Field(
        default=0, description="Total context size in tokens"
    )
    average_context_size: float = Field(default=0.0, description="Average context size")
    compression_ratio: float = Field(default=0.0, description="Compression ratio")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate")


class PerformanceMetrics(BaseModel):
    request_queue_size: int = Field(default=0, description="Request queue size")
    active_connections: int = Field(default=0, description="Active connections")
    max_connections: int = Field(default=0, description="Max connections")
    connection_utilization: float = Field(
        default=0.0, description="Connection utilization"
    )
    average_response_time: float = Field(
        default=0.0, description="Average response time in ms"
    )
    p50_response_time: float = Field(default=0.0, description="P50 response time in ms")
    p95_response_time: float = Field(default=0.0, description="P95 response time in ms")
    p99_response_time: float = Field(default=0.0, description="P99 response time in ms")


class MetricsHistory(BaseModel):
    timestamps: List[datetime]
    cpu_usage: List[float]
    memory_usage: List[float]
    request_count: List[int]
    response_time: List[float]


class MetricsDashboard(BaseModel):
    system: SystemMetrics
    llm: LLMMetrics
    context: ContextMetrics
    performance: PerformanceMetrics
    uptime: int = Field(description="Server uptime in seconds")
    version: str = Field(default="0.1.0", description="Server version")


_metrics_store: Dict[str, List[Any]] = {
    "cpu_usage": [],
    "memory_usage": [],
    "request_count": [],
    "response_time": [],
    "timestamps": [],
}

_llm_stats: Dict[str, Any] = {
    "total_requests": 0,
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_cost": 0.0,
    "latencies": [],
    "errors": 0,
    "models": {},
    "start_time": time.time(),
}


@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics():
    """Get current system metrics."""
    import os

    try:
        import psutil

        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        try:
            network_connections = len(psutil.net_connections())
        except (psutil.AccessDenied, PermissionError):
            network_connections = 0

        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            memory_used=_format_bytes(memory.used),
            memory_total=_format_bytes(memory.total),
            disk_usage=disk.percent,
            disk_used=_format_bytes(disk.used),
            disk_total=_format_bytes(disk.total),
            network_connections=network_connections,
            process_count=len(psutil.pids()),
            load_average=list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        )
    except ImportError:
        return SystemMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            memory_used="0B",
            memory_total="0B",
            disk_usage=0.0,
            disk_used="0B",
            disk_total="0B",
        )


@router.get("/llm", response_model=LLMMetrics)
async def get_llm_metrics():
    """Get LLM usage metrics."""
    elapsed_time = time.time() - _llm_stats["start_time"]
    minutes = max(elapsed_time / 60, 1)

    total_requests = _llm_stats["total_requests"]
    errors = _llm_stats["errors"]
    error_rate = (errors / total_requests * 100) if total_requests > 0 else 0

    latencies = _llm_stats["latencies"]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return LLMMetrics(
        total_requests=total_requests,
        total_tokens=_llm_stats["total_tokens"],
        input_tokens=_llm_stats["input_tokens"],
        output_tokens=_llm_stats["output_tokens"],
        total_cost=_llm_stats["total_cost"],
        average_latency=avg_latency,
        requests_per_minute=total_requests / minutes,
        tokens_per_minute=_llm_stats["total_tokens"] / minutes,
        error_rate=error_rate,
        models_used=_llm_stats["models"],
    )


@router.get("/context", response_model=ContextMetrics)
async def get_context_metrics():
    """Get context engineering metrics."""
    try:
        from src.context import ResearchMemoryManager

        memory_manager = ResearchMemoryManager.get_instance()

        sessions = memory_manager.list_sessions()
        total_sessions = len(sessions)
        active_sessions = sum(1 for s in sessions if s.get("status") == "active")

        total_messages = 0
        total_context_size = 0
        compression_ratios = []
        cache_hits = 0
        cache_misses = 0

        for session_id in sessions:
            memory = memory_manager.get_memory(session_id.get("id", session_id))
            if memory:
                state = memory.export_state()
                total_messages += len(state.get("messages", []))
                total_context_size += state.get("total_tokens", 0)
                compression_ratios.append(state.get("compression_ratio", 0))
                cache_hits += state.get("cache_hits", 0)
                cache_misses += state.get("cache_misses", 0)

        avg_context_size = (
            total_context_size / total_sessions if total_sessions > 0 else 0
        )
        avg_compression = (
            sum(compression_ratios) / len(compression_ratios)
            if compression_ratios
            else 0
        )
        cache_total = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0

        return ContextMetrics(
            total_sessions=total_sessions,
            active_sessions=active_sessions,
            total_messages=total_messages,
            total_context_size=total_context_size,
            average_context_size=avg_context_size,
            compression_ratio=avg_compression,
            cache_hit_rate=cache_hit_rate,
        )
    except Exception as e:
        logger.debug(f"Failed to get context metrics: {e}")
        return ContextMetrics()


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get performance metrics."""
    from src.server.app import get_global_component

    conn_pool = get_global_component("connection_pool")

    queue_size = 0
    active_connections = 0
    max_connections = 0
    utilization = 0

    if conn_pool:
        active_connections = conn_pool.get("active_connections", 0)
        max_connections = conn_pool.get("max_connections", 1)
        utilization = (
            (active_connections / max_connections) * 100 if max_connections > 0 else 0
        )

    response_times = _metrics_store.get("response_time", [])
    avg_response = sum(response_times) / len(response_times) if response_times else 0

    sorted_times = sorted(response_times) if response_times else []
    p50 = sorted_times[int(len(sorted_times) * 0.5)] if sorted_times else 0
    p95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
    p99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0

    return PerformanceMetrics(
        request_queue_size=queue_size,
        active_connections=active_connections,
        max_connections=max_connections,
        connection_utilization=utilization,
        average_response_time=avg_response,
        p50_response_time=p50,
        p95_response_time=p95,
        p99_response_time=p99,
    )


@router.get("/history", response_model=MetricsHistory)
async def get_metrics_history(
    duration: int = Query(3600, ge=60, le=86400, description="Duration in seconds")
):
    """Get metrics history for the specified duration."""
    now = datetime.utcnow()
    cutoff = now.timestamp() - duration

    timestamps = []
    cpu_usage = []
    memory_usage = []
    request_count = []
    response_time = []

    stored_timestamps = _metrics_store.get("timestamps", [])
    for i, ts in enumerate(stored_timestamps):
        if ts.timestamp() >= cutoff:
            timestamps.append(ts)
            if i < len(_metrics_store.get("cpu_usage", [])):
                cpu_usage.append(_metrics_store["cpu_usage"][i])
            if i < len(_metrics_store.get("memory_usage", [])):
                memory_usage.append(_metrics_store["memory_usage"][i])
            if i < len(_metrics_store.get("request_count", [])):
                request_count.append(_metrics_store["request_count"][i])
            if i < len(_metrics_store.get("response_time", [])):
                response_time.append(_metrics_store["response_time"][i])

    return MetricsHistory(
        timestamps=timestamps,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        request_count=request_count,
        response_time=response_time,
    )


@router.get("/dashboard", response_model=MetricsDashboard)
async def get_metrics_dashboard():
    """Get comprehensive metrics dashboard."""
    system = await get_system_metrics()
    llm = await get_llm_metrics()
    context = await get_context_metrics()
    performance = await get_performance_metrics()

    uptime = int(time.time() - _llm_stats["start_time"])

    return MetricsDashboard(
        system=system,
        llm=llm,
        context=context,
        performance=performance,
        uptime=uptime,
    )


@router.post("/record")
async def record_metric(
    metric_type: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Record a custom metric."""
    now = datetime.utcnow()

    if metric_type == "response_time":
        _metrics_store["response_time"].append(value)
        _metrics_store["timestamps"].append(now)
        _trim_metrics_store()

    return {"status": "recorded", "metric_type": metric_type, "value": value}


@router.post("/llm/record")
async def record_llm_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency: float,
    cost: float = 0.0,
    error: bool = False,
):
    """Record LLM usage metrics."""
    _llm_stats["total_requests"] += 1
    _llm_stats["input_tokens"] += input_tokens
    _llm_stats["output_tokens"] += output_tokens
    _llm_stats["total_tokens"] += input_tokens + output_tokens
    _llm_stats["total_cost"] += cost
    _llm_stats["latencies"].append(latency)

    if len(_llm_stats["latencies"]) > 1000:
        _llm_stats["latencies"] = _llm_stats["latencies"][-1000:]

    if error:
        _llm_stats["errors"] += 1

    if model not in _llm_stats["models"]:
        _llm_stats["models"][model] = 0
    _llm_stats["models"][model] += 1

    return {"status": "recorded"}


def _trim_metrics_store(max_size: int = 1000):
    """Trim metrics store to max size."""
    for key in _metrics_store:
        if len(_metrics_store[key]) > max_size:
            _metrics_store[key] = _metrics_store[key][-max_size:]


def _format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"
