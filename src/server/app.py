# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import base64
import json
import logging
import os
import time
import asyncio
from typing import Annotated, List, cast, Dict, Any
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage, BaseMessage
from langgraph.types import Command
from src.utils.decorators import safe_background_task

from src.config.report_style import ReportStyle
from src.config.tools import SELECTED_RAG_PROVIDER
from src.graph.builder import build_graph_with_memory
from src.utils.performance_optimizer import (
    AdvancedParallelExecutor,
    AdaptiveRateLimiter,
    SmartErrorRecovery,
    TaskPriority,
)
from src.utils.memory_manager import HierarchicalMemoryManager
from src.config.performance_config import (
    get_performance_config,
    update_performance_config,
)
from src.podcast.graph.builder import build_graph as build_podcast_graph
from src.ppt.graph.builder import build_graph as build_ppt_graph
from src.prose.graph.builder import build_graph as build_prose_graph
from src.prompt_enhancer.graph.builder import build_graph as build_prompt_enhancer_graph
from src.rag.builder import build_retriever
from src.rag.retriever import Resource
from src.server.chat_request import (
    ChatRequest,
    EnhancePromptRequest,
    GeneratePodcastRequest,
    GeneratePPTRequest,
    GenerateProseRequest,
    TTSRequest,
)
from src.server.mcp_request import MCPServerMetadataRequest, MCPServerMetadataResponse
from src.server.mcp_utils import load_mcp_tools
from src.server.rag_request import (
    RAGConfigResponse,
    RAGResourceRequest,
    RAGResourcesResponse,
)
from src.server.config_request import ConfigResponse
from src.llms.llm import get_configured_llm_models
from src.tools import VolcengineTTS

logger = logging.getLogger(__name__)

INTERNAL_SERVER_ERROR_DETAIL = "Internal Server Error"

# Global variables for advanced optimization with thread safety
import threading

# Thread-safe global state management
_global_state_lock = threading.RLock()
_global_state = {
    "connection_pool": None,
    "request_queue": None,
    "batch_processor": None,
    "advanced_parallel_executor": None,
    "adaptive_rate_limiter": None,
    "smart_error_recovery": None,
    "hierarchical_memory": None,
}


def get_global_component(name: str):
    """Thread-safe getter for global components"""
    with _global_state_lock:
        return _global_state.get(name)


def set_global_component(name: str, value):
    """Thread-safe setter for global components"""
    with _global_state_lock:
        _global_state[name] = value


# Initialize request queue
request_queue = asyncio.Queue(maxsize=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with configurable performance optimization."""

    # Get performance configuration
    config = get_performance_config()

    # Startup
    logger.info("Starting DeerFlow API server with configurable optimizations...")
    logger.info(f"Advanced optimization: {config.enable_advanced_optimization}")
    logger.info(f"Collaboration: {config.enable_collaboration}")

    # Initialize advanced performance components based on configuration
    if config.enable_advanced_optimization:
        set_global_component(
            "advanced_parallel_executor",
            AdvancedParallelExecutor(
                max_workers=config.parallel_execution.max_workers,
                enable_metrics=config.monitoring.metrics_enabled,
            ),
        )
        set_global_component(
            "adaptive_rate_limiter",
            AdaptiveRateLimiter(
                initial_rate=config.rate_limit.initial_rate,
                time_window=config.rate_limit.time_window,
            ),
        )
        set_global_component(
            "smart_error_recovery",
            SmartErrorRecovery(
                max_retries=config.error_recovery.max_retries,
                base_delay=config.error_recovery.base_delay,
                max_delay=config.error_recovery.max_delay,
            ),
        )
        set_global_component(
            "hierarchical_memory",
            HierarchicalMemoryManager(
                l1_max_size=config.cache.l1_size * 1024 * 1024,  # Convert MB to bytes
                l2_max_size=config.cache.l2_size * 1024 * 1024,  # Convert MB to bytes
                l3_max_size=config.cache.l3_size * 1024 * 1024,  # Convert MB to bytes
            ),
        )

        # Start advanced components
        await get_global_component("advanced_parallel_executor").start()
        await get_global_component("hierarchical_memory").start()

        logger.info("Advanced optimization components initialized")
    else:
        logger.info("Running in basic optimization mode")

    # Initialize enhanced connection pool with configuration
    set_global_component(
        "connection_pool",
        {
            "max_connections": config.connection_pool.max_connections,
            "active_connections": 0,
            "semaphore": asyncio.Semaphore(config.connection_pool.max_connections),
            "connection_metrics": {
                "total_acquired": 0,
                "total_released": 0,
                "peak_usage": 0,
            },
        },
    )

    # Start batch processor (enhanced or basic based on config)
    if config.enable_advanced_optimization:
        set_global_component(
            "batch_processor", asyncio.create_task(_enhanced_batch_processor())
        )
    else:
        set_global_component(
            "batch_processor", asyncio.create_task(_basic_batch_processor())
        )

    logger.info(
        "DeerFlow API server started successfully with configurable optimizations"
    )
    yield

    # Shutdown
    logger.info("Shutting down DeerFlow API server...")

    # Stop components
    batch_proc = get_global_component("batch_processor")
    if batch_proc:
        batch_proc.cancel()
        try:
            await batch_proc
        except asyncio.CancelledError:
            pass

    if config.enable_advanced_optimization:
        adv_executor = get_global_component("advanced_parallel_executor")
        if adv_executor:
            await adv_executor.stop()
        hier_memory = get_global_component("hierarchical_memory")
        if hier_memory:
            await hier_memory.stop()

    # Log final metrics if enabled
    conn_pool = get_global_component("connection_pool")
    if conn_pool and config.monitoring.metrics_enabled:
        metrics = conn_pool["connection_metrics"]
        logger.info(
            f"Connection pool metrics - Acquired: {metrics['total_acquired']}, "
            f"Released: {metrics['total_released']}, Peak: {metrics['peak_usage']}"
        )

    logger.info("DeerFlow API server shutdown complete")


async def _process_enhanced_batch(batch: List[Dict[str, Any]]):
    """Process a batch of requests with enhanced optimization."""
    if not batch:
        return

    logger.info(f"Processing enhanced batch of {len(batch)} requests")

    # Use hierarchical memory for caching if available
    hier_memory = get_global_component("hierarchical_memory")
    if hier_memory:
        # Check cache for similar requests
        cached_results = await hier_memory.get_batch(batch)
        uncached_batch = [
            req for req, result in zip(batch, cached_results) if result is None
        ]

        if uncached_batch:
            # Process uncached requests with smart error recovery
            error_recovery = get_global_component("smart_error_recovery")
            if error_recovery:
                tasks = []
                for request_data in uncached_batch:
                    task = error_recovery.execute_with_retry(
                        _process_single_request, request_data
                    )
                    tasks.append(task)

                # Wait for all tasks with adaptive rate limiting
                rate_limiter = get_global_component("adaptive_rate_limiter")
                if rate_limiter:
                    await rate_limiter.acquire_batch(len(tasks))

                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Fallback to regular processing
                await _process_batch(uncached_batch)
    else:
        # Fallback to regular batch processing
        await _process_batch(batch)


app = FastAPI(
    title="DeerFlow API",
    description="API for Deer",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

graph = build_graph_with_memory()


async def _enhanced_batch_processor():
    """Enhanced background batch processor with advanced optimization."""
    config = get_performance_config()
    batch_size = config.batch_processing.batch_size
    batch_timeout = config.batch_processing.batch_timeout

    while True:
        try:
            batch = []
            start_time = time.time()

            # Collect requests for batch processing with priority handling
            while (
                len(batch) < batch_size and (time.time() - start_time) < batch_timeout
            ):
                try:
                    request_data = await asyncio.wait_for(
                        request_queue.get(), timeout=0.05
                    )
                    batch.append(request_data)
                except asyncio.TimeoutError:
                    continue

            # Process batch if we have requests
            if batch:
                # Submit batch processing to advanced parallel executor
                adv_executor = get_global_component("advanced_parallel_executor")
                if adv_executor:
                    await adv_executor.submit_task(
                        _process_enhanced_batch, batch, priority=TaskPriority.HIGH
                    )
                else:
                    # Fallback to regular batch processing
                    await _process_batch(batch)

            # Adaptive delay based on queue size
            queue_size = request_queue.qsize()
            if queue_size > 50:
                await asyncio.sleep(0.001)  # Very short delay when busy
            elif queue_size > 10:
                await asyncio.sleep(0.005)  # Short delay when moderately busy
            else:
                await asyncio.sleep(0.01)  # Normal delay when not busy

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in enhanced batch processor: {e}")
            await asyncio.sleep(1.0)


async def _basic_batch_processor():
    """Basic batch processor for when advanced optimization is disabled."""
    config = get_performance_config()
    batch_size = min(
        config.batch_processing.batch_size, 5
    )  # Smaller batch size for basic mode
    batch_timeout = (
        config.batch_processing.batch_timeout * 2
    )  # Longer timeout for basic mode

    while True:
        try:
            batch = []
            start_time = time.time()

            # Collect requests for batch processing
            while (
                len(batch) < batch_size and (time.time() - start_time) < batch_timeout
            ):
                try:
                    request_data = await asyncio.wait_for(
                        request_queue.get(), timeout=0.1
                    )
                    batch.append(request_data)
                except asyncio.TimeoutError:
                    continue

            # Process batch if we have requests
            if batch:
                await _process_batch(batch)

            # Fixed delay for basic mode
            await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in basic batch processor: {e}")
            await asyncio.sleep(1.0)


async def _process_batch(batch: List[Dict[str, Any]]):
    """Process a batch of requests concurrently."""
    if not batch:
        return

    logger.info(f"Processing batch of {len(batch)} requests")

    # Process requests concurrently
    tasks = []
    for request_data in batch:
        task = asyncio.create_task(_process_single_request(request_data))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)


async def _process_single_request(request_data: Dict[str, Any]):
    """Process a single request from the batch."""
    try:
        # This would be implemented based on the specific request type
        # For now, just log the processing
        logger.debug(f"Processing request: {request_data.get('id', 'unknown')}")
        await asyncio.sleep(0.1)  # Simulate processing
    except Exception as e:
        logger.error(
            f"Error processing request {request_data.get('id', 'unknown')}: {e}"
        )


async def _acquire_connection():
    """Acquire a connection from the enhanced pool with metrics tracking."""
    conn_pool = get_global_component("connection_pool")
    if conn_pool:
        await conn_pool["semaphore"].acquire()
        conn_pool["active_connections"] += 1
        conn_pool["connection_metrics"]["total_acquired"] += 1

        # Update peak usage
        current_usage = conn_pool["active_connections"]
        if current_usage > conn_pool["connection_metrics"]["peak_usage"]:
            conn_pool["connection_metrics"]["peak_usage"] = current_usage

        logger.debug(f"Acquired connection (active: {current_usage})")


@safe_background_task
async def _release_connection():
    """Release a connection back to the enhanced pool with metrics tracking.

    This function is decorated with @safe_background_task to ensure any exceptions
    are properly caught and logged without causing silent failures.
    """
    conn_pool = get_global_component("connection_pool")
    if not conn_pool:
        logger.warning("Connection pool not available for release")
        return

    try:
        # Validate connection pool state before modification
        if conn_pool["active_connections"] <= 0:
            logger.warning("Attempting to release connection when none are active")
            return

        conn_pool["active_connections"] -= 1
        conn_pool["connection_metrics"]["total_released"] += 1
        conn_pool["semaphore"].release()

        current_usage = conn_pool["active_connections"]
        logger.debug(f"Released connection (active: {current_usage})")

        # Log warning if connection pool is getting full
        max_connections = conn_pool.get("max_connections", 1)
        if max_connections > 0:
            utilization = current_usage / max_connections
            if utilization > 0.8:
                logger.warning(f"High connection pool utilization: {utilization:.1%}")
    except KeyError as e:
        logger.error(f"Missing key in connection pool: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in _release_connection: {e}", exc_info=True)


async def _get_connection_metrics() -> Dict[str, Any]:
    """Get current connection pool metrics."""
    conn_pool = get_global_component("connection_pool")
    if conn_pool:
        metrics = conn_pool["connection_metrics"]
        return {
            "active_connections": conn_pool["active_connections"],
            "max_connections": conn_pool["max_connections"],
            "utilization": conn_pool["active_connections"]
            / conn_pool["max_connections"],
            "total_acquired": metrics["total_acquired"],
            "total_released": metrics["total_released"],
            "peak_usage": metrics["peak_usage"],
            "queue_size": request_queue.qsize(),
        }
    return {"status": "connection_pool_not_initialized"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    connection_acquired = False

    try:
        # Acquire connection from enhanced pool
        await _acquire_connection()
        connection_acquired = True

        # Check adaptive rate limiting
        rate_limiter = get_global_component("adaptive_rate_limiter")
        if rate_limiter:
            await rate_limiter.acquire()

        thread_id = request.thread_id
        if thread_id == "__default__":
            thread_id = str(uuid4())

        # Log request metrics with enhanced details
        logger.info(f"Starting enhanced chat stream for thread: {thread_id}")

        # Add cleanup task with enhanced monitoring
        background_tasks.add_task(_release_connection)
        background_tasks.add_task(_log_request_completion, thread_id, start_time)

        return StreamingResponse(
            _astream_workflow_generator(
                request.model_dump()["messages"],
                thread_id,
                request.resources,
                request.max_plan_iterations,
                request.max_step_num,
                request.max_search_results,
                request.auto_accepted_plan,
                request.interrupt_feedback,
                request.mcp_settings,
                request.enable_background_investigation,
                request.report_style,
                request.enable_deep_thinking,
                request.enable_collaboration,
                request.enable_parallel_execution,
                request.max_parallel_tasks,
            ),
            media_type="text/event-stream",
            headers={
                "X-Connection-ID": str(id(thread_id)),
                "X-Advanced-Optimization": (
                    "enabled"
                    if get_global_component("hierarchical_memory")
                    else "disabled"
                ),
            },
        )
    except Exception as e:
        # 确保在异常情况下释放连接
        if connection_acquired:
            try:
                await _release_connection()
            except Exception as release_error:
                logger.error(f"Error releasing connection: {release_error}")
        logger.error(f"Error in enhanced chat_stream: {e}")
        raise


@safe_background_task
async def _log_enhanced_request_completion(
    connection_id: str, start_time: float, message_length: int = 0
):
    """Log enhanced request completion metrics with detailed performance data.

    This function is decorated with @safe_background_task to ensure any exceptions
    are properly caught and logged without causing silent failures.
    """
    try:
        # Validate input parameters
        if not connection_id:
            logger.warning(
                "Empty connection_id provided to _log_enhanced_request_completion"
            )
            return

        if not isinstance(start_time, (int, float)) or start_time <= 0:
            logger.warning(f"Invalid start_time provided: {start_time}")
            return

        if not isinstance(message_length, int) or message_length < 0:
            logger.warning(f"Invalid message_length provided: {message_length}")
            message_length = 0

        completion_time = time.time()
        if completion_time < start_time:
            logger.warning(
                f"Completion time {completion_time} is before start_time {start_time}"
            )
            return

        processing_time = completion_time - start_time

        # Get current system metrics with error handling
        try:
            connection_metrics = await _get_connection_metrics()
        except Exception as e:
            logger.warning(f"Failed to get connection metrics: {e}")
            connection_metrics = {"utilization": 0}

        # Calculate throughput metrics
        chars_per_second = (
            message_length / processing_time if processing_time > 0 else 0
        )

        # Log comprehensive metrics
        logger.info(
            f"Enhanced request completed - Connection: {connection_id}, "
            f"Processing time: {processing_time:.2f}s, "
            f"Message length: {message_length} chars, "
            f"Throughput: {chars_per_second:.1f} chars/s, "
            f"Pool utilization: {connection_metrics.get('utilization', 0):.1%}"
        )

        # Log performance warnings if needed
        if processing_time > 10.0:
            logger.warning(f"Slow request detected: {processing_time:.2f}s")

        utilization = connection_metrics.get("utilization", 0)
        if isinstance(utilization, (int, float)) and utilization > 0.9:
            logger.warning("High connection pool utilization detected")

        # Release connection with additional error handling
        try:
            await _release_connection()
        except Exception as e:
            logger.error(f"Failed to release connection in enhanced completion: {e}")

    except Exception as e:
        logger.error(
            f"Unexpected error in _log_enhanced_request_completion: {e}", exc_info=True
        )


@safe_background_task
async def _log_request_completion(thread_id: str, start_time: float):
    """Legacy request completion logging for backward compatibility.

    This function is decorated with @safe_background_task to ensure any exceptions
    are properly caught and logged without causing silent failures.
    """
    try:
        # Validate input parameters
        if not thread_id:
            logger.warning("Empty thread_id provided to _log_request_completion")
            return

        if not isinstance(start_time, (int, float)) or start_time <= 0:
            logger.warning(f"Invalid start_time provided: {start_time}")
            return

        current_time = time.time()
        if current_time < start_time:
            logger.warning(
                f"Current time {current_time} is before start_time {start_time}"
            )
            return

        execution_time = current_time - start_time
        logger.info(
            f"Chat stream completed for thread {thread_id} in {execution_time:.2f} seconds"
        )

        # Log performance warnings for slow requests
        if execution_time > 30.0:
            logger.warning(
                f"Slow request detected for thread {thread_id}: {execution_time:.2f}s"
            )

    except Exception as e:
        logger.error(f"Unexpected error in _log_request_completion: {e}", exc_info=True)


async def _astream_workflow_generator(
    messages: List[dict],
    thread_id: str,
    resources: List[Resource],
    max_plan_iterations: int,
    max_step_num: int,
    max_search_results: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation: bool,
    report_style: ReportStyle,
    enable_deep_thinking: bool,
    enable_collaboration: bool,
    enable_parallel_execution: bool,
    max_parallel_tasks: int,
):
    import asyncio
    from src.llms.error_handler import LLMErrorHandler

    error_handler = LLMErrorHandler()
    max_retries = 3

    input_ = {
        "messages": messages,
        "plan_iterations": 0,
        "final_report": "",
        "current_plan": None,
        "observations": [],
        "auto_accepted_plan": auto_accepted_plan,
        "enable_background_investigation": enable_background_investigation,
        "research_topic": messages[-1]["content"] if messages else "",
    }
    if not auto_accepted_plan and interrupt_feedback:
        resume_msg = f"[{interrupt_feedback}]"
        # Add the last message to the resume message
        if messages:
            resume_msg += f" {messages[-1]['content']}"
        input_ = Command(resume=resume_msg)

    # Retry mechanism to handle Rate Limiting errors
    for attempt in range(max_retries + 1):
        try:
            async for agent, _, event_data in graph.astream(
                input_,
                config={
                    "thread_id": thread_id,
                    "resources": resources,
                    "max_plan_iterations": max_plan_iterations,
                    "max_step_num": max_step_num,
                    "max_search_results": max_search_results,
                    "mcp_settings": mcp_settings,
                    "report_style": report_style.value,
                    "enable_deep_thinking": enable_deep_thinking,
                    "enable_collaboration": enable_collaboration,
                    "enable_parallel_execution": enable_parallel_execution,
                    "max_parallel_tasks": max_parallel_tasks,
                },
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                if isinstance(event_data, dict):
                    if "__interrupt__" in event_data:
                        yield _make_event(
                            "interrupt",
                            {
                                "thread_id": thread_id,
                                "id": event_data["__interrupt__"][0].ns[0],
                                "role": "assistant",
                                "content": event_data["__interrupt__"][0].value,
                                "finish_reason": "interrupt",
                                "options": [
                                    {"text": "Edit plan", "value": "edit_plan"},
                                    {"text": "Start research", "value": "accepted"},
                                ],
                            },
                        )
                    continue
                message_chunk, message_metadata = cast(
                    tuple[BaseMessage, dict[str, any]], event_data
                )
                # Only use name field for concurrent messages (with name field and not ToolMessage)
                # Other messages use original agent logic
                if (
                    not isinstance(message_chunk, ToolMessage)
                    and hasattr(message_chunk, "name")
                    and message_chunk.name
                ):
                    message_agent = message_chunk.name
                else:
                    message_agent = agent[0].split(":")[0]
                event_stream_message: dict[str, any] = {
                    "thread_id": thread_id,
                    "agent": message_agent,
                    "id": message_chunk.id,
                    "role": "assistant",
                    "content": message_chunk.content,
                }
                if message_chunk.additional_kwargs.get("reasoning_content"):
                    event_stream_message["reasoning_content"] = (
                        message_chunk.additional_kwargs["reasoning_content"]
                    )
                if message_chunk.response_metadata.get("finish_reason"):
                    event_stream_message["finish_reason"] = (
                        message_chunk.response_metadata.get("finish_reason")
                    )
                if isinstance(message_chunk, ToolMessage):
                    # Tool Message - Return the result of the tool call
                    event_stream_message["tool_call_id"] = message_chunk.tool_call_id
                    yield _make_event("tool_call_result", event_stream_message)
                elif isinstance(message_chunk, AIMessageChunk):
                    # AI Message - Raw message tokens
                    if message_chunk.tool_calls:
                        # AI Message - Tool Call
                        event_stream_message["tool_calls"] = message_chunk.tool_calls
                        event_stream_message["tool_call_chunks"] = (
                            message_chunk.tool_call_chunks
                        )
                        yield _make_event("tool_calls", event_stream_message)
                    elif message_chunk.tool_call_chunks:
                        # AI Message - Tool Call Chunks
                        event_stream_message["tool_call_chunks"] = (
                            message_chunk.tool_call_chunks
                        )
                        yield _make_event("tool_call_chunks", event_stream_message)
                    else:
                        # AI Message - Raw message tokens
                        yield _make_event("message_chunk", event_stream_message)
            # Successfully completed, break out of retry loop
            return
        except Exception as e:
            error_message = str(e)
            error_type = error_handler.classify_error(error_message)

            # Check if it's a Rate Limiting error
            if error_handler.should_retry_error(error_type) and attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                logger.warning(
                    f"Rate limiting detected (attempt {attempt + 1}/{max_retries + 1}): {error_message}"
                )
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                # If it's a Rate Limiting error but max retries reached, log error but don't send to frontend
                if error_handler.should_retry_error(error_type):
                    logger.error(
                        f"Rate limiting error after {max_retries} retries, giving up: {error_message}"
                    )
                    # Send a generic error message to frontend without exposing Rate Limiting details
                    yield _make_event(
                        "error",
                        {
                            "thread_id": thread_id,
                            "error": "Service temporarily unavailable, please try again later.",
                        },
                    )
                    return
                else:
                    # Other types of errors, log with proper error classification
                    logger.error(
                        f"Non-retryable error in workflow (type: {error_type}): {error_message}"
                    )
                    # Send appropriate error message based on error type
                    if error_type in ["AUTHENTICATION_ERROR", "PERMISSION_ERROR"]:
                        yield _make_event(
                            "error",
                            {
                                "thread_id": thread_id,
                                "error": "Authentication or permission error occurred.",
                            },
                        )
                    elif error_type in ["NETWORK_ERROR", "TIMEOUT_ERROR"]:
                        yield _make_event(
                            "error",
                            {
                                "thread_id": thread_id,
                                "error": "Network or timeout error occurred. Please try again.",
                            },
                        )
                    else:
                        yield _make_event(
                            "error",
                            {
                                "thread_id": thread_id,
                                "error": "An unexpected error occurred. Please try again.",
                            },
                        )
                    return


def _make_event(event_type: str, data: dict[str, any]):
    if data.get("content") == "":
        data.pop("content")
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using volcengine TTS API."""
    app_id = os.getenv("VOLCENGINE_TTS_APPID", "")
    if not app_id:
        raise HTTPException(status_code=400, detail="VOLCENGINE_TTS_APPID is not set")
    access_token = os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN", "")
    if not access_token:
        raise HTTPException(
            status_code=400, detail="VOLCENGINE_TTS_ACCESS_TOKEN is not set"
        )

    try:
        cluster = os.getenv("VOLCENGINE_TTS_CLUSTER", "volcano_tts")
        voice_type = os.getenv("VOLCENGINE_TTS_VOICE_TYPE", "BV700_V2_streaming")

        tts_client = VolcengineTTS(
            appid=app_id,
            access_token=access_token,
            cluster=cluster,
            voice_type=voice_type,
        )
        # Call the TTS API
        result = tts_client.text_to_speech(
            text=request.text[:1024],
            encoding=request.encoding,
            speed_ratio=request.speed_ratio,
            volume_ratio=request.volume_ratio,
            pitch_ratio=request.pitch_ratio,
            text_type=request.text_type,
            with_frontend=request.with_frontend,
            frontend_type=request.frontend_type,
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=str(result["error"]))

        # Decode the base64 audio data
        audio_data = base64.b64decode(result["audio_data"])

        # Return the audio file
        return Response(
            content=audio_data,
            media_type=f"audio/{request.encoding}",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=tts_output.{request.encoding}"
                )
            },
        )

    except Exception as e:
        logger.exception(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/podcast/generate")
async def generate_podcast(request: GeneratePodcastRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_podcast_graph()
        final_state = workflow.invoke({"input": report_content})
        audio_bytes = final_state["output"]
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        logger.exception(f"Error occurred during podcast generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/ppt/generate")
async def generate_ppt(request: GeneratePPTRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_ppt_graph()
        final_state = workflow.invoke({"input": report_content})
        generated_file_path = final_state["generated_file_path"]
        with open(generated_file_path, "rb") as f:
            ppt_bytes = f.read()
        return Response(
            content=ppt_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    except Exception as e:
        logger.exception(f"Error occurred during ppt generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prose/generate")
async def generate_prose(request: GenerateProseRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Generating prose for prompt: {sanitized_prompt}")
        workflow = build_prose_graph()
        events = workflow.astream(
            {
                "content": request.prompt,
                "option": request.option,
                "command": request.command,
            },
            stream_mode="messages",
            subgraphs=True,
        )
        return StreamingResponse(
            (f"data: {event[0].content}\n\n" async for _, event in events),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(f"Error occurred during prose generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prompt/enhance")
async def enhance_prompt(request: EnhancePromptRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Enhancing prompt: {sanitized_prompt}")

        # Convert string report_style to ReportStyle enum
        report_style = None
        if request.report_style:
            try:
                # Handle both uppercase and lowercase input
                style_mapping = {
                    "ACADEMIC": ReportStyle.ACADEMIC,
                    "POPULAR_SCIENCE": ReportStyle.POPULAR_SCIENCE,
                    "NEWS": ReportStyle.NEWS,
                    "SOCIAL_MEDIA": ReportStyle.SOCIAL_MEDIA,
                    "academic": ReportStyle.ACADEMIC,
                    "popular_science": ReportStyle.POPULAR_SCIENCE,
                    "news": ReportStyle.NEWS,
                    "social_media": ReportStyle.SOCIAL_MEDIA,
                }
                report_style = style_mapping.get(
                    request.report_style, ReportStyle.ACADEMIC
                )
            except Exception:
                # If invalid style, default to ACADEMIC
                report_style = ReportStyle.ACADEMIC
        else:
            report_style = ReportStyle.ACADEMIC

        workflow = build_prompt_enhancer_graph()
        final_state = workflow.invoke(
            {
                "prompt": request.prompt,
                "context": request.context,
                "report_style": report_style,
            }
        )
        return {"result": final_state["output"]}
    except Exception as e:
        logger.exception(f"Error occurred during prompt enhancement: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/mcp/server/metadata", response_model=MCPServerMetadataResponse)
async def mcp_server_metadata(request: MCPServerMetadataRequest):
    """Get information about an MCP server."""
    try:
        # Set default timeout with a longer value for this endpoint
        timeout = 300  # Default to 300 seconds for this endpoint

        # Use custom timeout from request if provided
        if request.timeout_seconds is not None:
            timeout = request.timeout_seconds

        # Load tools from the MCP server using the utility function
        tools = await load_mcp_tools(
            server_type=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            timeout_seconds=timeout,
        )

        # Create the response with tools
        response = MCPServerMetadataResponse(
            transport=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            tools=tools,
        )

        return response
    except Exception as e:
        logger.exception(f"Error in MCP server metadata endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.get("/api/rag/config", response_model=RAGConfigResponse)
async def rag_config():
    """Get the config of the RAG."""
    return RAGConfigResponse(provider=SELECTED_RAG_PROVIDER)


@app.get("/api/rag/resources", response_model=RAGResourcesResponse)
async def rag_resources(request: Annotated[RAGResourceRequest, Query()]):
    """Get the resources of the RAG."""
    retriever = build_retriever()
    if retriever:
        return RAGResourcesResponse(resources=retriever.list_resources(request.query))
    return RAGResourcesResponse(resources=[])


@app.get("/api/config", response_model=ConfigResponse)
async def config():
    """Get the config of the server."""
    return ConfigResponse(
        rag=RAGConfigResponse(provider=SELECTED_RAG_PROVIDER),
        models=get_configured_llm_models(),
    )


@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system performance metrics."""
    import time
    import sys

    try:
        # Get connection pool metrics
        connection_metrics = await _get_connection_metrics()

        # Get advanced component metrics if available
        advanced_metrics = {}

        adv_executor = get_global_component("advanced_parallel_executor")
        if adv_executor:
            advanced_metrics["parallel_executor"] = {
                "active_tasks": (
                    len(adv_executor._tasks) if hasattr(adv_executor, "_tasks") else 0
                ),
                "status": "active",
            }

        rate_limiter = get_global_component("adaptive_rate_limiter")
        if rate_limiter:
            advanced_metrics["rate_limiter"] = {
                "current_rate": getattr(rate_limiter, "_current_rate", 0),
                "status": "active",
            }

        hier_memory = get_global_component("hierarchical_memory")
        if hier_memory:
            cache_stats = (
                hier_memory.get_stats() if hasattr(hier_memory, "get_stats") else {}
            )
            advanced_metrics["memory_manager"] = {
                "cache_stats": cache_stats,
                "status": "active",
            }

        error_recovery = get_global_component("smart_error_recovery")
        if error_recovery:
            advanced_metrics["error_recovery"] = {
                "circuit_breaker_state": getattr(
                    error_recovery, "_circuit_state", "unknown"
                ),
                "status": "active",
            }

        # Compile comprehensive metrics
        metrics = {
            "timestamp": time.time(),
            "connection_pool": connection_metrics,
            "request_queue": {
                "size": request_queue.qsize(),
                "max_size": getattr(request_queue, "_maxsize", "unlimited"),
            },
            "advanced_optimizations": advanced_metrics,
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
            },
        }

        return metrics

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Enhanced health check with component status."""
    import time

    try:
        config = get_performance_config()

        # Check core components
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "configuration": {
                "advanced_optimization": config.enable_advanced_optimization,
                "collaboration": config.enable_collaboration,
                "debug_mode": config.debug_mode,
            },
            "components": {
                "connection_pool": (
                    "healthy"
                    if get_global_component("connection_pool")
                    else "not_initialized"
                ),
                "request_queue": "healthy" if request_queue else "not_initialized",
                "batch_processor": (
                    "healthy"
                    if get_global_component("batch_processor")
                    and not get_global_component("batch_processor").done()
                    else "inactive"
                ),
            },
            "advanced_components": {
                "parallel_executor": (
                    "active"
                    if get_global_component("advanced_parallel_executor")
                    else "inactive"
                ),
                "rate_limiter": (
                    "active"
                    if get_global_component("adaptive_rate_limiter")
                    else "inactive"
                ),
                "memory_manager": (
                    "active"
                    if get_global_component("hierarchical_memory")
                    else "inactive"
                ),
                "error_recovery": (
                    "active"
                    if get_global_component("smart_error_recovery")
                    else "inactive"
                ),
            },
        }

        # Check if any critical component is unhealthy
        if not get_global_component("connection_pool") or not request_queue:
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}


@app.get("/config")
async def get_config():
    """Get current performance configuration."""
    try:
        config = get_performance_config()
        return {"timestamp": time.time(), "configuration": config.to_dict()}
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config")
async def update_config(config_update: Dict[str, Any]):
    """Update performance configuration dynamically."""
    try:
        # Validate and update configuration
        update_performance_config(config_update)

        # Get updated configuration
        updated_config = get_performance_config()

        logger.info(f"Configuration updated: {config_update}")

        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "timestamp": time.time(),
            "updated_configuration": updated_config.to_dict(),
        }

    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/config/reset")
async def reset_config():
    """Reset configuration to defaults."""
    try:
        from src.config.performance_config import reset_performance_config

        reset_performance_config()
        config = get_performance_config()

        logger.info("Configuration reset to defaults")

        return {
            "status": "success",
            "message": "Configuration reset to defaults",
            "timestamp": time.time(),
            "default_configuration": config.to_dict(),
        }

    except Exception as e:
        logger.error(f"Error resetting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
