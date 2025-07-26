# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
import time
from typing import Dict, Any, Optional, List
from src.graph.builder import build_graph
from src.collaboration.role_bidding import RoleBiddingSystem
from src.collaboration.human_loop import HumanLoopController
from src.collaboration.consensus_system import ConflictResolutionSystem
from src.utils.performance.performance_optimizer import (
    AdvancedParallelExecutor,
    AdaptiveRateLimiter,
    SmartErrorRecovery,
)
from src.utils.performance.workflow_optimizer import (
    WorkflowOptimizer,
    WorkflowOptimizationLevel,
    create_optimized_workflow,
)
from src.utils.performance.memory_manager import HierarchicalMemoryManager
from src.config import get_settings

# Import collaboration modules
try:
    from src.collaboration.role_bidding import (
        create_default_agents,
    )

    COLLABORATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Collaboration modules not available: {e}")
    COLLABORATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
graph = build_graph()

# Advanced performance optimization and caching
advanced_parallel_executor = AdvancedParallelExecutor(
    max_workers=8, enable_metrics=True
)
adaptive_rate_limiter = AdaptiveRateLimiter(initial_rate=150, time_window=60)
smart_error_recovery = SmartErrorRecovery(max_retries=3, base_delay=1.0, max_delay=60.0)
hierarchical_memory = HierarchicalMemoryManager(
    l1_max_size=200 * 1024 * 1024,  # 200MB L1 cache
    l2_max_size=2 * 1024 * 1024 * 1024,  # 2GB L2 cache
    l3_max_size=20 * 1024 * 1024 * 1024,  # 20GB L3 cache
)
workflow_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 3600  # 1 hour cache TTL

# Legacy compatibility
rate_limiter = adaptive_rate_limiter
parallel_executor = advanced_parallel_executor
error_recovery = smart_error_recovery

# Global workflow optimizer instance
_global_workflow_optimizer: Optional[WorkflowOptimizer] = None


async def get_workflow_optimizer(
    optimization_level: WorkflowOptimizationLevel = WorkflowOptimizationLevel.ADVANCED,
) -> WorkflowOptimizer:
    """Get global workflow optimizer instance"""
    global _global_workflow_optimizer

    if _global_workflow_optimizer is None:
        _global_workflow_optimizer = await create_optimized_workflow(
            optimization_level=optimization_level, max_workers=8
        )

    return _global_workflow_optimizer


async def run_optimized_research_workflow(
    user_input: str,
    workflow_type: str = "research",
    optimization_level: WorkflowOptimizationLevel = WorkflowOptimizationLevel.ADVANCED,
    enable_parallel_tasks: bool = True,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """Run optimized research workflow with parallel processing optimization

    Args:
        user_input: User query
        workflow_type: Workflow type ('research', 'analysis', 'report')
        optimization_level: Optimization level
        enable_parallel_tasks: Whether to enable parallel task processing
        max_workers: Maximum number of worker threads

    Returns:
        Dictionary containing research results and performance metrics
    """
    start_time = time.time()
    logger.info(f"Starting optimized research workflow: {user_input}")

    try:
        if enable_parallel_tasks and optimization_level in [
            WorkflowOptimizationLevel.ADVANCED,
            WorkflowOptimizationLevel.MAXIMUM,
        ]:
            # Use advanced parallel optimization
            optimizer = await get_workflow_optimizer(optimization_level)

            # Execute optimized research workflow
            result = await optimizer.optimize_research_workflow(
                user_query=user_input, workflow_type=workflow_type
            )

            # Get performance metrics
            metrics = await optimizer.get_optimization_metrics()
            result["workflow_metrics"] = metrics

        else:
            # Fallback to standard workflow
            result = await run_agent_workflow_async(
                user_input=user_input,
                enable_advanced_optimization=True,
                enable_background_research=True,
            )
            result["optimization_applied"] = False

        # Add execution time
        execution_time = time.time() - start_time
        result["total_execution_time"] = execution_time
        result["workflow_type"] = workflow_type

        logger.info(
            f"Optimized research workflow completed in {execution_time:.2f} seconds"
        )
        return result

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            f"Optimized research workflow failed after {execution_time:.2f} seconds: {e}"
        )

        # Fallback to basic workflow
        try:
            result = await run_agent_workflow_async(
                user_input=user_input, enable_advanced_optimization=False
            )
            result["fallback_used"] = True
            result["original_error"] = str(e)
            return result
        except Exception as fallback_error:
            raise RuntimeError(
                f"Both optimized and fallback workflows failed. Original: {e}, Fallback: {fallback_error}"
            )


async def run_parallel_report_generation(
    content_sections: List[str],
    report_type: str = "comprehensive",
    user_context: Optional[str] = None,
    optimization_level: WorkflowOptimizationLevel = WorkflowOptimizationLevel.ADVANCED,
) -> Dict[str, Any]:
    """Parallel report generation workflow

    Args:
        content_sections: List of report content sections
        report_type: Type of report
        user_context: User context information
        optimization_level: Optimization level

    Returns:
        Generated report and performance metrics
    """
    start_time = time.time()
    logger.info(
        f"Starting parallel report generation with {len(content_sections)} sections"
    )

    try:
        optimizer = await get_workflow_optimizer(optimization_level)

        # Generate report sections in parallel
        result = await optimizer.optimize_report_generation(
            content_sections=content_sections, report_type=report_type
        )

        # Post-process with user context if provided
        if user_context:
            result["user_context"] = user_context
            result["context_applied"] = True

        # Add execution statistics
        execution_time = time.time() - start_time
        result["generation_time"] = execution_time
        result["sections_processed"] = len(content_sections)

        logger.info(
            f"Parallel report generation completed in {execution_time:.2f} seconds"
        )
        return result

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            f"Parallel report generation failed after {execution_time:.2f} seconds: {e}"
        )

        # Fallback to sequential processing
        return {
            "error": str(e),
            "fallback_used": True,
            "execution_time": execution_time,
            "sections_count": len(content_sections),
        }


async def run_agent_workflow_async(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 3,
    max_step_num: Optional[int] = None,
    enable_background_research: bool = True,
    enable_collaboration: bool = False,
    enable_caching: bool = True,
    thread_id: Optional[str] = None,
    enable_advanced_optimization: bool = True,
    optimization_level: Optional[WorkflowOptimizationLevel] = None,
    enable_intelligent_task_decomposition: bool = True,
    enable_dynamic_resource_allocation: bool = True,
    settings: Optional[Any] = None,
    locale: str = "en-US",
) -> Dict[str, Any]:
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan (overrides config)
        enable_background_research: If True, performs web search before planning to enhance context
        enable_collaboration: If True, enables collaboration features
        enable_caching: If True, enables result caching
        thread_id: Optional thread ID for session management
        enable_advanced_optimization: If True, enables advanced optimization features
        optimization_level: Workflow optimization level (BASIC, STANDARD, ADVANCED, MAXIMUM)
        enable_intelligent_task_decomposition: If True, enables intelligent task decomposition
        enable_dynamic_resource_allocation: If True, enables dynamic resource allocation
        settings: Configuration settings object

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    # Performance monitoring
    start_time = time.time()
    logger.info(f"Starting async workflow with user input: {user_input}")

    # Determine optimization level
    if optimization_level is None:
        if enable_advanced_optimization:
            optimization_level = WorkflowOptimizationLevel.ADVANCED
        else:
            optimization_level = WorkflowOptimizationLevel.BASIC

    # Initialize optimization components
    if optimization_level in [
        WorkflowOptimizationLevel.ADVANCED,
        WorkflowOptimizationLevel.MAXIMUM,
    ]:
        # Get workflow optimizer
        workflow_optimizer = None
        if enable_intelligent_task_decomposition or enable_dynamic_resource_allocation:
            try:
                workflow_optimizer = await get_workflow_optimizer(optimization_level)
                logger.info(f"Using {optimization_level.value} workflow optimization")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize workflow optimizer: {e}, falling back to standard optimization"
                )

        # Start advanced components
        await advanced_parallel_executor.start()
        await hierarchical_memory.start()

        # Advanced rate limiting check
        if not await adaptive_rate_limiter.acquire():
            raise RuntimeError("Adaptive rate limit exceeded. Please try again later.")
    else:
        # Fallback to basic rate limiting
        if not await rate_limiter.acquire():
            raise RuntimeError("Rate limit exceeded. Please try again later.")

    # Cache key generation
    cache_key = None
    if enable_caching:
        # Determine effective max_step_num for caching
        cache_max_step_num = max_step_num
        if cache_max_step_num is None and settings is not None:
            cache_max_step_num = getattr(settings.agents, "max_step_num", 2)
        if cache_max_step_num is None:
            cache_max_step_num = 2

        if enable_advanced_optimization:
            # Use hierarchical memory for advanced caching
            cache_key = {
                "user_input": user_input,
                "max_plan_iterations": max_plan_iterations,
                "max_step_num": cache_max_step_num,
                "enable_background_research": enable_background_research,
                "enable_collaboration": enable_collaboration,
                "thread_id": thread_id,
            }

            # Check hierarchical cache
            cached_result = await hierarchical_memory.get(cache_key)
            if cached_result:
                logger.info("Hierarchical cache hit for workflow")
                return cached_result
        else:
            # Use legacy caching
            import hashlib

            cache_key = hashlib.md5(
                f"{user_input}_{max_plan_iterations}_{cache_max_step_num}_{enable_background_research}_{enable_collaboration}".encode()
            ).hexdigest()

            # Check cache
            cached_result = _get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached_result

    # Initialize collaboration systems if available
    collaboration_systems = None
    if COLLABORATION_AVAILABLE and enable_collaboration:
        collaboration_systems = {
            "role_bidding": RoleBiddingSystem(),
            "human_loop": HumanLoopController(),
            "conflict_resolution": ConflictResolutionSystem(),
        }

        # Register default agents
        try:
            for agent in create_default_agents():
                collaboration_systems["role_bidding"].register_agent(agent)
            logger.info("Collaboration systems initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize collaboration systems: {e}")
            collaboration_systems = None

    initial_state = {
        # Runtime Variables
        "messages": [{"role": "user", "content": user_input}],
        "locale": (
            locale
        ),  # Use passed locale parameter to ensure correct multilingual settings for initial researcher report
        "auto_accepted_plan": True,
        "enable_background_research": enable_background_research,
        "enable_collaboration": (
            enable_collaboration and collaboration_systems is not None
        ),
        "collaboration_systems": collaboration_systems,
        "start_time": start_time,
        "thread_id": thread_id or "default",
        "enable_advanced_optimization": enable_advanced_optimization,
        # Additional optimization configuration
        "optimization_level": (
            optimization_level.value if optimization_level else "basic"
        ),
        "enable_intelligent_task_decomposition": enable_intelligent_task_decomposition,
        "enable_dynamic_resource_allocation": enable_dynamic_resource_allocation,
        "workflow_optimizer_available": (
            workflow_optimizer is not None
            if "workflow_optimizer" in locals()
            else False
        ),
    }
    # Load model token limits from configuration
    try:
        if settings is None:
            settings = get_settings()
        model_token_limits = getattr(settings, "model_token_limits", {})
        logger.debug(
            f"Loaded model_token_limits: {list(model_token_limits.keys()) if model_token_limits else 'None'}"
        )
    except Exception as e:
        logger.warning(f"Failed to load model_token_limits from config: {e}")
        model_token_limits = {}

    # Determine max_step_num value
    effective_max_step_num = max_step_num
    if effective_max_step_num is None and settings is not None:
        effective_max_step_num = getattr(settings.agents, "max_step_num", 2)
    if effective_max_step_num is None:
        effective_max_step_num = 2

    # Get recursion limit from environment variable
    default_recursion_limit = 10
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)
        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Workflow recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    config = {
        "configurable": {
            "thread_id": thread_id or "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": effective_max_step_num,
            "enable_collaboration": (
                enable_collaboration and collaboration_systems is not None
            ),
            "model_token_limits": model_token_limits,
            "mcp_settings": {
                "servers": {
                    "mcp-github-trending": {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-github-trending"],
                        "enabled_tools": ["get_github_trending_repositories"],
                        "add_to_agents": ["researcher"],
                    }
                }
            },
        },
        "recursion_limit": recursion_limit,
    }
    last_message_cnt = 0
    final_state = None

    try:
        # Execute workflow with smart error recovery
        async def _execute_workflow():
            nonlocal final_state, last_message_cnt
            async for s in graph.astream(
                input=initial_state, config=config, stream_mode="values"
            ):
                try:
                    final_state = s  # Keep track of the latest state
                    if isinstance(s, dict) and "messages" in s:
                        if len(s["messages"]) <= last_message_cnt:
                            continue
                        last_message_cnt = len(s["messages"])
                        message = s["messages"][-1]
                        if isinstance(message, tuple):
                            print(message)
                        else:
                            message.pretty_print()
                    else:
                        # For any other output format
                        print(f"Output: {s}")
                except Exception as e:
                    logger.error(f"Error processing stream output: {e}")
                    print(f"Error processing output: {str(e)}")
            return final_state

        # Execute with error recovery based on optimization level
        if enable_advanced_optimization:
            final_state = await smart_error_recovery.execute_with_recovery(
                _execute_workflow, operation_id=f"workflow_{thread_id or 'default'}"
            )
        else:
            final_state = await error_recovery.execute_with_recovery(
                _execute_workflow,
                operation_id=f"workflow_{thread_id or 'default'}_fallback",
            )

        # Cache the result if caching is enabled
        if enable_caching and cache_key and final_state:
            if enable_advanced_optimization:
                await hierarchical_memory.set(
                    cache_key, final_state, ttl=CACHE_TTL, priority=2
                )
            else:
                _cache_result(cache_key, final_state)

        # Performance logging with enhanced metrics
        execution_time = time.time() - start_time
        if enable_advanced_optimization:
            performance_metrics = {
                "parallel_executor": advanced_parallel_executor.get_metrics(),
                "rate_limiter": adaptive_rate_limiter.get_stats(),
                "error_recovery": smart_error_recovery.get_stats(),
                "memory_manager": hierarchical_memory.get_stats(),
            }
            final_state["performance_metrics"] = performance_metrics

        logger.info(
            f"Async workflow completed successfully in {execution_time:.2f} seconds"
        )

        return final_state

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Workflow failed after {execution_time:.2f} seconds: {str(e)}")
        raise


def _get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result if available and not expired."""
    if cache_key not in workflow_cache:
        return None

    cached_data = workflow_cache[cache_key]
    if time.time() - cached_data["timestamp"] > CACHE_TTL:
        # Cache expired, remove it
        del workflow_cache[cache_key]
        return None

    return cached_data["result"]


async def _process_workflow_output(
    output: Dict[str, Any], debug: bool
) -> Dict[str, Any]:
    """Process workflow output with optional debugging."""
    try:
        if debug:
            logger.debug(f"Processing workflow output: {output}")

        # Add processing timestamp
        processed_output = {**output, "processed_at": time.time()}

        return processed_output
    except Exception as e:
        logger.error(f"Error processing workflow output: {e}")
        return output


def _cache_result(cache_key: str, result: Dict[str, Any]) -> None:
    """Cache the workflow result."""
    workflow_cache[cache_key] = {"result": result, "timestamp": time.time()}

    # Simple cache cleanup - remove oldest entries if cache gets too large
    if len(workflow_cache) > 100:  # Max 100 cached results
        oldest_key = min(
            workflow_cache.keys(), key=lambda k: workflow_cache[k]["timestamp"]
        )
        del workflow_cache[oldest_key]


if __name__ == "__main__":
    print(graph.get_graph(xray=True).draw_mermaid())
