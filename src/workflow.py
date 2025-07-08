# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from src.graph.builder import build_graph
from src.collaboration.role_bidding import RoleBiddingSystem
from src.collaboration.human_loop import HumanLoopController
from src.collaboration.consensus_system import ConflictResolutionSystem
from src.utils.performance_optimizer import (
    AdvancedParallelExecutor, AdaptiveRateLimiter, SmartErrorRecovery,
    TaskPriority, optimize_report_generation_workflow
)
from src.utils.workflow_optimizer import (
    WorkflowOptimizer, WorkflowOptimizationLevel, WorkflowOptimizationConfig,
    create_optimized_workflow, optimize_single_research_task
)
from src.utils.memory_manager import HierarchicalMemoryManager, cached
from src.utils.rate_limiter import RateLimiter
from src.utils.parallel_executor import ParallelExecutor
from src.utils.error_recovery import ErrorRecoveryManager
from src.config.config_loader import config_loader

# Import collaboration modules
try:
    from src.collaboration.role_bidding import TaskRequirement, TaskType, create_default_agents
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
advanced_parallel_executor = AdvancedParallelExecutor(max_workers=8, enable_metrics=True)
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


async def get_workflow_optimizer(optimization_level: WorkflowOptimizationLevel = WorkflowOptimizationLevel.ADVANCED) -> WorkflowOptimizer:
    """获取全局工作流优化器实例"""
    global _global_workflow_optimizer
    
    if _global_workflow_optimizer is None:
        _global_workflow_optimizer = await create_optimized_workflow(
            optimization_level=optimization_level,
            max_workers=8
        )
    
    return _global_workflow_optimizer


async def run_optimized_research_workflow(
    user_input: str,
    workflow_type: str = 'research',
    optimization_level: WorkflowOptimizationLevel = WorkflowOptimizationLevel.ADVANCED,
    enable_parallel_tasks: bool = True,
    max_workers: int = 8
) -> Dict[str, Any]:
    """运行优化的研究工作流，专门针对并行化处理优化
    
    Args:
        user_input: 用户查询
        workflow_type: 工作流类型 ('research', 'analysis', 'report')
        optimization_level: 优化级别
        enable_parallel_tasks: 是否启用并行任务处理
        max_workers: 最大工作线程数
    
    Returns:
        包含研究结果和性能指标的字典
    """
    start_time = time.time()
    logger.info(f"Starting optimized research workflow: {user_input}")
    
    try:
        if enable_parallel_tasks and optimization_level in [WorkflowOptimizationLevel.ADVANCED, WorkflowOptimizationLevel.MAXIMUM]:
            # 使用高级并行优化
            optimizer = await get_workflow_optimizer(optimization_level)
            
            # 执行优化的研究工作流
            result = await optimizer.optimize_research_workflow(
                user_query=user_input,
                workflow_type=workflow_type
            )
            
            # 获取性能指标
            metrics = await optimizer.get_optimization_metrics()
            result['workflow_metrics'] = metrics
            
        else:
            # 回退到标准工作流
            result = await run_agent_workflow_async(
                user_input=user_input,
                enable_advanced_optimization=True,
                enable_background_research=True
            )
            result['optimization_applied'] = False
        
        # 添加执行时间
        execution_time = time.time() - start_time
        result['total_execution_time'] = execution_time
        result['workflow_type'] = workflow_type
        
        logger.info(f"Optimized research workflow completed in {execution_time:.2f} seconds")
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Optimized research workflow failed after {execution_time:.2f} seconds: {e}")
        
        # 回退到基础工作流
        try:
            result = await run_agent_workflow_async(
                user_input=user_input,
                enable_advanced_optimization=False
            )
            result['fallback_used'] = True
            result['original_error'] = str(e)
            return result
        except Exception as fallback_error:
            raise RuntimeError(f"Both optimized and fallback workflows failed. Original: {e}, Fallback: {fallback_error}")


async def run_parallel_report_generation(
    content_sections: List[str],
    report_type: str = 'comprehensive',
    user_context: Optional[str] = None,
    optimization_level: WorkflowOptimizationLevel = WorkflowOptimizationLevel.ADVANCED
) -> Dict[str, Any]:
    """并行化报告生成工作流
    
    Args:
        content_sections: 报告内容部分列表
        report_type: 报告类型
        user_context: 用户上下文信息
        optimization_level: 优化级别
    
    Returns:
        生成的报告和性能指标
    """
    start_time = time.time()
    logger.info(f"Starting parallel report generation with {len(content_sections)} sections")
    
    try:
        optimizer = await get_workflow_optimizer(optimization_level)
        
        # 并行生成报告各部分
        result = await optimizer.optimize_report_generation(
            content_sections=content_sections,
            report_type=report_type
        )
        
        # 如果有用户上下文，进行后处理
        if user_context:
            result['user_context'] = user_context
            result['context_applied'] = True
        
        # 添加执行统计
        execution_time = time.time() - start_time
        result['generation_time'] = execution_time
        result['sections_processed'] = len(content_sections)
        
        logger.info(f"Parallel report generation completed in {execution_time:.2f} seconds")
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Parallel report generation failed after {execution_time:.2f} seconds: {e}")
        
        # 回退到顺序处理
        return {
            'error': str(e),
            'fallback_used': True,
            'execution_time': execution_time,
            'sections_count': len(content_sections)
        }


@cached(ttl=CACHE_TTL, priority=2)
async def run_agent_workflow_async(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 3,
    enable_background_research: bool = True,
    enable_collaboration: bool = False,
    enable_caching: bool = True,
    thread_id: Optional[str] = None,
    enable_advanced_optimization: bool = True,
    optimization_level: Optional[WorkflowOptimizationLevel] = None,
    enable_intelligent_task_decomposition: bool = True,
    enable_dynamic_resource_allocation: bool = True
) -> Dict[str, Any]:
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        enable_background_research: If True, performs web search before planning to enhance context
        enable_collaboration: If True, enables collaboration features
        enable_caching: If True, enables result caching
        thread_id: Optional thread ID for session management
        enable_advanced_optimization: If True, enables advanced optimization features
        optimization_level: Workflow optimization level (BASIC, STANDARD, ADVANCED, MAXIMUM)
        enable_intelligent_task_decomposition: If True, enables intelligent task decomposition
        enable_dynamic_resource_allocation: If True, enables dynamic resource allocation

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
    
    # 确定优化级别
    if optimization_level is None:
        if enable_advanced_optimization:
            optimization_level = WorkflowOptimizationLevel.ADVANCED
        else:
            optimization_level = WorkflowOptimizationLevel.BASIC
    
    # 初始化优化组件
    if optimization_level in [WorkflowOptimizationLevel.ADVANCED, WorkflowOptimizationLevel.MAXIMUM]:
        # 获取工作流优化器
        workflow_optimizer = None
        if enable_intelligent_task_decomposition or enable_dynamic_resource_allocation:
            try:
                workflow_optimizer = await get_workflow_optimizer(optimization_level)
                logger.info(f"Using {optimization_level.value} workflow optimization")
            except Exception as e:
                logger.warning(f"Failed to initialize workflow optimizer: {e}, falling back to standard optimization")
        
        # 启动高级组件
        await advanced_parallel_executor.start()
        await hierarchical_memory.start()
        
        # 高级速率限制检查
        if not await adaptive_rate_limiter.acquire():
            raise RuntimeError("Adaptive rate limit exceeded. Please try again later.")
    else:
        # 回退到基础速率限制
        if not await rate_limiter.acquire():
            raise RuntimeError("Rate limit exceeded. Please try again later.")
    
    # Cache key generation
    cache_key = None
    if enable_caching:
        if enable_advanced_optimization:
            # Use hierarchical memory for advanced caching
            cache_key = {
                "user_input": user_input,
                "max_plan_iterations": max_plan_iterations,
                "enable_background_research": enable_background_research,
                "enable_collaboration": enable_collaboration,
                "thread_id": thread_id
            }
            
            # Check hierarchical cache
            cached_result = await hierarchical_memory.get(cache_key)
            if cached_result:
                logger.info(f"Hierarchical cache hit for workflow")
                return cached_result
        else:
            # Use legacy caching
            import hashlib
            cache_key = hashlib.md5(
                f"{user_input}_{max_plan_iterations}_{enable_background_research}_{enable_collaboration}".encode()
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
            "conflict_resolution": ConflictResolutionSystem()
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
        "auto_accepted_plan": True,
        "enable_background_research": enable_background_research,
        "enable_collaboration": enable_collaboration and collaboration_systems is not None,
        "collaboration_systems": collaboration_systems,
        "start_time": start_time,
        "thread_id": thread_id or "default",
        "enable_advanced_optimization": enable_advanced_optimization,
        # 新增优化配置
        "optimization_level": optimization_level.value if optimization_level else "basic",
        "enable_intelligent_task_decomposition": enable_intelligent_task_decomposition,
        "enable_dynamic_resource_allocation": enable_dynamic_resource_allocation,
        "workflow_optimizer_available": workflow_optimizer is not None if 'workflow_optimizer' in locals() else False
    }
    # Load model token limits from configuration
    try:
        configuration = config_loader.create_configuration()
        model_token_limits = configuration.model_token_limits
        logger.debug(f"Loaded model_token_limits: {list(model_token_limits.keys())}")
    except Exception as e:
        logger.warning(f"Failed to load model_token_limits from config: {e}")
        model_token_limits = {}
    
    config = {
        "configurable": {
            "thread_id": thread_id or "default",
            "max_plan_iterations": max_plan_iterations,
            "enable_collaboration": enable_collaboration and collaboration_systems is not None,
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
        "recursion_limit": 100,
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
                _execute_workflow,
                operation_id=f"workflow_{thread_id or 'default'}"
            )
        else:
            final_state = await error_recovery.execute_with_retry(_execute_workflow)
        
        # Cache the result if caching is enabled
        if enable_caching and cache_key and final_state:
            if enable_advanced_optimization:
                await hierarchical_memory.set(cache_key, final_state, ttl=CACHE_TTL, priority=2)
            else:
                _cache_result(cache_key, final_state)
        
        # Performance logging with enhanced metrics
        execution_time = time.time() - start_time
        if enable_advanced_optimization:
            performance_metrics = {
                "parallel_executor": advanced_parallel_executor.get_metrics(),
                "rate_limiter": adaptive_rate_limiter.get_stats(),
                "error_recovery": smart_error_recovery.get_stats(),
                "memory_manager": hierarchical_memory.get_stats()
            }
            final_state["performance_metrics"] = performance_metrics
        
        logger.info(f"Async workflow completed successfully in {execution_time:.2f} seconds")
        
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


async def _process_workflow_output(output: Dict[str, Any], debug: bool) -> Dict[str, Any]:
    """Process workflow output with optional debugging."""
    try:
        if debug:
            logger.debug(f"Processing workflow output: {output}")
        
        # Add processing timestamp
        processed_output = {
            **output,
            "processed_at": time.time()
        }
        
        return processed_output
    except Exception as e:
        logger.error(f"Error processing workflow output: {e}")
        return output


def _cache_result(cache_key: str, result: Dict[str, Any]) -> None:
    """Cache the workflow result."""
    workflow_cache[cache_key] = {
        "result": result,
        "timestamp": time.time()
    }
    
    # Simple cache cleanup - remove oldest entries if cache gets too large
    if len(workflow_cache) > 100:  # Max 100 cached results
        oldest_key = min(workflow_cache.keys(), 
                        key=lambda k: workflow_cache[k]["timestamp"])
        del workflow_cache[oldest_key]


if __name__ == "__main__":
    print(graph.get_graph(xray=True).draw_mermaid())
