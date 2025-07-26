"""Enhanced Researcher Main Class - Integrates all managers and provides unified research interface"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .isolation_config_manager import IsolationConfigManager
from .research_tool_manager import ResearchToolManager
from .mcp_client_manager import MCPClientManager
from .reflection_system_manager import ReflectionSystemManager
from .iterative_research_engine import IterativeResearchEngine
from .research_result_processor import ResearchResultProcessor
from .performance_monitor import global_performance_tracker
from .config_cache_optimizer import global_config_cache_optimizer
from .concurrent_optimizer import (
    TaskPriority,
    global_concurrent_manager,
)

logger = logging.getLogger(__name__)


class EnhancedResearcher:
    """Enhanced Researcher - Main class that unifies management of all research components"""

    def __init__(self, state: Dict[str, Any]):
        """Initialize Enhanced Researcher

        Args:
            state: Research state dictionary
        """
        self.state = state
        self.initialization_time = datetime.now()

        # Create a simple configurable object
        from types import SimpleNamespace

        configurable = SimpleNamespace()
        configurable.locale = state.get("locale", "en-US")

        # Create nested agents and content objects
        configurable.agents = SimpleNamespace()
        configurable.agents.max_search_results = 10

        configurable.content = SimpleNamespace()
        configurable.content.enable_smart_filtering = True

        # Add other required attributes
        configurable.reflection = SimpleNamespace()
        configurable.reflection.reflection_model = "basic"

        configurable.mcp = SimpleNamespace()
        configurable.mcp.enabled = False
        configurable.mcp.servers = []

        configurable.enable_enhanced_reflection = True
        configurable.max_reflection_loops = 3
        configurable.knowledge_gap_threshold = 0.7
        configurable.sufficiency_threshold = 0.8
        configurable.enable_reflection_integration = True

        # Initialize all managers
        self.config_manager = IsolationConfigManager(configurable)
        self.unified_config = self.config_manager.get_unified_config()

        self.tool_manager = ResearchToolManager(configurable, state)
        self.mcp_manager = MCPClientManager(self.config_manager)
        self.reflection_manager = ReflectionSystemManager(self.unified_config)
        self.research_engine = IterativeResearchEngine(self.unified_config)
        self.result_processor = ResearchResultProcessor(self.unified_config)

        # Research components (lazy initialization)
        self.reflection_agent = None
        self.reflection_integrator = None
        self.research_tools = None
        self.mcp_tools = None

        # Performance monitoring
        self.performance_monitor = global_performance_tracker.create_monitor(
            f"enhanced_researcher_{id(self)}"
        )

        # Cache optimizer
        self.cache_optimizer = global_config_cache_optimizer

        # Concurrent manager
        self.concurrent_manager = global_concurrent_manager

        logger.info("EnhancedResearcher initialized successfully")

    async def initialize_research_components(self) -> Dict[str, Any]:
        """Initialize all research components

        Returns:
            Initialization result summary
        """
        logger.info("Initializing research components")

        initialization_results = {
            "config_initialized": True,
            "tools_initialized": False,
            "mcp_initialized": False,
            "reflection_initialized": False,
            "errors": [],
        }

        try:
            # Initialize research tools
            self.research_tools = await self.tool_manager.get_all_tools()
            initialization_results["tools_initialized"] = True
            logger.info(f"Initialized {len(self.research_tools)} research tools")

        except Exception as e:
            error_msg = f"Failed to initialize research tools: {e}"
            logger.error(error_msg)
            initialization_results["errors"].append(error_msg)

        try:
            # Initialize MCP tools
            if self.unified_config.enable_mcp:
                self.mcp_tools = await self.mcp_manager.get_mcp_tools()
                initialization_results["mcp_initialized"] = True
                logger.info(f"Initialized {len(self.mcp_tools)} MCP tools")
            else:
                logger.info("MCP is disabled, skipping MCP tools initialization")

        except Exception as e:
            error_msg = f"Failed to initialize MCP tools: {e}"
            logger.error(error_msg)
            initialization_results["errors"].append(error_msg)

        try:
            # Initialize reflection system
            self.reflection_agent, self.reflection_integrator = (
                await self.reflection_manager.initialize_reflection_system()
            )
            initialization_results["reflection_initialized"] = (
                self.reflection_agent is not None
            )

        except Exception as e:
            error_msg = f"Failed to initialize reflection system: {e}"
            logger.error(error_msg)
            initialization_results["errors"].append(error_msg)

        # Start research session
        self.research_engine.start_research_session()

        logger.info(
            f"Research components initialization completed with {len(initialization_results['errors'])} errors"
        )
        return initialization_results

    async def conduct_research(self, initial_query: str) -> Dict[str, Any]:
        """Execute complete research workflow

        Args:
            initial_query: Initial query

        Returns:
            Research result dictionary
        """
        logger.info(f"Starting research for query: {initial_query}")

        # Start performance monitoring
        async with self.performance_monitor.monitor_context(sample_interval=0.5):
            self.performance_monitor.record_task_start()

            # Ensure components are initialized
            if not self.research_tools:
                await self.initialize_research_components()

            research_results = {
                "initial_query": initial_query,
                "start_time": datetime.now(),
                "iterations": [],
                "final_result": None,
                "termination_reason": None,
                "total_iterations": 0,
                "errors": [],
            }

        try:
            # Set initial query
            current_queries = [initial_query]
            self.performance_monitor.add_custom_metric("phase", "research_loop")

            while True:
                # Increment iteration count
                iteration = self.research_engine.increment_iteration()

                logger.info(f"Starting research iteration {iteration}")
                self.performance_monitor.add_custom_metric(
                    "current_iteration", iteration
                )

                iteration_result = {
                    "iteration": iteration,
                    "queries": current_queries.copy(),
                    "search_results": [],
                    "reflection_result": None,
                    "follow_up_queries": [],
                    "timestamp": datetime.now(),
                }

                # Execute search queries
                self.performance_monitor.add_custom_metric("phase", "search_execution")
                for query in current_queries:
                    try:
                        search_result = await self._execute_search_query(query)
                        processed_result = self.result_processor.process_search_result(
                            search_result, query, iteration
                        )
                        iteration_result["search_results"].append(processed_result)

                        # Add to research history
                        self.research_engine.add_research_record(query, search_result)

                    except Exception as e:
                        error_msg = f"Search failed for query '{query}': {e}"
                        logger.error(error_msg)
                        research_results["errors"].append(error_msg)

                # Execute reflection analysis (if enabled)
                if self.reflection_agent:
                    try:
                        self.performance_monitor.add_custom_metric(
                            "phase", "reflection_analysis"
                        )
                        reflection_context = self._build_reflection_context(
                            iteration_result
                        )
                        reflection_result = (
                            await self.reflection_manager.execute_reflection_analysis(
                                reflection_context
                            )
                        )
                        iteration_result["reflection_result"] = reflection_result

                    except Exception as e:
                        error_msg = f"Reflection analysis failed: {e}"
                        logger.error(error_msg)
                        research_results["errors"].append(error_msg)

                # Check termination conditions
                should_terminate, termination_reason, decision_factors = (
                    self.research_engine.check_termination_conditions(
                        self.state, iteration_result.get("reflection_result")
                    )
                )

                iteration_result["termination_check"] = {
                    "should_terminate": should_terminate,
                    "reason": termination_reason,
                    "decision_factors": decision_factors,
                }

                if should_terminate:
                    research_results["termination_reason"] = termination_reason
                    research_results["iterations"].append(iteration_result)
                    break

                # Generate follow-up queries
                if iteration_result.get("reflection_result"):
                    follow_up_queries = self.research_engine.generate_follow_up_queries(
                        iteration_result["reflection_result"], self.state
                    )
                    valid_queries = self.research_engine.filter_valid_queries(
                        follow_up_queries, self.state
                    )
                    iteration_result["follow_up_queries"] = valid_queries
                    current_queries = valid_queries
                else:
                    current_queries = []

                research_results["iterations"].append(iteration_result)

                # If no follow-up queries, terminate research
                if not current_queries:
                    research_results["termination_reason"] = (
                        "No valid follow-up queries"
                    )
                    break

        except Exception as e:
            error_msg = f"Research process failed: {e}"
            logger.error(error_msg)
            research_results["errors"].append(error_msg)
            research_results["termination_reason"] = f"Error: {e}"
            self.performance_monitor.record_task_error()

        # Aggregate final results
        self.performance_monitor.add_custom_metric("phase", "result_aggregation")
        research_results["final_result"] = self._aggregate_final_result(
            research_results
        )
        research_results["total_iterations"] = self.research_engine.iteration_count
        research_results["end_time"] = datetime.now()

        # Record successful completion
        if not research_results["errors"]:
            self.performance_monitor.record_task_success()

        self.performance_monitor.add_custom_metric("research_completed", True)
        self.performance_monitor.add_custom_metric(
            "total_iterations", research_results["total_iterations"]
        )

        logger.info(
            f"Research completed after {research_results['total_iterations']} iterations"
        )
        return research_results

    async def _execute_search_query(self, query: str) -> Any:
        """Execute search query

        Args:
            query: Query string

        Returns:
            Search results
        """
        if not self.research_tools:
            raise RuntimeError("Research tools not initialized")

        # Use the first available search tool
        search_tool = None
        for tool in self.research_tools:
            if hasattr(tool, "search") or "search" in tool.name.lower():
                search_tool = tool
                break

        if not search_tool:
            raise RuntimeError("No search tool available")

        logger.debug(f"Executing search query: {query}")

        # Execute search
        if hasattr(search_tool, "search"):
            result = await search_tool.search(query)
        elif hasattr(search_tool, "run"):
            result = await search_tool.run(query)
        elif hasattr(search_tool, "__call__"):
            result = await search_tool(query)
        else:
            raise RuntimeError(f"Search tool {search_tool.name} has no callable method")

        return result

    def _build_reflection_context(self, iteration_result: Dict[str, Any]) -> Any:
        """Build reflection context

        Args:
            iteration_result: Iteration result

        Returns:
            Reflection context object
        """
        # Need to build context based on actual reflection system interface
        # Return a simple dictionary structure for now
        context = {
            "iteration": iteration_result["iteration"],
            "queries": iteration_result["queries"],
            "search_results": iteration_result["search_results"],
            "research_history": self.research_engine.get_research_history(),
            "state": self.state,
        }

        return context

    def _aggregate_final_result(
        self, research_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate final research results

        Args:
            research_results: Research results dictionary

        Returns:
            Aggregated final results
        """
        all_search_results = []

        # Collect all search results
        for iteration in research_results["iterations"]:
            all_search_results.extend(iteration.get("search_results", []))

        # Use result processor to aggregate results
        aggregated_result = self.result_processor.aggregate_results(all_search_results)

        # Add research summary
        aggregated_result["research_summary"] = self.get_research_summary()

        return aggregated_result

    def get_research_summary(self) -> Dict[str, Any]:
        """Get complete research summary

        Returns:
            Research summary dictionary
        """
        return {
            "initialization_time": self.initialization_time.isoformat(),
            "config_summary": self.config_manager.get_config_summary(),
            "tool_summary": self.tool_manager.get_tool_summary(),
            "mcp_summary": self.mcp_manager.get_mcp_summary(),
            "reflection_summary": self.reflection_manager.get_reflection_summary(),
            "research_engine_summary": self.research_engine.get_research_summary(),
            "result_processor_summary": self.result_processor.get_processing_summary(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "components_status": {
                "reflection_agent_available": self.reflection_agent is not None,
                "reflection_integrator_available": (
                    self.reflection_integrator is not None
                ),
                "research_tools_count": (
                    len(self.research_tools) if self.research_tools else 0
                ),
                "mcp_tools_count": len(self.mcp_tools) if self.mcp_tools else 0,
            },
        }

    async def execute_research_with_reflection(
        self, query: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute research workflow with reflection"""

        # 开始性能监控
        async with self.performance_monitor.monitor_context(sample_interval=0.5):
            self.performance_monitor.record_task_start()

            try:
                logger.info(
                    f"Starting research with reflection for query: {query[:100]}..."
                )

                # Check configuration cache
                config = config or {}
                cached_config = self.cache_optimizer.get_cached_config(
                    "research_config", config
                )
                if cached_config is not None:
                    logger.debug("Using cached research configuration")
                    processed_config = cached_config
                else:
                    # Process configuration and cache
                    processed_config = await self._process_research_config(config)
                    self.cache_optimizer.cache_config(
                        "research_config", config, processed_config
                    )

                # Set up isolation environment
                self.performance_monitor.add_custom_metric("isolation_setup", "started")
                isolation_context = await self.config_manager.setup_isolation_context(
                    processed_config
                )
                self.performance_monitor.add_custom_metric(
                    "isolation_setup", "completed"
                )

                # Use concurrent manager to execute iterative research
                self.performance_monitor.add_custom_metric(
                    "iterative_research", "started"
                )
                research_task = self.concurrent_manager.execute_async_task(
                    self.research_engine.execute_iterative_research(
                        query, isolation_context
                    ),
                    name="iterative_research",
                    priority=TaskPriority.HIGH,
                    timeout=300.0,  # 5分钟超时
                )
                research_task_info = await research_task

                if research_task_info.status.value != "completed":
                    raise Exception(
                        f"Iterative research failed: {research_task_info.error}"
                    )

                research_result = research_task_info.result
                self.performance_monitor.add_custom_metric(
                    "iterative_research", "completed"
                )

                # Concurrently execute reflection analysis and result preprocessing
                self.performance_monitor.add_custom_metric(
                    "reflection_analysis", "started"
                )

                reflection_coro = self.reflection_manager.analyze_and_reflect(
                    query, research_result
                )
                preprocessing_coro = self._preprocess_research_result(research_result)

                concurrent_tasks = await self.concurrent_manager.execute_batch_async(
                    [reflection_coro, preprocessing_coro],
                    names=["reflection_analysis", "result_preprocessing"],
                    timeout=120.0,  # 2分钟超时
                )

                reflection_task_info = concurrent_tasks[0]
                preprocessing_task_info = concurrent_tasks[1]

                if reflection_task_info.status.value != "completed":
                    raise Exception(
                        f"Reflection analysis failed: {reflection_task_info.error}"
                    )
                if preprocessing_task_info.status.value != "completed":
                    raise Exception(
                        f"Result preprocessing failed: {preprocessing_task_info.error}"
                    )

                reflection_result = reflection_task_info.result
                preprocessed_result = preprocessing_task_info.result
                self.performance_monitor.add_custom_metric(
                    "reflection_analysis", "completed"
                )

                # Process final results
                self.performance_monitor.add_custom_metric(
                    "result_processing", "started"
                )
                final_result = await self.result_processor.process_research_result(
                    preprocessed_result, reflection_result
                )
                self.performance_monitor.add_custom_metric(
                    "result_processing", "completed"
                )

                # Record task success
                self.performance_monitor.record_task_success()
                self.performance_monitor.add_custom_metric(
                    "total_sources", len(final_result.get("sources", []))
                )
                self.performance_monitor.add_custom_metric(
                    "reflection_score", reflection_result.get("quality_score", 0)
                )
                self.performance_monitor.add_custom_metric(
                    "concurrent_tasks_executed", len(concurrent_tasks)
                )

                logger.info("Research with reflection completed successfully")
                return final_result

            except Exception as e:
                # Record task failure
                self.performance_monitor.record_task_error()
                logger.error(f"Research with reflection failed: {e}")
                raise

    async def _process_research_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process research configuration"""
        # Merge default configuration and user configuration
        processed_config = self.unified_config.__dict__.copy()
        processed_config.update(config)
        return processed_config

    async def _preprocess_research_result(self, research_result: Any) -> Any:
        """Preprocess research results"""
        # Can add result preprocessing logic here
        return research_result

    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up EnhancedResearcher resources")

        try:
            # Clean up MCP connections
            if self.mcp_manager:
                # Can add MCP connection cleanup logic here
                pass

            # Clean up other resources
            self.research_tools = None
            self.mcp_tools = None
            self.reflection_agent = None
            self.reflection_integrator = None

            logger.info("EnhancedResearcher cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
