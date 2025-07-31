"""Enhanced Researcher Main Class - Integrates all managers and provides unified research interface"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from src.agents import create_agent
from src.llms.error_handler import safe_llm_call_async
from src.utils.context.execution_context_manager import (
    ExecutionContextManager,
    ContextConfig,
)

from .isolation_config_manager import IsolationConfigManager
from .research_tool_manager import ResearchToolManager
from .mcp_client_manager import MCPClientManager
from .reflection_system_manager import ReflectionSystemManager
from src.utils.reflection.enhanced_reflection import ReflectionContext

# IterativeResearchEngine functionality merged into unified reflection system
from .research_result_processor import ResearchResultProcessor
from .performance_monitor import global_performance_tracker
from .config_cache_optimizer import global_config_cache_optimizer
from .concurrent_optimizer import (
    TaskPriority,
    global_concurrent_manager,
)
from src.models.planner_model import Step, StepType

logger = logging.getLogger(__name__)


class EnhancedResearcher:
    """Enhanced Researcher - Main class that unifies management of all research components"""

    def __init__(self, state: Dict[str, Any], config: Optional[Any] = None):
        """Initialize Enhanced Researcher

        Args:
            state: Research state dictionary
            config: RunnableConfig object containing configuration
        """
        self.state = state
        self.config = config
        self.initialization_time = datetime.now()

        # Extract the full context needed for the prompt
        self.research_topic = state.get("research_topic", "")
        self.resources = state.get("resources", [])

        # Extract current step from the plan
        current_plan = state.get("current_plan")
        self.current_step_title = ""
        self.current_step_description = ""

        if current_plan and hasattr(current_plan, "steps"):
            for step in current_plan.steps:
                if not step.execution_res:
                    self.current_step_title = step.title
                    self.current_step_description = step.description
                    break

        # Pre-build resources_info
        self.resources_info = ""
        if self.resources:
            resources_text_list = []
            for r in self.resources:
                if isinstance(r, str):
                    # Handle string resources
                    resources_text_list.append(f"- {r}")
                elif hasattr(r, "title") and hasattr(r, "description"):
                    # Handle resource objects with title and description
                    resources_text_list.append(f"- {r.title} ({r.description})")
                else:
                    # Handle other types by converting to string
                    resources_text_list.append(f"- {str(r)}")

            resources_text = "\n".join(resources_text_list)
            self.resources_info = (
                "**The user mentioned the following resource files:**\n\n"
                f"{resources_text}\n\n"
                "You MUST use the **local_search_tool** to retrieve the information from the resource files."
            )

        # Extract configurable from config if provided, otherwise create default
        if config:
            from src.graph.nodes import get_configuration_from_config

            configurable = get_configuration_from_config(config)
        else:
            # Create a simple configurable object for backward compatibility
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
            configurable.reflection.model = "basic"

            configurable.mcp = SimpleNamespace()
            configurable.mcp.enabled = False
            configurable.mcp.servers = []

            configurable.enabled = True
            configurable.max_loops = 2
            configurable.quality_threshold = 0.75
            # Reflection integration is now controlled by enabled

        self.configurable = configurable

        # Initialize all managers
        self.config_manager = IsolationConfigManager(state, config)
        self.unified_config = self.config_manager.get_unified_config()

        self.tool_manager = ResearchToolManager(self.configurable, state)
        self.mcp_manager = MCPClientManager(self.config_manager)
        self.reflection_manager = ReflectionSystemManager(self.unified_config)
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

    def _prepare_research_tools(self) -> list:
        """Prepare research tools for the researcher agent

        Returns:
            List of tools for the researcher
        """
        from src.tools.search import get_web_search_tool
        from src.tools.crawl import crawl_tool
        from src.tools.retriever import get_retriever_tool

        # Set up standard research tools
        tools = [
            get_web_search_tool(
                self.configurable.agents.max_search_results,
                self.configurable.content.enable_smart_filtering,
            ),
            crawl_tool,
        ]

        # Add retriever tool if resources are available
        retriever_tool = get_retriever_tool(self.resources)
        if retriever_tool:
            tools.insert(0, retriever_tool)

        logger.info(f"Prepared {len(tools)} research tools")
        return tools

    def _build_agent_state(self) -> Dict[str, Any]:
        """Build agent state for researcher execution

        This method consolidates the agent state building logic from nodes.py

        Returns:
            Complete agent state dictionary
        """
        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        current_plan = self.state.get("current_plan")
        # observations = self.state.get("observations", [])  # Not used currently

        # Find the first unexecuted step
        current_step = None
        completed_steps = []
        if current_plan and hasattr(current_plan, "steps"):
            for step in current_plan.steps:
                if not step.execution_res:
                    current_step = step
                    break
                else:
                    completed_steps.append(
                        {
                            "step": step.title,
                            "description": step.description,
                            "execution_res": step.execution_res,
                        }
                    )

        if not current_step:
            logger.warning("No unexecuted step found")
            return {}

        # Use unified config for context management
        config_manager = IsolationConfigManager(self.configurable)
        unified_config = config_manager.get_unified_config()

        context_config = ContextConfig(
            max_context_steps=unified_config.max_context_steps_parallel,
            max_step_content_length=2000,
            max_observations_length=10000,
            enable_content_deduplication=True,
            enable_smart_truncation=True,
        )

        # Check if context sharing should be disabled
        if unified_config.disable_context_parallel:
            context_config.max_context_steps = 0
            logger.info(
                "Context sharing disabled - no historical context will be included"
            )

        context_manager = ExecutionContextManager(context_config)

        # Prepare optimized context using unified manager
        current_step_dict = {
            "step": current_step.title,
            "description": current_step.description,
        }

        optimized_steps, context_info = context_manager.prepare_context_for_execution(
            completed_steps, current_step_dict, "researcher"
        )

        # Prepare the agent state with current task information
        agent_state = {
            **self.state,  # Include all existing state
            "current_step_title": current_step.title,
            "current_step_description": current_step.description,
            "context_info": context_info,
            "completed_steps": optimized_steps,
            "agent_name": "researcher",
            "original_user_query": self.research_topic,
        }

        agent_state["messages"] = [HumanMessage(content=current_step.description)]

        # Add resources information for researcher agent
        if self.resources:
            agent_state["resources_info"] = self.resources_info

            # Add citation reminder
            agent_state["citation_reminder"] = (
                "IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)"
            )

        return agent_state

    async def execute(self) -> Dict[str, Any]:
        """Unified execute method for researcher agent

        This method serves as the main entry point for research execution,
        integrating tool preparation, context building, and agent execution.

        Returns:
            Command result dictionary
        """
        import os
        import traceback

        logger.info("Starting unified researcher execution")

        try:
            # Step 0: Initialize research components (including reflection system)
            await self.initialize_research_components()

            # Step 1: Prepare research tools
            tools = self._prepare_research_tools()

            # Step 2: Build agent state
            agent_state = self._build_agent_state()

            if not agent_state:
                logger.warning("No agent state built, returning to research_team")
                return Command(goto="research_team")

            # Step 3: Create researcher agent with tools
            agent = create_agent(
                agent_name="researcher",
                agent_type="researcher",
                tools=tools,
                prompt_template="researcher",
                configurable=self.configurable,
            )

            # Step 4: Execute agent with recursion limit
            default_recursion_limit = 10
            try:
                env_value_str = os.getenv(
                    "AGENT_RECURSION_LIMIT", str(default_recursion_limit)
                )
                parsed_limit = int(env_value_str)

                if parsed_limit > 0:
                    recursion_limit = parsed_limit
                    logger.info(f"Recursion limit set to: {recursion_limit}")
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
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                recursion_limit = default_recursion_limit

            logger.info(
                f"Agent state prepared for researcher with step: {agent_state.get('current_step_title', 'Unknown')}"
            )

            # Execute the agent
            result = await safe_llm_call_async(
                agent.ainvoke,
                input=agent_state,
                config={"recursion_limit": recursion_limit},
                operation_name="researcher executor",
                context=f"Execute step: {agent_state.get('current_step_title', 'Unknown')}",
                enable_smart_processing=True,
                max_retries=3,
            )

            # Step 5: Process the result
            if hasattr(result, "get") and "messages" in result:
                response_content = result["messages"][-1].content
            else:
                response_content = (
                    result.content if hasattr(result, "content") else str(result)
                )

            logger.debug(f"Researcher full response: {response_content}")

            # Step 6: Update observations and plan
            current_plan = self.state.get("current_plan")
            observations = self.state.get("observations", [])

            # Find current step
            current_step = None
            if current_plan and hasattr(current_plan, "steps"):
                for step in current_plan.steps:
                    if not step.execution_res:
                        current_step = step
                        break

            if current_step:
                # Use advanced observation management
                context_config = self.config_manager.get_unified_config()
                context_manager = ExecutionContextManager(
                    {
                        "max_context_steps": context_config.max_context_steps_parallel,
                        "max_step_content_length": 2000,
                        "max_observations_length": 10000,
                        "enable_content_deduplication": True,
                        "enable_smart_truncation": True,
                    }
                )

                new_observation = f"Step: {current_step.title}\nAgent: researcher\nResult: {response_content}"
                optimized_observations = context_manager.manage_observations_advanced(
                    observations + [new_observation], optimization_level="standard"
                )

                # Create updated step with execution result
                updated_step = current_step.copy(
                    update={"execution_res": response_content}
                )

                # Create updated plan with the new step
                updated_steps = list(current_plan.steps)
                step_index = next(
                    (
                        i
                        for i, s in enumerate(current_plan.steps)
                        if s.title == current_step.title
                    ),
                    None,
                )
                if step_index is not None:
                    updated_steps[step_index] = updated_step

                updated_plan = current_plan.copy(update={"steps": updated_steps})
                logger.info(
                    f"Step '{current_step.title}' execution completed by researcher"
                )

                # Step 6.5: (如果启用) 对研究结果执行反射分析
                reflection_insights = None
                # 检查全局开关和当前 agent_state 是否允许反射
                if getattr(self.configurable.reflection, "enabled", False):
                    # 检查反射循环次数是否已达到限制
                    current_reflection_count = self.state.get("reflection", {}).get(
                        "count", 0
                    )
                    max_reflection_loops = getattr(
                        self.configurable.reflection, "max_loops", 2
                    )

                    if current_reflection_count >= max_reflection_loops:
                        logger.info(
                            f"Reflection loop limit reached ({current_reflection_count}/{max_reflection_loops}). Skipping reflection analysis."
                        )
                        reflection_insights = {
                            "is_sufficient": True,
                            "reason": (
                                f"Maximum reflection loops ({max_reflection_loops}) reached"
                            ),
                            "follow_up_queries": [],
                        }
                    else:
                        logger.info(
                            f"Executing reflection analysis on research result (loop {current_reflection_count + 1}/{max_reflection_loops})."
                        )
                        try:
                            # 构建反射所需的上下文
                            reflection_context = self._build_reflection_context(
                                response_content
                            )

                            # 调用反射系统
                            reflection_insights = await self.reflection_manager.execute_reflection_analysis(
                                reflection_context
                            )
                            logger.info("Reflection analysis completed.")

                        except Exception as e:
                            logger.error(
                                f"Reflection analysis failed: {e}", exc_info=True
                            )
                            # 关键：即使反射失败，也应继续执行，不应中断主流程，只记录错误。
                            reflection_insights = {
                                "error": f"Reflection failed: {str(e)}"
                            }

                # Step 7: Process reflection insights and decide next action
                if (
                    reflection_insights
                    and hasattr(reflection_insights, "is_sufficient")
                    and not reflection_insights.is_sufficient
                    and hasattr(reflection_insights, "follow_up_queries")
                    and reflection_insights.follow_up_queries
                ):
                    logger.info(
                        "Reflection identified knowledge gaps. Adding new research steps."
                    )

                    # 根据反思结果中的 follow_up_queries 创建新的研究步骤
                    new_steps = []
                    for query in reflection_insights.follow_up_queries:
                        new_steps.append(
                            Step(
                                need_search=True,
                                title=f"Follow-up Research: {query}",
                                description=f"Based on reflection analysis, further research is needed on: {query}",
                                step_type=StepType.RESEARCH,
                            )
                        )

                    # 将新步骤添加到现有计划中
                    updated_steps.extend(new_steps)
                    updated_plan_with_reflection = updated_plan.copy(
                        update={"steps": updated_steps}
                    )

                    # 将反思结果添加到观察记录中（用于调试和追踪）
                    new_observation = {
                        "type": "reflection",
                        "timestamp": datetime.now().isoformat(),
                        "content": reflection_insights,
                        "source": "enhanced_reflection_system",
                    }
                    optimized_observations.append(new_observation)

                    # 更新反射计数器
                    current_reflection_count = self.state.get("reflection", {}).get(
                        "count", 0
                    )
                    updated_reflection_state = self.state.get("reflection", {})
                    updated_reflection_state["count"] = current_reflection_count + 1

                    # 返回指令，更新计划并重新回到 research_team 执行新步骤
                    return Command(
                        update={
                            "messages": [
                                AIMessage(content=response_content, name="researcher")
                            ],
                            "observations": optimized_observations,
                            "current_plan": updated_plan_with_reflection,
                            "research_topic": self.state.get("research_topic", ""),
                            "resources": self.state.get("resources", []),
                            "locale": self.state.get("locale", "en-US"),
                            "reflection": updated_reflection_state,
                        },
                        goto="research_team",
                    )
                else:
                    # 如果研究是充分的，或没有后续问题，则按原逻辑继续
                    update_dict = {
                        "messages": [
                            AIMessage(content=response_content, name="researcher")
                        ],
                        "observations": optimized_observations,
                        "current_plan": updated_plan,
                        "research_topic": self.state.get("research_topic", ""),
                        "resources": self.state.get("resources", []),
                        "locale": self.state.get("locale", "en-US"),
                    }

                    if reflection_insights:
                        new_observation = {
                            "type": "reflection",
                            "timestamp": datetime.now().isoformat(),
                            "content": reflection_insights,
                            "source": "enhanced_reflection_system",
                        }
                        update_dict["observations"].append(new_observation)

                    # 正常结束当前研究循环，将结果交给 planner 判断
                    return Command(update=update_dict, goto="research_team")
            else:
                logger.warning("No current step found for execution")
                return Command(goto="research_team")

        except Exception as e:
            logger.error(f"Error in unified researcher execution: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Return error observation
            return Command(
                update={
                    "observations": [{"error": "Research failed", "details": str(e)}],
                    "research_topic": self.state.get("research_topic", ""),
                    "resources": self.state.get("resources", []),
                },
                goto="research_team",
            )

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
            self.research_tools = self.tool_manager.get_all_tools()
            initialization_results["tools_initialized"] = True
            logger.info(f"Initialized {len(self.research_tools)} research tools")

        except Exception as e:
            error_msg = f"Failed to initialize research tools: {e}"
            logger.error(error_msg)
            initialization_results["errors"].append(error_msg)

        try:
            # Initialize MCP tools
            if self.unified_config.mcp_enabled:
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

        # Research session tracking (moved from research_engine)
        self.iteration_count = 0
        self.research_history = []

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

            # --- Start of new code ---
            # Build the full prompt context (commented out as not currently used)
            # prompt_context = {
            #     "original_user_query": self.research_topic,
            #     "resources_info": self.resources_info,
            #     "current_step_title": self.current_step_title,
            #     "current_step_description": self.current_step_description,
            # }
            # --- End of new code ---

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
                self.iteration_count += 1
                iteration = self.iteration_count

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
                        self.research_history.append(
                            {
                                "query": query,
                                "result": search_result,
                                "timestamp": datetime.now(),
                            }
                        )

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
                    self._check_termination_conditions(
                        iteration, iteration_result.get("reflection_result")
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
                    follow_up_queries = self._generate_follow_up_queries(
                        iteration_result["reflection_result"]
                    )
                    valid_queries = self._filter_valid_queries(follow_up_queries)
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

        # --- Start of new code ---
        # Build enhanced query with full context
        enhanced_query = self._build_enhanced_query(query)
        logger.debug(f"Executing enhanced search query: {enhanced_query}")
        # --- End of new code ---

        # Execute search
        if hasattr(search_tool, "search"):
            result = await search_tool.search(enhanced_query)  # Use enhanced query
        elif hasattr(search_tool, "run"):
            result = await search_tool.run(enhanced_query)  # Use enhanced query
        elif hasattr(search_tool, "__call__"):
            result = await search_tool(enhanced_query)  # Use enhanced query
        else:
            raise RuntimeError(f"Search tool {search_tool.name} has no callable method")

        return result

    def _build_enhanced_query(self, query: str) -> str:
        """Build enhanced query with full context

        Args:
            query: Original query string

        Returns:
            Enhanced query with context
        """
        # --- Start of new code ---
        # Build the enhanced query with full context, similar to _execute_agent_step
        enhanced_parts = []

        # Add original user query context
        if self.research_topic:
            enhanced_parts.append(f"Original user query: {self.research_topic}")

        # Add current step context
        if self.current_step_title:
            enhanced_parts.append(f"Current step: {self.current_step_title}")
            if self.current_step_description:
                enhanced_parts.append(
                    f"Step description: {self.current_step_description}"
                )

        # Add resources info
        if self.resources_info:
            enhanced_parts.append(self.resources_info)

        # Add the actual research query
        enhanced_parts.append(f"Research query: {query}")

        # Combine all parts
        enhanced_query = "\n\n".join(enhanced_parts)
        # --- End of new code ---

        return enhanced_query

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

    def _build_reflection_context_from_research(
        self, query: str, research_result: Any
    ) -> Any:
        """Build reflection context from query and research result

        Args:
            query: Research query
            research_result: Research result

        Returns:
            Reflection context object
        """
        try:
            from src.utils.reflection.enhanced_reflection import ReflectionContext

            # Extract relevant information from research result
            completed_steps = []
            execution_results = []

            if isinstance(research_result, dict):
                # Extract iterations if available
                iterations = research_result.get("iterations", [])
                for i, iteration in enumerate(iterations):
                    completed_steps.append(
                        {
                            "step": i + 1,
                            "action": f"Research iteration {i + 1}",
                            "queries": iteration.get("queries", []),
                            "results_count": len(iteration.get("search_results", [])),
                        }
                    )

                    # Add search results as execution results
                    for result in iteration.get("search_results", []):
                        if isinstance(result, dict) and "content" in result:
                            execution_results.append(
                                str(result["content"])[:500]
                            )  # Truncate for context
                        else:
                            execution_results.append(str(result)[:500])

            # Create reflection context
            context = ReflectionContext(
                research_topic=query,
                completed_steps=completed_steps,
                execution_results=execution_results,
                resources_found=len(execution_results),
                total_steps=len(completed_steps),
                current_step_index=len(completed_steps),
                locale=self.state.get("locale", "en-US"),
                max_loops=getattr(self.unified_config, "max_loops", 2),
                current_reflection_loop=0,
            )

            return context

        except ImportError:
            # Fallback to dictionary if ReflectionContext is not available
            logger.warning("ReflectionContext not available, using dictionary fallback")
            return {
                "research_topic": query,
                "completed_steps": [],
                "execution_results": (
                    [str(research_result)[:500]] if research_result else []
                ),
                "resources_found": 1 if research_result else 0,
                "total_steps": 1,
                "current_step_index": 1,
                "locale": self.state.get("locale", "en-US"),
                "max_reflection_loops": getattr(self.unified_config, "max_loops", 2),
                "current_reflection_loop": 0,
            }
        except Exception as e:
            logger.error(f"Failed to build reflection context: {e}")
            # Return minimal context
            return {
                "research_topic": query,
                "completed_steps": [],
                "execution_results": [],
                "resources_found": 0,
                "total_steps": 0,
                "current_step_index": 0,
                "locale": "en-US",
                "max_reflection_loops": 2,
                "current_reflection_loop": 0,
            }

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

                # Build prompt context for iterative research
                prompt_context = {
                    "original_user_query": self.research_topic,
                    "resources_info": self.resources_info,
                    "current_step_title": self.current_step_title,
                    "current_step_description": self.current_step_description,
                }

                # Use concurrent manager to execute iterative research
                self.performance_monitor.add_custom_metric(
                    "iterative_research", "started"
                )
                research_task = self.concurrent_manager.execute_async_task(
                    self.research_engine.execute_iterative_research(
                        query, isolation_context, prompt_context=prompt_context
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

                # Build reflection context from query and research result
                reflection_context = self._build_reflection_context_from_research(
                    query, research_result
                )
                reflection_coro = self.reflection_manager.execute_reflection_analysis(
                    reflection_context
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

    def _build_reflection_context(self, research_content: str) -> ReflectionContext:
        """为反射系统构建必要的上下文。

        Args:
            research_content: 研究结果内容

        Returns:
            反射上下文对象
        """
        # 构建 ReflectionContext 对象，包含所有必需的属性
        context = ReflectionContext(
            research_topic=self.research_topic,
            completed_steps=self._get_completed_steps_summary(),
            current_step=(
                {
                    "title": getattr(self, "current_step_title", ""),
                    "content": research_content,
                }
                if hasattr(self, "current_step_title")
                else None
            ),
            execution_results=[research_content],  # 将研究内容作为执行结果
            observations=[
                str(obs) for obs in self.state.get("observations", [])
            ],  # 确保observations是字符串列表
            resources_found=len(self.state.get("resources", [])),
            total_steps=getattr(self, "total_steps", 1),
            current_step_index=getattr(self, "current_step_index", 0),
            locale=getattr(self.configurable, "locale", "en-US"),
            max_loops=getattr(self.unified_config, "max_loops", 1),
            current_reflection_loop=0,
        )
        return context

    def _get_completed_steps_summary(self) -> list:
        """获取已完成步骤的摘要列表。

        Returns:
            已完成步骤的摘要列表
        """
        if not (plan := self.state.get("current_plan")):
            return []

        return [
            {
                "title": step.title,
                "result_summary": str(step.execution_res)[:200],
            }  # 截断以保持上下文简洁
            for step in plan.steps
            if step.execution_res
        ]

    def _check_termination_conditions(
        self, iteration: int, reflection_result: Any = None
    ) -> tuple[bool, str, Dict[str, Any]]:
        """检查迭代研究终止条件

        Args:
            iteration: 当前迭代次数
            reflection_result: 反射结果（可选）

        Returns:
            (should_terminate, termination_reason, decision_factors) 元组
        """
        decision_factors = {
            "current_iteration": iteration,
            "max_iterations": getattr(self.unified_config, "max_loops", 2),
            "has_reflection_result": reflection_result is not None,
        }

        # 检查最大迭代次数
        max_loops = getattr(self.unified_config, "max_loops", 2)
        if iteration >= max_loops:
            reason = f"Reached maximum iterations ({max_loops})"
            logger.info(f"Termination condition met: {reason}")
            return True, reason, decision_factors

        # 检查反射结果是否表明研究已充分
        if reflection_result and hasattr(reflection_result, "is_sufficient"):
            if reflection_result.is_sufficient:
                reason = "Reflection indicates research is sufficient"
                logger.info(f"Termination condition met: {reason}")
                decision_factors["reflection_sufficient"] = True
                decision_factors["confidence_score"] = getattr(
                    reflection_result, "confidence_score", None
                )
                return True, reason, decision_factors
            else:
                decision_factors["reflection_sufficient"] = False
                decision_factors["knowledge_gaps"] = len(
                    getattr(reflection_result, "knowledge_gaps", [])
                )

        # 继续研究
        logger.debug(
            f"Continuing research - iteration {iteration}/{getattr(self.unified_config, 'max_loops', 2)}"
        )
        return False, "Continue research", decision_factors

    def _generate_follow_up_queries(self, reflection_result: Any) -> list[str]:
        """基于反射结果生成后续查询

        Args:
            reflection_result: 反射分析结果

        Returns:
            后续查询列表
        """
        follow_up_queries = []

        # 从反射结果中提取知识缺口
        if hasattr(reflection_result, "knowledge_gaps"):
            knowledge_gaps = reflection_result.knowledge_gaps
            # 限制查询数量
            max_queries = getattr(
                self.unified_config, "reflection_max_followup_queries", 3
            )
            for gap in knowledge_gaps[:max_queries]:
                if isinstance(gap, str):
                    follow_up_queries.append(gap)
                elif hasattr(gap, "query") or hasattr(gap, "question"):
                    query = getattr(gap, "query", None) or getattr(
                        gap, "question", None
                    )
                    if query:
                        follow_up_queries.append(str(query))

        return follow_up_queries

    def _filter_valid_queries(self, queries: list[str]) -> list[str]:
        """过滤有效的查询

        Args:
            queries: 原始查询列表

        Returns:
            过滤后的有效查询列表
        """
        valid_queries = []

        for query in queries:
            # 基本验证：非空且长度合理
            if query and isinstance(query, str) and 5 <= len(query.strip()) <= 200:
                # 避免重复查询
                if query.strip() not in [q.strip() for q in valid_queries]:
                    valid_queries.append(query.strip())

        # 限制最大查询数量
        max_queries = getattr(self.unified_config, "reflection_max_followup_queries", 3)
        return valid_queries[:max_queries]
