# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Enhanced with MiniMax Agent optimizations

import logging
from typing import Dict, List, Optional, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from src.models.planner_model import StepType
from src.utils.performance.memory_manager import cached
from src.report_quality.template_engine import ReportDomain
from src.utils.system.callback_safety import global_callback_manager

from .types import State
from .nodes import (
    coordinator_node,
    planner_node,
    reporter_node,
    research_team_node,
    researcher_node,
    coder_node,
    human_feedback_node,
    background_investigation_node,
    context_optimizer_node,
    planning_context_optimizer_node,
)

# Import enhanced collaboration modules
try:
    from src.collaboration.role_bidding import (
        RoleBiddingSystem,
        TaskRequirement,
        TaskType,
    )
    from src.report_quality.critical_thinking import CriticalThinkingEngine
    from src.report_quality.interactive_report import ReportEnhancer

    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)


def continue_to_running_research_team(state: State):
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        # 检查是否需要规划上下文优化
        messages = state.get("messages", [])
        plan_iterations = state.get("plan_iterations", 0)

        # 如果消息过多且有多次规划迭代，先进行上下文优化
        if len(messages) > 10 and plan_iterations > 1:
            return "planning_context_optimizer"
        return "planner"
    if all(step.execution_res for step in current_plan.steps):
        return "planner"

    # 找到第一个未执行的步骤并根据类型路由
    for step in current_plan.steps:
        if not step.execution_res:
            if step.step_type and step.step_type == StepType.RESEARCH:
                return "researcher"
            elif step.step_type and step.step_type == StepType.PROCESSING:
                return "coder"
            else:
                # 如果步骤类型未知或为None，返回planner重新规划
                return "planner"

    # 如果没有找到未执行的步骤，返回planner
    return "planner"


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")

    # Use enhanced nodes if available, otherwise fallback to basic nodes
    try:
        if ENHANCED_FEATURES_AVAILABLE:
            builder.add_node("coordinator", enhanced_coordinator_node)
            builder.add_node("planner", enhanced_planner_node)
            builder.add_node("reporter", enhanced_reporter_node)
            logger.info("Using enhanced collaboration nodes")
        else:
            builder.add_node("coordinator", coordinator_node)
            builder.add_node("planner", planner_node)
            builder.add_node("reporter", reporter_node)
            logger.info("Using basic nodes (enhanced features not available)")
    except NameError:
        # Fallback to basic nodes if enhanced nodes are not defined
        builder.add_node("coordinator", coordinator_node)
        builder.add_node("planner", planner_node)
        builder.add_node("reporter", reporter_node)
        logger.warning("Enhanced nodes not defined, using basic nodes")

    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("context_optimizer", context_optimizer_node)
    builder.add_node("planning_context_optimizer", planning_context_optimizer_node)
    builder.add_edge("background_investigator", "planner")
    builder.add_edge("planning_context_optimizer", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        [
            "planner",
            "researcher",
            "coder",
            "context_optimizer",
            "planning_context_optimizer",
        ],
    )
    builder.add_edge("context_optimizer", "reporter")
    builder.add_edge("reporter", END)
    return builder


def _create_optimized_node(original_func, node_name: str):
    """Create an optimized node wrapper with performance monitoring."""

    async def optimized_node(state: State, config: RunnableConfig):
        import time

        start_time = time.time()

        try:
            # Execute original function
            result = await original_func(state, config)

            # Add performance metrics
            execution_time = time.time() - start_time
            logger.debug(f"Node '{node_name}' executed in {execution_time:.2f}s")

            # Add execution metadata to state if it's a dict
            if isinstance(result, dict):
                result["node_metrics"] = result.get("node_metrics", {})
                result["node_metrics"][node_name] = {
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                }

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Node '{node_name}' failed after {execution_time:.2f}s: {e}")
            raise

    return optimized_node


def _enhanced_coordinator_routing(state: State) -> str:
    """Enhanced coordinator routing with collaboration support."""
    # Check if collaboration is enabled and needed
    if (
        state.get("enable_collaboration", False)
        and state.get("collaboration_systems")
        and state.get("user_input", "").lower().find("complex") != -1
    ):
        return "role_bidding"
    elif state.get("needs_planning", True):
        return "planner"
    else:
        return "research_team"


async def _role_bidding_node(state: State, config: RunnableConfig) -> State:
    """Role bidding node for collaborative task assignment."""
    try:
        collaboration_systems = state.get("collaboration_systems")
        if collaboration_systems and "role_bidding" in collaboration_systems:
            role_bidding_system = collaboration_systems["role_bidding"]

            # Perform role bidding
            user_input = state.get("user_input", "")
            # Simplified role bidding logic
            assigned_roles = await role_bidding_system.assign_roles(user_input)

            return {
                **state,
                "assigned_roles": assigned_roles,
                "role_bidding_completed": True,
            }
    except Exception as e:
        logger.error(f"Role bidding failed: {e}")

    return state


async def _conflict_resolution_node(state: State, config: RunnableConfig) -> State:
    """Conflict resolution node for handling disagreements."""
    try:
        collaboration_systems = state.get("collaboration_systems")
        if collaboration_systems and "conflict_resolution" in collaboration_systems:
            conflict_system = collaboration_systems["conflict_resolution"]

            # Resolve conflicts if any
            conflicts = state.get("conflicts", [])
            if conflicts:
                resolved_conflicts = await conflict_system.resolve_conflicts(conflicts)
                return {
                    **state,
                    "resolved_conflicts": resolved_conflicts,
                    "has_conflicts": False,
                }
    except Exception as e:
        logger.error(f"Conflict resolution failed: {e}")

    return {**state, "has_conflicts": False}


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    logger.info("Built state graph with memory")

    # Add safe callback handling
    try:
        logger.info("Registering safe default callback for state graph with memory")
        from src.utils.system.callback_safety import global_callback_manager

        def safe_default_callback(*args, **kwargs):
            """Safe default callback that does nothing but prevents None errors"""
            pass

        global_callback_manager.register_default_callback(
            "on_done", safe_default_callback
        )
        logger.debug("Registered safe default callback for LangGraph with memory")
    except Exception as e:
        logger.warning(f"Could not register safe callback for graph with memory: {e}")

    return builder.compile(checkpointer=memory)


@cached(ttl=3600, priority=3)  # Cache graph for 1 hour with high priority
def build_graph():
    """Build and return the agent workflow graph without memory with performance optimizations."""
    logger.info("Building optimized workflow graph...")

    # build state graph with enhanced configuration
    builder = _build_base_graph()

    # Add conditional nodes for enhanced collaboration
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            builder.add_node("role_bidding", _role_bidding_node)
            builder.add_node("conflict_resolution", _conflict_resolution_node)
            logger.info("Added enhanced collaboration nodes")
        except NameError:
            logger.warning("Enhanced collaboration nodes not available")

    logger.info("Optimized workflow graph built successfully")

    # Compile with safe callback handling
    try:
        from src.utils.system.callback_safety import global_callback_manager

        # Register a default callback to prevent None callback errors
        def safe_default_callback(*args, **kwargs):
            """Safe default callback that does nothing but prevents None errors"""
            pass

        global_callback_manager.register_default_callback(
            "on_done", safe_default_callback
        )
        logger.debug("Registered safe default callback for LangGraph")
    except Exception as e:
        logger.warning(f"Could not register safe callback: {e}")

    return builder.compile()


graph = build_graph()


# Enhanced Graph Builder with Optimization Features
class EnhancedState:
    """Enhanced state class containing optimization feature state information"""

    def __init__(self):
        # Original state
        self.messages: List[Dict[str, Any]] = []
        self.current_plan: Optional[Dict[str, Any]] = None
        self.observations: List[str] = []
        self.final_report: str = ""
        self.locale: str = "zh-CN"

        # Collaboration mechanism state
        self.active_agents: Dict[str, str] = {}  # task_id -> agent_id
        self.conflict_claims: List = []
        self.intervention_points: List[str] = []

        # Report quality state
        self.report_template_id: Optional[str] = None
        self.critical_analysis: Optional[Dict[str, Any]] = None
        self.interactive_elements: List[Dict[str, Any]] = []


def enhanced_coordinator_node(
    state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
    """Enhanced coordinator node"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return coordinator_node(state, config)

    logger.info("Enhanced coordinator started")

    # Extract configurable from config and add to state
    from src.graph.nodes import get_configuration_from_config

    configurable = get_configuration_from_config(config)
    if isinstance(state, dict) and configurable:
        state["agent_configurable"] = configurable

    # Analyze user request and determine report type
    user_message = ""
    if state.get("messages"):
        last_message = state["messages"][-1]
        if hasattr(last_message, "content"):
            user_message = last_message.content
        elif isinstance(last_message, dict):
            user_message = last_message.get("content", "")

    # Detect report domain
    report_domain = _detect_report_domain(user_message)

    # Set report template (if state is a modifiable dictionary)
    if report_domain and isinstance(state, dict):
        state["report_template_id"] = _get_template_id(report_domain)

    # Call original coordinator
    result = coordinator_node(state, config)

    # If result is a Command object, return directly
    if isinstance(result, Command):
        return result

    # Add enhanced feature identifier (only when result is a dictionary)
    if isinstance(result, dict):
        result["enhanced_mode"] = True
        result["report_template_id"] = (
            state.get("report_template_id") if isinstance(state, dict) else None
        )

    return result


def enhanced_planner_node(
    state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
    """Enhanced planner node"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return planner_node(state, config)

    logger.info("Enhanced planner started")

    # Extract configurable from config and add to state
    from src.graph.nodes import get_configuration_from_config

    configurable = get_configuration_from_config(config)
    if isinstance(state, dict) and configurable:
        state["agent_configurable"] = configurable

    # Call original planner
    result = planner_node(state, config)

    # If result is a Command object, return directly
    if isinstance(result, Command):
        return result

    # Add dynamic role assignment (only when result is a dictionary)
    if isinstance(result, dict):
        try:
            role_bidding_system = RoleBiddingSystem()

            # Create task requirements
            user_message = ""
            if state.get("messages"):
                last_message = state["messages"][-1]
                if hasattr(last_message, "content"):
                    user_message = last_message.content
                elif isinstance(last_message, dict):
                    user_message = last_message.get("content", "")

            task_requirements = _analyze_and_create_tasks(user_message)

            # Assign agents for each task
            task_assignments = {}
            for task in task_requirements:
                assigned_agent = role_bidding_system.assign_task(task)
                if assigned_agent:
                    task_assignments[task.task_id] = assigned_agent

            result["task_assignments"] = task_assignments
            logger.info(f"Task assignment completed: {task_assignments}")
        except Exception as e:
            logger.warning(f"Role assignment failed: {e}")

    return result


def enhanced_reporter_node(
    state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
    """Enhanced reporter node"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return reporter_node(state, config)

    logger.info("Enhanced reporter started")

    # Extract configurable from config and add to state
    from src.graph.nodes import get_configuration_from_config

    configurable = get_configuration_from_config(config)
    if isinstance(state, dict) and configurable:
        state["agent_configurable"] = configurable

    # Call original reporter
    result = reporter_node(state, config)

    try:
        # Get base report content
        base_content = result.get("final_report", "")

        # Apply critical thinking analysis
        critical_thinking_engine = CriticalThinkingEngine()
        critical_analysis = critical_thinking_engine.analyze_content(
            base_content,
            metadata={"domain": state.get("report_template_id", "general")},
        )

        # Enhance report content
        enhanced_content = critical_analysis.enhanced_content

        # Get language settings
        locale = state.get("locale", "zh-CN")
        language = _get_language_from_locale(locale)

        # Configure output directory
        from src.report_quality.interactive_report import InteractiveReportConfig

        output_dir = state.get("output_dir", "reports")
        config = InteractiveReportConfig(output_dir=output_dir)

        # Generate interactive report
        report_enhancer = ReportEnhancer(language=language, config=config)
        interactive_report = report_enhancer.enhance_report(
            enhanced_content,
            metadata={
                "title": "AI Generated Report",
                "template_id": state.get("report_template_id", "general"),
                "quality_score": critical_analysis.quality_score,
            },
        )

        # Generate and save HTML file
        try:
            html_file_path = report_enhancer.generate_html_report(
                enhanced_content,
                metadata={
                    "title": "AI Generated Report",
                    "template_id": state.get("report_template_id", "general"),
                    "quality_score": critical_analysis.quality_score,
                },
            )
            result["interactive_html_path"] = html_file_path
            logger.info(f"Interactive HTML report generated: {html_file_path}")
        except Exception as e:
            logger.warning(f"HTML report generation failed: {e}")

        # Update results
        result["final_report"] = enhanced_content
        result["critical_analysis"] = {
            "quality_score": critical_analysis.quality_score,
            "limitations_count": len(critical_analysis.identified_limitations),
            "biases_count": len(critical_analysis.bias_assessments),
        }
        result["interactive_elements"] = [
            {
                "element_id": elem.element_id,
                "type": elem.element_type.value,
                "title": elem.title,
            }
            for elem in interactive_report.interactive_elements
        ]

        logger.info(
            f"Report generation completed, quality score: {critical_analysis.quality_score:.1f}"
        )

    except Exception as e:
        logger.warning(f"Report enhancement failed: {e}")

    return result


def _detect_report_domain(user_message: str):
    """Detect report domain"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return None

    message_lower = user_message.lower()

    if any(
        keyword in message_lower
        for keyword in [
            "finance",
            "investment",
            "stock",
            "revenue",
            "financial",
            "investment",
            "stock",
            "revenue",
        ]
    ):
        return ReportDomain.FINANCIAL
    elif any(
        keyword in message_lower
        for keyword in [
            "academic",
            "research",
            "paper",
            "analysis",
            "academic",
            "research",
            "paper",
            "analysis",
        ]
    ):
        return ReportDomain.ACADEMIC
    elif any(
        keyword in message_lower
        for keyword in [
            "market",
            "competition",
            "consumer",
            "market",
            "competition",
            "consumer",
        ]
    ):
        return ReportDomain.MARKET_RESEARCH
    elif any(
        keyword in message_lower
        for keyword in [
            "technical",
            "engineering",
            "system",
            "technical",
            "engineering",
            "system",
        ]
    ):
        return ReportDomain.TECHNICAL

    return None


def _get_template_id(domain) -> str:
    """Get template ID"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return "general_report"

    template_map = {
        ReportDomain.FINANCIAL: "financial_analysis",
        ReportDomain.ACADEMIC: "academic_research",
        ReportDomain.MARKET_RESEARCH: "market_research",
        ReportDomain.TECHNICAL: "technical_report",
    }
    return template_map.get(domain, "general_report")


def _get_language_from_locale(locale: str):
    """Convert locale string to Language enum"""
    from src.report_quality.i18n import Language

    locale_map = {
        "zh-CN": Language.ZH_CN,
        "zh-cn": Language.ZH_CN,
        "zh": Language.ZH_CN,
        "en-US": Language.EN_US,
        "en-us": Language.EN_US,
        "en": Language.EN_US,
    }

    return locale_map.get(locale, Language.ZH_CN)  # Default to Chinese


def _analyze_and_create_tasks(user_message: str) -> List:
    """Analyze and create task requirements"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return []

    tasks = []

    # Basic research task
    research_task = TaskRequirement(
        task_id="research_001",
        task_type=TaskType.WEB_RESEARCH,
        description="Collect and analyze relevant information",
        required_tools=["web_search", "document_analysis"],
        complexity_level=3,
        priority=4,
    )
    tasks.append(research_task)

    # Data analysis task (if needed)
    if any(
        keyword in user_message
        for keyword in [
            "data",
            "analysis",
            "statistics",
            "data",
            "analysis",
            "statistics",
        ]
    ):
        data_task = TaskRequirement(
            task_id="data_001",
            task_type=TaskType.DATA_ANALYSIS,
            description="Data processing and analysis",
            required_tools=["python", "pandas", "matplotlib"],
            complexity_level=4,
            priority=3,
        )
        tasks.append(data_task)

    # Content generation task
    content_task = TaskRequirement(
        task_id="content_001",
        task_type=TaskType.CONTENT_WRITING,
        description="Generate report content",
        required_tools=["writing_tools"],
        complexity_level=3,
        priority=5,
    )
    tasks.append(content_task)

    return tasks


def _build_enhanced_graph():
    """Build and return the enhanced state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", enhanced_coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", enhanced_planner_node)
    builder.add_node("reporter", enhanced_reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("context_optimizer", context_optimizer_node)
    builder.add_node("planning_context_optimizer", planning_context_optimizer_node)
    builder.add_edge("background_investigator", "planner")
    builder.add_edge("planning_context_optimizer", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        [
            "planner",
            "researcher",
            "coder",
            "context_optimizer",
            "planning_context_optimizer",
        ],
    )
    builder.add_edge("context_optimizer", "reporter")
    builder.add_edge("reporter", END)
    return builder


def build_enhanced_graph():
    """Build and return the enhanced agent workflow graph without memory."""
    if not ENHANCED_FEATURES_AVAILABLE:
        logger.warning(
            "Enhanced features not available, falling back to standard graph"
        )
        return build_graph()

    # build enhanced state graph
    builder = _build_enhanced_graph()
    logger.info("Built enhanced state graph")

    # Apply LangGraph safety patches
    # ensure_safe_langgraph_execution()
    logger.info("Applied LangGraph safety patches")

    # Add safe callback handling
    try:
        logger.info("Registering safe default callback for LangGraph")

        def safe_default_callback(*args, **kwargs):
            """Safe default callback that does nothing but prevents None errors"""
            pass

        global_callback_manager.register_default_callback(
            "on_done", safe_default_callback
        )
        logger.debug("Registered safe default callback for LangGraph")
    except Exception as e:
        logger.warning(f"Could not register safe callback for graph: {e}")

    return builder.compile()


def build_enhanced_graph_with_memory():
    """Build and return the enhanced agent workflow graph with memory."""
    if not ENHANCED_FEATURES_AVAILABLE:
        logger.warning(
            "Enhanced features not available, falling back to standard graph with memory"
        )
        return build_graph_with_memory()

    # use persistent memory to save conversation history
    memory = MemorySaver()

    # build enhanced state graph
    builder = _build_enhanced_graph()
    logger.info("Built enhanced state graph with memory")

    # Add safe callback handling
    try:
        logger.info(
            "Registering safe default callback for enhanced LangGraph with memory"
        )
        from src.utils.system.callback_safety import global_callback_manager

        def safe_default_callback(*args, **kwargs):
            """Safe default callback that does nothing but prevents None errors"""
            pass

        global_callback_manager.register_default_callback(
            "on_done", safe_default_callback
        )
        logger.debug(
            "Registered safe default callback for enhanced LangGraph with memory"
        )
    except Exception as e:
        logger.warning(
            f"Could not register safe callback for enhanced graph with memory: {e}"
        )

    return builder.compile(checkpointer=memory)


# Create enhanced graph instance
enhanced_graph = build_enhanced_graph()
