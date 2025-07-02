# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Enhanced with MiniMax Agent optimizations

import logging
import asyncio
from typing import Dict, List, Optional, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from src.prompts.planner_model import StepType

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
)

# Import enhanced collaboration modules
try:
    from src.collaboration.role_bidding import RoleBiddingSystem, TaskRequirement, TaskType
    from src.collaboration.human_loop import HumanLoopController
    from src.collaboration.consensus_system import ConflictResolutionSystem, ConflictingClaim
    from src.report_quality.template_engine import ReportBuilder, ReportDomain
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
        return "planner"
    if all(step.execution_res for step in current_plan.steps):
        return "planner"
    for step in current_plan.steps:
        if not step.execution_res:
            break
    if step.step_type and step.step_type == StepType.RESEARCH:
        return "researcher"
    if step.step_type and step.step_type == StepType.PROCESSING:
        return "coder"
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
    builder.add_edge("background_investigator", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "coder"],
    )
    builder.add_edge("reporter", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
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


def enhanced_coordinator_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Enhanced coordinator node"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return coordinator_node(state, config)
    
    logger.info("Enhanced coordinator started")
    
    # Analyze user request and determine report type
    user_message = ""
    if state.get("messages"):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
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
        result["report_template_id"] = state.get("report_template_id") if isinstance(state, dict) else None
    
    return result


def enhanced_planner_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Enhanced planner node"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return planner_node(state, config)
    
    logger.info("Enhanced planner started")
    
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
                if hasattr(last_message, 'content'):
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


def enhanced_reporter_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Enhanced reporter node"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return reporter_node(state, config)
    
    logger.info("Enhanced reporter started")
    
    # Call original reporter
    result = reporter_node(state, config)
    
    try:
        # Get base report content
        base_content = result.get("final_report", "")
        
        # Apply critical thinking analysis
        critical_thinking_engine = CriticalThinkingEngine()
        critical_analysis = critical_thinking_engine.analyze_content(
            base_content,
            metadata={"domain": state.get("report_template_id", "general")}
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
                "quality_score": critical_analysis.quality_score
            }
        )
        
        # Generate and save HTML file
        try:
            html_file_path = report_enhancer.generate_html_report(
                enhanced_content,
                metadata={
                    "title": "AI Generated Report",
                    "template_id": state.get("report_template_id", "general"),
                    "quality_score": critical_analysis.quality_score
                }
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
            "biases_count": len(critical_analysis.bias_assessments)
        }
        result["interactive_elements"] = [
            {
                "element_id": elem.element_id,
                "type": elem.element_type.value,
                "title": elem.title
            }
            for elem in interactive_report.interactive_elements
        ]
        
        logger.info(f"Report generation completed, quality score: {critical_analysis.quality_score:.1f}")
        
    except Exception as e:
        logger.warning(f"Report enhancement failed: {e}")
    
    return result


def _detect_report_domain(user_message: str):
    """Detect report domain"""
    if not ENHANCED_FEATURES_AVAILABLE:
        return None
    
    message_lower = user_message.lower()
    
    if any(keyword in message_lower for keyword in ["finance", "investment", "stock", "revenue", "financial", "investment", "stock", "revenue"]):
        return ReportDomain.FINANCIAL
    elif any(keyword in message_lower for keyword in ["academic", "research", "paper", "analysis", "academic", "research", "paper", "analysis"]):
        return ReportDomain.ACADEMIC
    elif any(keyword in message_lower for keyword in ["market", "competition", "consumer", "market", "competition", "consumer"]):
        return ReportDomain.MARKET_RESEARCH
    elif any(keyword in message_lower for keyword in ["technical", "engineering", "system", "technical", "engineering", "system"]):
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
        ReportDomain.TECHNICAL: "technical_report"
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
        "en": Language.EN_US
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
        priority=4
    )
    tasks.append(research_task)
    
    # Data analysis task (if needed)
    if any(keyword in user_message for keyword in ["data", "analysis", "statistics", "data", "analysis", "statistics"]):
        data_task = TaskRequirement(
            task_id="data_001",
            task_type=TaskType.DATA_ANALYSIS,
            description="Data processing and analysis",
            required_tools=["python", "pandas", "matplotlib"],
            complexity_level=4,
            priority=3
        )
        tasks.append(data_task)
        
    # Content generation task
    content_task = TaskRequirement(
        task_id="content_001",
        task_type=TaskType.CONTENT_WRITING,
        description="Generate report content",
        required_tools=["writing_tools"],
        complexity_level=3,
        priority=5
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
    builder.add_edge("background_investigator", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "coder"],
    )
    builder.add_edge("reporter", END)
    return builder


def build_enhanced_graph():
    """Build and return the enhanced agent workflow graph without memory."""
    if not ENHANCED_FEATURES_AVAILABLE:
        logger.warning("Enhanced features not available, falling back to standard graph")
        return build_graph()
    
    # build enhanced state graph
    builder = _build_enhanced_graph()
    return builder.compile()


def build_enhanced_graph_with_memory():
    """Build and return the enhanced agent workflow graph with memory."""
    if not ENHANCED_FEATURES_AVAILABLE:
        logger.warning("Enhanced features not available, falling back to standard graph with memory")
        return build_graph_with_memory()
    
    # use persistent memory to save conversation history
    memory = MemorySaver()

    # build enhanced state graph
    builder = _build_enhanced_graph()
    return builder.compile(checkpointer=memory)


# Create enhanced graph instance
enhanced_graph = build_enhanced_graph()
