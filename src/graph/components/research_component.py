"""
Research component for modular graph architecture.
Provides reusable research workflow that can be integrated into larger systems.
"""

import logging
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.graph.types import State as AgentState
from src.agents.agents import create_agent_with_managed_prompt
from src.tools.base_tool import get_tool_registry


class ResearchConfig:
    """Configuration for the research component."""

    def __init__(
        self,
        max_search_results: int = 2,
        max_research_steps: int = 5,
        enable_smart_filtering: bool = True,
        enable_content_summarization: bool = True,
        summary_type: str = "comprehensive",
    ):
        self.max_search_results = max_search_results
        self.max_research_steps = max_research_steps
        self.enable_smart_filtering = enable_smart_filtering
        self.enable_content_summarization = enable_content_summarization
        self.summary_type = summary_type


def create_research_agent(
    tools: List, prompt_name: str = "researcher", configurable=None
):
    """Create a research agent with the specified prompt.

    Args:
        tools: List of tools available to the agent.
        prompt_name: Name of the prompt to use for research.
        configurable: Optional configurable object containing locale and other parameters.

    Returns:
        Configured research agent.
    """
    return create_agent_with_managed_prompt(
        prompt_name, "researcher", tools, configurable=configurable
    )


def create_research_tools() -> List:
    """Create research tools for the agent.

    Returns:
        List of research tools.
    """
    registry = get_tool_registry()

    # Get specific research tools
    tools = []

    # Search tools
    if registry.has_tool("search"):
        tools.append(registry.get_tool("search"))

    if registry.has_tool("tavily_search"):
        tools.append(registry.get_tool("tavily_search"))

    # Crawl tools
    if registry.has_tool("crawl"):
        tools.append(registry.get_tool("crawl"))

    # Retriever tools
    if registry.has_tool("retriever"):
        tools.append(registry.get_tool("retriever"))

    return tools


def research_step(
    state: AgentState, research_agent, tools: List, config: ResearchConfig
) -> Dict[str, Any]:
    """Execute a single research step.

    Args:
        state: Current agent state.
        research_agent: Agent responsible for research.
        tools: Available research tools.
        config: Research configuration.

    Returns:
        Updated state with research results.
    """
    # Prepare research context
    query = state.get("current_query", state.get("query", ""))
    previous_findings = state.get("findings", [])
    research_steps = state.get("research_steps", [])

    research_input = {
        "query": query,
        "previous_findings": previous_findings,
        "research_step": len(research_steps) + 1,
        "max_search_results": config.max_search_results,
        "tools": [tool.name for tool in tools],
    }

    # Execute research
    try:
        result = research_agent.invoke(research_input)

        # Extract findings
        new_findings = result.get("findings", [])
        search_results = result.get("search_results", [])

        # Process findings
        processed_findings = process_research_findings(
            new_findings, search_results, config
        )

        # Update state
        updates = {
            "findings": previous_findings + processed_findings,
            "research_steps": (
                research_steps
                + [
                    {
                        "step": len(research_steps) + 1,
                        "query": query,
                        "findings": len(processed_findings),
                        "timestamp": result.get("timestamp", ""),
                    }
                ]
            ),
            "last_research_result": result,
        }

        return updates

    except Exception as e:
        logger.error(f"Research step failed: {e}")
        return {
            "findings": previous_findings,
            "research_steps": (
                research_steps
                + [
                    {
                        "step": len(research_steps) + 1,
                        "query": query,
                        "error": str(e),
                        "status": "failed",
                    }
                ]
            ),
        }


def process_research_findings(
    findings: List, search_results: List, config: ResearchConfig
) -> List:
    """Process and filter research findings.

    Args:
        findings: Raw research findings.
        search_results: Search results from tools.
        config: Research configuration.

    Returns:
        Processed and filtered findings.
    """
    processed = []

    for finding in findings:
        if isinstance(finding, dict):
            # Apply smart filtering if enabled
            if config.enable_smart_filtering:
                if should_include_finding(finding):
                    processed.append(finding)
            else:
                processed.append(finding)
        else:
            # Convert simple findings to structured format
            processed.append(
                {
                    "content": str(finding),
                    "source": "research",
                    "relevance": 0.5,
                    "timestamp": "",
                }
            )

    return processed[: config.max_search_results]


def should_include_finding(finding: Dict) -> bool:
    """Determine if a finding should be included based on relevance.

    Args:
        finding: Research finding to evaluate.

    Returns:
        True if finding should be included.
    """
    # Simple relevance check - can be enhanced with LLM-based filtering
    relevance = finding.get("relevance", 0.5)
    return relevance >= 0.3


def should_continue_research(state: AgentState, config: ResearchConfig) -> str:
    """Determine if research should continue.

    Args:
        state: Current agent state.
        config: Research configuration.

    Returns:
        "continue" or "end" based on research criteria.
    """
    research_steps = state.get("research_steps", [])
    findings = state.get("findings", [])

    # Check max research steps
    if len(research_steps) >= config.max_research_steps:
        return "end"

    # Check if we have sufficient findings
    if len(findings) >= config.max_search_results:
        return "end"

    # Check if we should search for more specific information
    if state.get("needs_more_research", False):
        return "continue"

    return "end"


def create_research_component(
    llm, config: Optional[ResearchConfig] = None, prompt_name: str = "researcher"
) -> CompiledStateGraph:
    """Create a research component as a reusable sub-graph.

    Args:
        llm: Language model instance for research.
        config: Research configuration. Uses defaults if not provided.
        prompt_name: Name of the prompt to use for research.

    Returns:
        Compiled research sub-graph.
    """
    if config is None:
        config = ResearchConfig()

    # Create research tools
    tools = create_research_tools()

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add research node
    def research_node(state: AgentState) -> Dict[str, Any]:
        # Get configurable from state, or create one with locale
        configurable = state.get("agent_configurable")
        if not configurable:
            locale = state.get("locale", "en-US")
            configurable = {"locale": locale}

        research_agent = create_research_agent(tools, prompt_name, configurable)
        return research_step(state, research_agent, tools, config)

    workflow.add_node("research", research_node)

    # Set entry point
    workflow.set_entry_point("research")

    # Conditional edge based on research criteria
    def research_condition(state: AgentState) -> str:
        return should_continue_research(state, config)

    workflow.add_conditional_edges(
        "research", research_condition, {"continue": "research", "end": END}
    )

    return workflow.compile()


def create_research_workflow(
    llm,
    max_search_results: int = 3,
    max_research_steps: int = 5,
    enable_smart_filtering: bool = True,
) -> CompiledStateGraph:
    """Create a complete research workflow component.

    Args:
        llm: Language model instance.
        max_search_results: Maximum search results per step.
        max_research_steps: Maximum research steps.
        enable_smart_filtering: Whether to enable smart filtering.

    Returns:
        Compiled research workflow graph.
    """
    config = ResearchConfig(
        max_search_results=max_search_results,
        max_research_steps=max_research_steps,
        enable_smart_filtering=enable_smart_filtering,
    )

    return create_research_component(llm, config)


def integrate_research(
    main_graph: StateGraph,
    research_graph: CompiledStateGraph,
    integration_point: str = "after_planning",
) -> StateGraph:
    """Integrate research component into a main graph.

    Args:
        main_graph: The main workflow graph.
        research_graph: The research sub-graph.
        integration_point: Where to integrate research ("after_planning", "standalone", etc.).

    Returns:
        Updated main graph with research integrated.
    """
    # Add the research graph as a sub-graph
    main_graph.add_node("research_workflow", research_graph)

    # Update edges based on integration point
    if integration_point == "after_planning":
        # Add research after planning
        main_graph.add_edge("plan", "research_workflow")
        main_graph.add_edge("research_workflow", "synthesize")
    elif integration_point == "standalone":
        # Use research as standalone component
        main_graph.set_entry_point("research_workflow")
        main_graph.add_edge("research_workflow", END)

    return main_graph


logger = logging.getLogger(__name__)
