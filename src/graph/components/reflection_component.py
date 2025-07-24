"""
Reflection component for modular graph architecture.
Provides a reusable reflection loop that can be integrated into larger workflows.
"""

import logging
from typing import Any, Dict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.graph.types import State as AgentState
from src.agents.agents import create_agent_with_managed_prompt

logger = logging.getLogger(__name__)


# ReflectionConfig removed - using unified ReflectionSettings from src.config.models


def create_reflection_agent(
    tools: List, prompt_name: str = "reflection_analysis", configurable=None
):
    """Create a reflection agent with the specified prompt.

    Args:
        tools: List of tools available to the agent.
        prompt_name: Name of the prompt to use for reflection.
        configurable: Optional configurable object containing locale and other parameters.

    Returns:
        Configured reflection agent.
    """
    return create_agent_with_managed_prompt(
        prompt_name, "reflection", tools, configurable=configurable
    )


def should_continue_reflection(
    state: AgentState, max_loops: int = 3, confidence_threshold: float = 0.7
) -> str:
    """Determine if reflection should continue.

    Args:
        state: Current agent state.
        max_loops: Maximum number of reflection loops.
        confidence_threshold: Confidence threshold to stop reflection.

    Returns:
        "continue" or "end" based on reflection criteria.
    """
    reflection_count = state.get("reflection_count", 0)
    logger.info(f"Current reflection count: {reflection_count}")
    # Check if we've reached max loops
    if reflection_count >= max_loops:
        return "end"

    # Always allow reflection to trigger (minimum threshold is 1)
    research_steps = len(state.get("research_steps", []))
    if research_steps < 1:
        return "end"

    # Check confidence threshold
    confidence = state.get("confidence", 0.0)
    if confidence >= confidence_threshold:
        return "end"

    return "continue"


def reflection_node(state: AgentState, reflection_agent) -> Dict[str, Any]:
    """Execute reflection analysis on current research.

    Args:
        state: Current agent state.
        reflection_agent: Agent responsible for reflection.

    Returns:
        Updated state with reflection results.
    """
    # Prepare reflection context
    research_steps = state.get("research_steps", [])
    current_plan = state.get("plan", "")
    findings = state.get("findings", [])

    reflection_input = {
        "research_steps": research_steps,
        "current_plan": current_plan,
        "findings": findings,
        "reflection_count": state.get("reflection_count", 0) + 1,
        "locale": state.get("locale", "en-US"),
    }

    # Execute reflection
    import logging

    logger = logging.getLogger(__name__)

    logger.info(
        f"Starting reflection analysis (iteration {reflection_input['reflection_count']})"
    )
    logger.debug(f"Reflection input: {reflection_input}")

    reflection_result = reflection_agent.invoke(reflection_input)

    # Log AI reflection results
    logger.info("Reflection analysis completed successfully")
    logger.debug(f"Reflection agent result: {reflection_result}")

    # Log specific reflection content
    if reflection_result.get("content"):
        logger.info(f"Reflection content length: {len(reflection_result['content'])}")
        logger.debug(f"Reflection content: {reflection_result['content']}")

    if reflection_result.get("suggestions"):
        logger.info(
            f"Reflection suggestions count: {len(reflection_result['suggestions'])}"
        )
        logger.debug(f"Reflection suggestions: {reflection_result['suggestions']}")

    # Update state with reflection results
    updates = {
        "reflection_count": reflection_input["reflection_count"],
        "last_reflection": reflection_result.get("content", ""),
        "confidence": reflection_result.get("confidence", 0.5),
        "suggested_improvements": reflection_result.get("suggestions", []),
    }

    logger.info(
        f"Reflection state updates: confidence={updates['confidence']}, improvements_count={len(updates['suggested_improvements'])}"
    )

    # Update plan if suggestions include plan modifications
    if "suggested_improvements" in reflection_result:
        improvements = reflection_result["suggested_improvements"]
        for improvement in improvements:
            if improvement.get("type") == "plan_update":
                updates["plan"] = improvement.get("new_plan", current_plan)
                break

    return updates


def create_reflection_component(
    llm,
    max_loops: int = 3,
    confidence_threshold: float = 0.7,
    prompt_name: str = "reflection_analysis",
) -> CompiledStateGraph:
    """Create a reflection component as a reusable sub-graph.

    Args:
        llm: Language model instance for reflection.
        max_loops: Maximum number of reflection loops.
        confidence_threshold: Confidence threshold to stop reflection.
        prompt_name: Name of the prompt to use for reflection.

    Returns:
        Compiled reflection sub-graph.
    """
    # Create reflection agent
    reflection_agent = create_reflection_agent(llm, prompt_name)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("reflect", lambda state: reflection_node(state, reflection_agent))

    # Add edges
    workflow.set_entry_point("reflect")

    # Conditional edge based on reflection criteria
    def reflection_condition(state: AgentState) -> str:
        return should_continue_reflection(state, max_loops, confidence_threshold)

    workflow.add_conditional_edges(
        "reflect", reflection_condition, {"continue": "reflect", "end": END}
    )

    return workflow.compile()


def create_reflection_loop(
    llm,
    max_loops: int = 3,
    confidence_threshold: float = 0.7,
) -> CompiledStateGraph:
    """Create a complete reflection loop component.

    Args:
        llm: Language model instance.
        max_loops: Maximum number of reflection iterations.
        confidence_threshold: Confidence threshold to stop reflection.

    Returns:
        Compiled reflection loop graph.
    """
    return create_reflection_component(llm, max_loops, confidence_threshold)


def integrate_reflection(
    main_graph: StateGraph,
    reflection_graph: CompiledStateGraph,
    integration_point: str = "after_research",
) -> StateGraph:
    """Integrate reflection component into a main graph.

    Args:
        main_graph: The main workflow graph.
        reflection_graph: The reflection sub-graph.
        integration_point: Where to integrate reflection ("after_research", "after_plan", etc.).

    Returns:
        Updated main graph with reflection integrated.
    """
    # Add the reflection graph as a sub-graph
    main_graph.add_node("reflection_loop", reflection_graph)

    # Update edges based on integration point
    if integration_point == "after_research":
        # Add reflection after research steps
        main_graph.add_edge("research", "reflection_loop")
        main_graph.add_edge("reflection_loop", "synthesize")
    elif integration_point == "after_plan":
        # Add reflection after planning
        main_graph.add_edge("plan", "reflection_loop")
        main_graph.add_edge("reflection_loop", "execute")

    return main_graph
