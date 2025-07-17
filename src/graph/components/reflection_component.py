"""
Reflection component for modular graph architecture.
Provides a reusable reflection loop that can be integrated into larger workflows.
"""

from typing import Any, Dict, List, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from src.graph.types import AgentState
from src.agents.agents import create_agent
from src.prompts.prompt_manager import get_prompt


class ReflectionConfig:
    """Configuration for the reflection component."""

    def __init__(
        self,
        max_reflection_loops: int = 3,
        reflection_trigger_threshold: int = 2,
        reflection_confidence_threshold: float = 0.7,
        reflection_temperature: float = 0.7,
        enable_progressive_reflection: bool = True,
    ):
        self.max_reflection_loops = max_reflection_loops
        self.reflection_trigger_threshold = reflection_trigger_threshold
        self.reflection_confidence_threshold = reflection_confidence_threshold
        self.reflection_temperature = reflection_temperature
        self.enable_progressive_reflection = enable_progressive_reflection


def create_reflection_agent(llm, prompt_name: str = "reflection_analysis"):
    """Create a reflection agent with the specified prompt.

    Args:
        llm: Language model instance.
        prompt_name: Name of the prompt to use for reflection.

    Returns:
        Configured reflection agent.
    """
    prompt = get_prompt(prompt_name)
    return create_agent(llm, prompt)


def should_continue_reflection(state: AgentState, config: ReflectionConfig) -> str:
    """Determine if reflection should continue.

    Args:
        state: Current agent state.
        config: Reflection configuration.

    Returns:
        "continue" or "end" based on reflection criteria.
    """
    reflection_count = state.get("reflection_count", 0)

    # Check if we've reached max loops
    if reflection_count >= config.max_reflection_loops:
        return "end"

    # Check if we have enough research steps to trigger reflection
    research_steps = len(state.get("research_steps", []))
    if research_steps < config.reflection_trigger_threshold:
        return "end"

    # Check confidence threshold
    confidence = state.get("confidence", 0.0)
    if confidence >= config.reflection_confidence_threshold:
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
    }

    # Execute reflection
    reflection_result = reflection_agent.invoke(reflection_input)

    # Update state with reflection results
    updates = {
        "reflection_count": reflection_input["reflection_count"],
        "last_reflection": reflection_result.get("content", ""),
        "confidence": reflection_result.get("confidence", 0.5),
        "suggested_improvements": reflection_result.get("suggestions", []),
    }

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
    config: Optional[ReflectionConfig] = None,
    prompt_name: str = "reflection_analysis",
) -> CompiledStateGraph:
    """Create a reflection component as a reusable sub-graph.

    Args:
        llm: Language model instance for reflection.
        config: Reflection configuration. Uses defaults if not provided.
        prompt_name: Name of the prompt to use for reflection.

    Returns:
        Compiled reflection sub-graph.
    """
    if config is None:
        config = ReflectionConfig()

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
        return should_continue_reflection(state, config)

    workflow.add_conditional_edges(
        "reflect", reflection_condition, {"continue": "reflect", "end": END}
    )

    return workflow.compile()


def create_reflection_loop(
    llm,
    max_loops: int = 3,
    trigger_threshold: int = 2,
    confidence_threshold: float = 0.7,
) -> CompiledStateGraph:
    """Create a complete reflection loop component.

    Args:
        llm: Language model instance.
        max_loops: Maximum number of reflection iterations.
        trigger_threshold: Minimum research steps before reflection triggers.
        confidence_threshold: Confidence threshold to stop reflection.

    Returns:
        Compiled reflection loop graph.
    """
    config = ReflectionConfig(
        max_reflection_loops=max_loops,
        reflection_trigger_threshold=trigger_threshold,
        reflection_confidence_threshold=confidence_threshold,
    )

    return create_reflection_component(llm, config)


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
