"""
Debate component for modular graph architecture.
Provides multi-agent debate functionality for critical analysis.
"""

from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from src.graph.types import AgentState
from src.agents.agents import create_agent
from src.prompts.prompt_manager import get_prompt


class DebateConfig:
    """Configuration for the debate component."""

    def __init__(
        self,
        num_debaters: int = 3,
        max_rounds: int = 3,
        debate_topic: str = "research_findings",
        consensus_threshold: float = 0.8,
    ):
        self.num_debaters = num_debaters
        self.max_rounds = max_rounds
        self.debate_topic = debate_topic
        self.consensus_threshold = consensus_threshold


def create_debater_agent(llm, role: str, prompt_name: str = None):
    """Create a debater agent with specific role and prompt.

    Args:
        llm: Language model instance.
        role: Role of the debater (e.g., "optimist", "pessimist", "neutral").
        prompt_name: Name of the prompt to use. Auto-generated if None.

    Returns:
        Configured debater agent.
    """
    if prompt_name is None:
        prompt_name = f"debater_{role}"

    prompt = get_prompt(prompt_name, get_prompt("debater_default"))

    # Customize prompt with role
    role_prompt = prompt.format(role=role)

    return create_agent(llm, role_prompt)


def prepare_debate_context(state: AgentState, config: DebateConfig) -> Dict[str, Any]:
    """Prepare context for debate based on current state.

    Args:
        state: Current agent state.
        config: Debate configuration.

    Returns:
        Debate context dictionary.
    """
    return {
        "research_findings": state.get("findings", []),
        "current_plan": state.get("plan", ""),
        "evidence": state.get("evidence", []),
        "debate_topic": config.debate_topic,
        "round": state.get("debate_round", 0) + 1,
        "previous_arguments": state.get("debate_arguments", []),
    }


def execute_debate_round(
    state: AgentState, debaters: List, config: DebateConfig
) -> Dict[str, Any]:
    """Execute a single round of debate.

    Args:
        state: Current agent state.
        debaters: List of debater agents.
        config: Debate configuration.

    Returns:
        Updated state with debate results.
    """
    debate_context = prepare_debate_context(state, config)

    # Collect arguments from all debaters
    arguments = []
    for i, debater in enumerate(debaters):
        debater_input = {
            **debate_context,
            "debater_id": i,
            "total_debaters": len(debaters),
        }

        try:
            result = debater.invoke(debater_input)
            arguments.append(
                {
                    "debater_id": i,
                    "argument": result.get("content", ""),
                    "stance": result.get("stance", "neutral"),
                    "confidence": result.get("confidence", 0.5),
                }
            )
        except Exception as e:
            logger.error(f"Debate error for debater {i}: {e}")
            arguments.append(
                {
                    "debater_id": i,
                    "argument": f"Error in debate: {str(e)}",
                    "stance": "error",
                    "confidence": 0.0,
                }
            )

    # Analyze consensus
    stances = [arg["stance"] for arg in arguments]
    confidences = [arg["confidence"] for arg in arguments]

    # Simple consensus calculation
    unique_stances = set(stances)
    consensus_score = 1.0 - (len(unique_stances) - 1) / max(len(stances), 1)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "debate_round": debate_context["round"],
        "debate_arguments": arguments,
        "consensus_score": consensus_score,
        "avg_confidence": avg_confidence,
        "debate_summary": summarize_debate(arguments),
    }


def summarize_debate(arguments: List[Dict[str, Any]]) -> str:
    """Summarize debate arguments.

    Args:
        arguments: List of debate arguments.

    Returns:
        Summary of the debate.
    """
    if not arguments:
        return "No arguments provided."

    stances = {}
    for arg in arguments:
        stance = arg["stance"]
        if stance not in stances:
            stances[stance] = []
        stances[stance].append(arg["argument"])

    summary_parts = []
    for stance, args in stances.items():
        summary_parts.append(f"{stance.upper()}: {' '.join(args[:2])}")

    return " | ".join(summary_parts)


def should_continue_debate(state: AgentState, config: DebateConfig) -> str:
    """Determine if debate should continue.

    Args:
        state: Current agent state.
        config: Debate configuration.

    Returns:
        "continue" or "end" based on debate criteria.
    """
    current_round = state.get("debate_round", 0)
    consensus_score = state.get("consensus_score", 0.0)

    # Check max rounds
    if current_round >= config.max_rounds:
        return "end"

    # Check consensus threshold
    if consensus_score >= config.consensus_threshold:
        return "end"

    return "continue"


def create_debate_component(
    llm, config: Optional[DebateConfig] = None, roles: Optional[List[str]] = None
) -> CompiledStateGraph:
    """Create a debate component as a reusable sub-graph.

    Args:
        llm: Language model instance for debaters.
        config: Debate configuration. Uses defaults if not provided.
        roles: List of debater roles. Auto-generated if not provided.

    Returns:
        Compiled debate sub-graph.
    """
    if config is None:
        config = DebateConfig()

    if roles is None:
        roles = ["optimist", "pessimist", "neutral"]

    # Ensure we have enough roles
    while len(roles) < config.num_debaters:
        roles.append(f"debater_{len(roles)}")

    # Create debater agents
    debaters = [
        create_debater_agent(llm, role) for role in roles[: config.num_debaters]
    ]

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add debate node
    def debate_node(state: AgentState) -> Dict[str, Any]:
        return execute_debate_round(state, debaters, config)

    workflow.add_node("debate", debate_node)

    # Set entry point
    workflow.set_entry_point("debate")

    # Conditional edge based on debate criteria
    def debate_condition(state: AgentState) -> str:
        return should_continue_debate(state, config)

    workflow.add_conditional_edges(
        "debate", debate_condition, {"continue": "debate", "end": END}
    )

    return workflow.compile()


def create_multi_agent_debate(
    llm,
    num_debaters: int = 3,
    max_rounds: int = 3,
    consensus_threshold: float = 0.8,
    roles: Optional[List[str]] = None,
) -> CompiledStateGraph:
    """Create a complete multi-agent debate component.

    Args:
        llm: Language model instance.
        num_debaters: Number of debater agents.
        max_rounds: Maximum debate rounds.
        consensus_threshold: Threshold for consensus.
        roles: List of debater roles.

    Returns:
        Compiled debate graph.
    """
    config = DebateConfig(
        num_debaters=num_debaters,
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold,
    )

    return create_debate_component(llm, config, roles)


def integrate_debate(
    main_graph: StateGraph,
    debate_graph: CompiledStateGraph,
    integration_point: str = "before_synthesis",
) -> StateGraph:
    """Integrate debate component into a main graph.

    Args:
        main_graph: The main workflow graph.
        debate_graph: The debate sub-graph.
        integration_point: Where to integrate debate ("before_synthesis", "after_research", etc.).

    Returns:
        Updated main graph with debate integrated.
    """
    # Add the debate graph as a sub-graph
    main_graph.add_node("debate_session", debate_graph)

    # Update edges based on integration point
    if integration_point == "before_synthesis":
        # Add debate before synthesis
        main_graph.add_edge("research", "debate_session")
        main_graph.add_edge("debate_session", "synthesize")
    elif integration_point == "after_research":
        # Add debate after research
        main_graph.add_edge("research", "debate_session")
        main_graph.add_edge("debate_session", "analyze")

    return main_graph
