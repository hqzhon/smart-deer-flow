"""Researcher Context Extension - Phase 1 Implementation

This module provides context extension functionality for researcher nodes,
integrating with the existing architecture while adding isolation capabilities.
"""

import logging
import os
import traceback
from typing import Dict, Any, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage
from .researcher_context_isolator import (
    ResearcherContextIsolator,
    ResearcherContextConfig,
)
from .researcher_isolation_metrics import ResearcherIsolationMetrics
from .researcher_progressive_enablement import (
    ResearcherProgressiveEnabler,
    ScenarioContext,
)
from ..context.execution_context_manager import ExecutionContextManager
from src.llms.error_handler import safe_llm_call_async
from src.graph.types import State
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

# Remove TYPE_CHECKING import for Configuration as it's no longer needed

logger = logging.getLogger(__name__)


class ResearcherContextExtension:
    """Extension class for researcher node with context isolation capabilities.

    This class provides enhanced researcher execution with built-in context
    isolation to prevent context accumulation issues in parallel scenarios.
    """

    def __init__(
        self,
        base_manager: Optional[ExecutionContextManager] = None,
        isolation_config: Optional[ResearcherContextConfig] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the researcher context extension.

        Args:
            base_manager: Base ExecutionContextManager for integration. Optional for backward compatibility.
            isolation_config: Configuration for context isolation. Uses defaults if None.
            config: Configuration dictionary for Phase 3 features
        """
        self.base_manager = base_manager
        self.isolation_config = isolation_config
        self.isolator = ResearcherContextIsolator(isolation_config)
        self.active_isolators = {}  # Track active isolation sessions
        self.config = config or {}

        # Initialize Phase 3 components if metrics are enabled
        self.metrics = None
        self.progressive_enabler = None

        if self.config.get("researcher_isolation_metrics", False):
            self.metrics = ResearcherIsolationMetrics()

        if self.config.get("researcher_auto_isolation", False):
            self.progressive_enabler = ResearcherProgressiveEnabler()

        # Phase 2 - GFLQ Integration: Initialize reflection agent
        self.reflection_agent = None
        if self.config.get("enable_enhanced_reflection", True):
            try:
                from src.utils.reflection.enhanced_reflection import (
                    EnhancedReflectionAgent,
                )

                self.reflection_agent = EnhancedReflectionAgent(config=self.config)
                logger.info(
                    "Enhanced reflection agent initialized for context extension"
                )
            except ImportError as e:
                logger.warning(f"Failed to initialize reflection agent: {e}")

        logger.info(
            "ResearcherContextExtension initialized with base_manager integration"
        )

    def get_isolated_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get isolated context by ID.

        Args:
            context_id: Context identifier

        Returns:
            Isolated context data or None if not found
        """
        return self.active_isolators.get(context_id)

    def create_isolated_context(
        self,
        context_id: str,
        research_topic: str,
        description: str,
        config: Optional[ResearcherContextConfig] = None,
    ) -> str:
        """Create a new isolated context for analysis.

        Args:
            context_id: Unique identifier for the context
            research_topic: Topic for research
            description: Description of the context
            config: Configuration for isolation

        Returns:
            Context ID
        """
        if config:
            isolator = ResearcherContextIsolator(config)
        else:
            isolator = ResearcherContextIsolator(self.isolator.config)

        self.active_isolators[context_id] = isolator
        return context_id

    def finalize_isolated_context(self, context_id: str) -> Dict[str, Any]:
        """Finalize and clean up an isolated context.

        Args:
            context_id: Context identifier to finalize

        Returns:
            Cleanup result dictionary
        """
        if context_id in self.active_isolators:
            del self.active_isolators[context_id]
            return {"status": "cleaned", "context_id": context_id}
        else:
            return {"status": "not_found", "context_id": context_id}

    async def execute_researcher_step_with_isolation(
        self,
        state: State,
        config: RunnableConfig,
        agent,
        agent_name: str = "researcher",
    ) -> Command[Literal["research_team"]]:
        """Execute researcher step with context isolation.

        This method extends the existing _execute_agent_step functionality
        with researcher-specific context isolation.

        Args:
            state: Current state
            config: Runnable configuration
            agent: The researcher agent to execute
            agent_name: Name of the agent (default: "researcher")

        Returns:
            Command to update state and continue to research_team
        """
        current_plan = state.get("current_plan")
        observations = state.get("observations", [])

        # Find the first unexecuted step
        current_step = None
        completed_steps = []
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
            return Command(goto="research_team")

        # Phase 3: Progressive enablement check
        should_use_isolation = self._should_use_isolation(
            state, current_step, agent_name
        )
        if not should_use_isolation:
            logger.info(
                f"Progressive enablement determined isolation not needed for: {current_step.title}"
            )
            return await self._execute_without_isolation(
                state, agent, agent_name, current_step
            )

        # Phase 3: Start isolation session metrics
        session_id = None
        if self.metrics:
            scenario_context = self._create_scenario_context(
                state, current_step, agent_name
            )
            session_id = self.metrics.start_isolation_session(
                agent_name=agent_name,
                step_title=current_step.title,
                scenario_context=scenario_context,
            )

        logger.info(
            f"Executing step with isolation: {current_step.title}, agent: {agent_name}"
        )

        # Prepare isolated context using the isolator
        current_step_dict = {
            "step": current_step.title,
            "description": current_step.description,
        }

        optimized_steps, context_info = self.isolator.prepare_isolated_context(
            completed_steps, current_step_dict, agent_name
        )

        # Prepare the input for the agent with isolated context
        agent_input = {
            "messages": [
                HumanMessage(
                    content=f"{context_info}\n\n# Current Task\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
                )
            ]
        }

        # Add researcher-specific guidance and resource information
        agent_input = await self._add_researcher_guidance(agent_input, state)

        # Execute the agent with proper error handling
        try:
            result = await self._execute_agent_with_isolation(
                agent, agent_input, current_step
            )

            # Phase 3: Update metrics on successful execution
            if self.metrics and session_id:
                response_content = self._extract_response_content(result)
                self.metrics.update_isolation_session(
                    session_id=session_id,
                    tokens_saved=len(context_info) // 4,  # Rough estimate
                    execution_time=0,  # Will be calculated by metrics
                    success=True,
                    response_length=len(response_content),
                )

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Phase 3: Update metrics on failure
            if self.metrics and session_id:
                self.metrics.update_isolation_session(
                    session_id=session_id,
                    tokens_saved=0,
                    execution_time=0,
                    success=False,
                    error_message=str(e),
                )

            # Return a command with error information
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"Research execution failed: {str(e)}",
                            name=agent_name,
                        )
                    ]
                },
                goto="research_team",
            )

        # Process the result
        response_content = self._extract_response_content(result)
        logger.debug(f"{agent_name.capitalize()} response: {response_content[:200]}...")

        # Manage observations with isolation
        new_observation = f"Step: {current_step.title}\nAgent: {agent_name}\nResult: {response_content}"
        optimized_observations = self.isolator.manage_isolated_observations(
            observations, new_observation
        )

        # Create updated step and plan
        updated_step = current_step.copy(update={"execution_res": response_content})
        updated_plan = self._update_plan_with_step(
            current_plan, current_step, updated_step
        )

        # Phase 3: End isolation session and record outcome
        if self.metrics and session_id:
            self.metrics.end_isolation_session(session_id)

        if self.progressive_enabler:
            scenario_context = self._create_scenario_context(
                state, current_step, agent_name
            )
            self.progressive_enabler.record_scenario_outcome(
                scenario_context=scenario_context,
                used_isolation=True,
                success=True,
                performance_gain=0.1,  # Placeholder - could be calculated from metrics
            )

        logger.info(
            f"Step '{current_step.title}' execution completed with isolation by {agent_name}"
        )

        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=response_content,
                        name=agent_name,
                    )
                ],
                "observations": optimized_observations,
                "current_plan": updated_plan,
            },
            goto="research_team",
        )

    async def _add_researcher_guidance(
        self, agent_input: Dict[str, Any], state: State
    ) -> Dict[str, Any]:
        """Add researcher-specific guidance and resource information.

        Args:
            agent_input: Base agent input dictionary
            state: Current state

        Returns:
            Enhanced agent input with researcher guidance
        """
        # Phase 2 - GFLQ Integration: Add reflection-enhanced guidance
        if self.reflection_agent:
            try:
                # Analyze current research context
                current_plan = state.get("current_plan")
                observations = state.get("observations", [])

                # Create proper ReflectionContext object
                from src.utils.reflection.enhanced_reflection import ReflectionContext

                completed_steps = []
                if current_plan and hasattr(current_plan, "steps"):
                    for step in current_plan.steps:
                        if step.execution_res:
                            completed_steps.append(
                                {
                                    "step": step.title,
                                    "description": step.description,
                                    "execution_res": step.execution_res,
                                }
                            )

                execution_results = []
                if current_plan and hasattr(current_plan, "steps"):
                    execution_results = [
                        step.execution_res
                        for step in current_plan.steps
                        if step.execution_res
                    ]

                context = ReflectionContext(
                    research_topic=(
                        getattr(current_plan, "description", "") if current_plan else ""
                    ),
                    completed_steps=completed_steps,
                    execution_results=execution_results,
                    observations=observations,
                    total_steps=len(current_plan.steps) if current_plan else 0,
                    current_step_index=len(completed_steps),
                    locale=state.get("locale", "en-US"),
                )

                # Get reflection insights - need to await the coroutine
                reflection_result = (
                    await self.reflection_agent.assess_research_sufficiency(context)
                )

                if (
                    not reflection_result.is_sufficient
                    and reflection_result.follow_up_queries
                ):
                    guidance_content = "**Research Enhancement Suggestions:**\n\n"
                    guidance_content += (
                        "Based on analysis, consider exploring these areas:\n\n"
                    )

                    for i, query in enumerate(
                        reflection_result.follow_up_queries[:3], 1
                    ):
                        guidance_content += f"{i}. {query}\n"

                    if reflection_result.knowledge_gaps:
                        guidance_content += "\n**Identified Knowledge Gaps:**\n\n"
                        for gap in reflection_result.knowledge_gaps[:3]:
                            guidance_content += f"- {gap}\n"

                    agent_input["messages"].append(
                        HumanMessage(
                            content=guidance_content, name="reflection_guidance"
                        )
                    )

            except Exception as e:
                logger.warning(f"Reflection guidance failed: {e}")

        # Add resource information if available
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        # Add citation guidance
        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )

        return agent_input

    async def _execute_agent_with_isolation(
        self, agent, agent_input: Dict[str, Any], current_step
    ) -> Any:
        """Execute agent with isolation-specific settings.

        Args:
            agent: The agent to execute
            agent_input: Input for the agent
            current_step: Current step being executed

        Returns:
            Agent execution result
        """
        # Get recursion limit from environment
        default_recursion_limit = 25
        try:
            env_value_str = os.getenv(
                "AGENT_RECURSION_LIMIT", str(default_recursion_limit)
            )
            parsed_limit = int(env_value_str)
            recursion_limit = (
                parsed_limit if parsed_limit > 0 else default_recursion_limit
            )
        except ValueError:
            logger.warning(
                f"Invalid AGENT_RECURSION_LIMIT value. Using default: {default_recursion_limit}"
            )
            recursion_limit = default_recursion_limit

        # Optimize messages for token efficiency
        max_messages = 15  # Reduced for isolation
        if len(agent_input["messages"]) > max_messages:
            agent_input["messages"] = agent_input["messages"][-max_messages:]
            logger.debug(f"Truncated messages to {max_messages} for isolation")

        logger.info(f"Executing agent with {len(agent_input['messages'])} messages")

        # Execute with safe LLM call
        result = await safe_llm_call_async(
            agent.ainvoke,
            input=agent_input,
            config={"recursion_limit": recursion_limit},
            operation_name="researcher executor with isolation",
            context=f"Execute step with isolation: {current_step.title}",
            enable_smart_processing=True,
            max_retries=3,
        )

        return result

    def _extract_response_content(self, result: Any) -> str:
        """Extract response content from agent result.

        Args:
            result: Agent execution result

        Returns:
            Extracted response content as string
        """
        if hasattr(result, "get") and "messages" in result:
            return result["messages"][-1].content
        else:
            return result.content if hasattr(result, "content") else str(result)

    def _update_plan_with_step(self, current_plan, current_step, updated_step):
        """Update plan with executed step.

        Args:
            current_plan: Current plan object
            current_step: Original step object
            updated_step: Updated step with execution result

        Returns:
            Updated plan object
        """
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

        return current_plan.copy(update={"steps": updated_steps})

    def _should_use_isolation(
        self, state: State, current_step, agent_name: str
    ) -> bool:
        """Determine if isolation should be used for this step.

        Args:
            state: Current state
            current_step: Current step being executed
            agent_name: Name of the agent

        Returns:
            True if isolation should be used
        """
        if not self.progressive_enabler:
            return True  # Default to using isolation if no progressive enabler

        scenario_context = self._create_scenario_context(
            state, current_step, agent_name
        )
        result, reason, factors = self.progressive_enabler.should_enable_isolation(
            scenario_context
        )
        return result

    def _create_scenario_context(
        self, state: State, current_step, agent_name: str
    ) -> ScenarioContext:
        """Create scenario context for progressive enablement.

        Args:
            state: Current state
            current_step: Current step being executed
            agent_name: Name of the agent

        Returns:
            ScenarioContext object
        """
        observations = state.get("observations", [])
        current_plan = state.get("current_plan")
        step_count = len(current_plan.steps) if current_plan else 1

        # Estimate tokens based on content length
        estimated_tokens = len(current_step.description) * 2  # Rough estimate
        if observations:
            estimated_tokens += sum(len(str(obs)) for obs in observations) // 4

        return ScenarioContext(
            task_description=f"{current_step.title}: {current_step.description}",
            step_count=step_count,
            context_size=len(str(observations)),
            parallel_execution=state.get("parallel_execution", False),
            has_search_results=any(
                "search" in str(obs).lower() for obs in observations
            ),
            has_complex_queries=len(current_step.description) > 200,
            estimated_tokens=estimated_tokens,
        )

    async def _execute_without_isolation(
        self, state: State, agent, agent_name: str, current_step
    ) -> Command:
        """Execute step without isolation for comparison.

        Args:
            state: Current state
            agent: The agent to execute
            agent_name: Name of the agent
            current_step: Current step to execute

        Returns:
            Command with updated state
        """
        # Simple execution without isolation - placeholder implementation
        logger.info(f"Executing step without isolation: {current_step.title}")

        # Basic agent input without isolation optimizations
        agent_input = {
            "messages": [
                HumanMessage(
                    content=f"# Current Task\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
                )
            ]
        }

        try:
            result = await self._execute_agent_with_isolation(
                agent, agent_input, current_step
            )
            response_content = self._extract_response_content(result)

            # Update step and plan
            updated_step = current_step.copy(update={"execution_res": response_content})
            updated_plan = self._update_plan_with_step(
                state.get("current_plan"), current_step, updated_step
            )

            # Record outcome for progressive enablement
            if self.progressive_enabler:
                scenario_context = self._create_scenario_context(
                    state, current_step, agent_name
                )
                self.progressive_enabler.record_scenario_outcome(
                    scenario=scenario_context,
                    isolation_enabled=False,
                    execution_time=1.0,
                    token_savings=0,
                )

            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=response_content,
                            name=agent_name,
                        )
                    ],
                    "current_plan": updated_plan,
                },
                goto="research_team",
            )

        except Exception as e:
            logger.error(f"Non-isolation execution failed: {e}")
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"Research execution failed: {str(e)}",
                            name=agent_name,
                        )
                    ]
                },
                goto="research_team",
            )
