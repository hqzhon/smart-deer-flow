# -*- coding: utf-8 -*-
"""
Reflection Integration Module - Phase 1 Implementation

Integrates enhanced reflection capabilities with existing DeerFlow researcher components,
providing seamless integration with ResearcherContextExtension and other Phase 3 components.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from src.graph.types import State
from src.models.planner_model import Step, Plan
from .enhanced_reflection import (
    EnhancedReflectionAgent,
    ReflectionResult,
    ReflectionContext,
)
from ..researcher.researcher_progressive_enablement import (
    ResearcherProgressiveEnabler,
    ScenarioContext,
    TaskComplexity,
)
from ..researcher.researcher_isolation_metrics import ResearcherIsolationMetrics

logger = logging.getLogger(__name__)


@dataclass
class ReflectionIntegrationConfig:
    """Configuration for reflection integration"""

    enable_reflection_integration: bool = True
    reflection_trigger_threshold: int = 2  # Trigger reflection after N steps
    max_reflection_iterations: int = 3
    reflection_confidence_threshold: float = 0.7
    enable_progressive_reflection: bool = True
    enable_reflection_metrics: bool = True
    skip_initial_stage_reflection: bool = True  # Skip reflection in initial research stage
    initial_stage_min_observations: int = 2  # Minimum observations to exit initial stage
    initial_stage_min_content_length: int = 500  # Minimum content length to exit initial stage


class ReflectionIntegrator:
    """Integrates enhanced reflection with existing researcher components.

    This class provides the integration layer between the new reflection capabilities
    and existing DeerFlow components like ResearcherContextExtension and
    ResearcherProgressiveEnabler.
    """

    def __init__(
        self,
        reflection_agent: Optional[EnhancedReflectionAgent] = None,
        progressive_enabler: Optional[ResearcherProgressiveEnabler] = None,
        metrics: Optional[ResearcherIsolationMetrics] = None,
        config: Optional[ReflectionIntegrationConfig] = None,
    ):
        """Initialize the reflection integrator.

        Args:
            reflection_agent: Enhanced reflection agent instance
            progressive_enabler: Progressive enablement component
            metrics: Isolation metrics component
            config: Integration configuration
        """
        self.reflection_agent = reflection_agent or EnhancedReflectionAgent()
        self.progressive_enabler = progressive_enabler
        self.metrics = metrics
        self.config = config or ReflectionIntegrationConfig()

        # Integration state
        self.active_reflections: Dict[str, ReflectionContext] = {}
        self.reflection_sessions: Dict[str, List[ReflectionResult]] = {}

        logger.info(
            f"Initialized ReflectionIntegrator with integration_enabled={self.config.enable_reflection_integration}"
        )

    def should_trigger_reflection(
        self,
        state: State,
        current_step: Optional[Step] = None,
        agent_name: str = "researcher",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Determine if reflection should be triggered for the current state.

        Args:
            state: Current graph state
            current_step: Current step being executed
            agent_name: Name of the agent

        Returns:
            Tuple of (should_trigger, reason, decision_factors)
        """
        if not self.config.enable_reflection_integration:
            return (
                False,
                "Reflection integration disabled",
                {"integration_enabled": False},
            )

        decision_factors = {
            "integration_enabled": True,
            "step_count": len(state.get("observations", [])),
            "has_plan": state.get("current_plan") is not None,
            "has_results": len(state.get("observations", [])) > 0,
        }

        # Check if we're in the initial research stage - skip reflection
        if self.config.skip_initial_stage_reflection and self._is_initial_research_stage(state, current_step):
            decision_factors["trigger_reason"] = "initial_stage_skip"
            return (
                False,
                "Skipping reflection in initial research stage",
                decision_factors,
            )

        # Check step count threshold
        step_count = len(state.get("observations", []))
        if step_count >= self.config.reflection_trigger_threshold:
            decision_factors["trigger_reason"] = "step_count_threshold"
            return (
                True,
                f"Step count ({step_count}) exceeds threshold ({self.config.reflection_trigger_threshold})",
                decision_factors,
            )

        # Check if progressive enabler suggests reflection
        if self.config.enable_progressive_reflection and self.progressive_enabler:
            scenario_context = self._create_scenario_context(
                state, current_step, agent_name
            )
            should_enable, reason, enabler_factors = (
                self.progressive_enabler.should_enable_isolation(scenario_context)
            )

            decision_factors.update(enabler_factors)

            if should_enable:
                decision_factors["trigger_reason"] = "progressive_enabler"
                return (
                    True,
                    f"Progressive enabler suggests reflection: {reason}",
                    decision_factors,
                )

        # Check for complex research scenarios
        if self._is_complex_research_scenario(state, current_step):
            decision_factors["trigger_reason"] = "complex_scenario"
            return True, "Complex research scenario detected", decision_factors

        return False, "No reflection trigger conditions met", decision_factors

    async def execute_reflection_analysis(
        self,
        state: State,
        session_id: str,
        current_step: Optional[Step] = None,
        runnable_config: Optional[RunnableConfig] = None,
    ) -> Tuple[ReflectionResult, ReflectionContext]:
        """Execute reflection analysis for the current research state.

        Args:
            state: Current graph state
            current_step: Current step being executed
            session_id: Unique session identifier
            runnable_config: Configuration for LLM execution

        Returns:
            Tuple of (reflection_result, reflection_context)
        """
        # Create reflection context
        context = self._build_reflection_context(state, current_step, session_id)

        # Store active reflection
        self.active_reflections[session_id] = context

        # Start metrics tracking if enabled
        if self.config.enable_reflection_metrics and self.metrics:
            self.metrics.start_isolation_session(
                session_id=f"reflection_{session_id}",
                task_complexity=self._assess_task_complexity(context).value,
                isolation_level="reflection",
            )

        try:
            # Execute reflection analysis
            reflection_result = await self.reflection_agent.analyze_knowledge_gaps(
                context=context, runnable_config=runnable_config
            )

            # Store reflection result
            if session_id not in self.reflection_sessions:
                self.reflection_sessions[session_id] = []
            self.reflection_sessions[session_id].append(reflection_result)

            # Update metrics
            if self.config.enable_reflection_metrics and self.metrics:
                self.metrics.update_session_context(
                    session_id=f"reflection_{session_id}",
                    original_size=len(str(state)),
                    compressed_size=len(str(reflection_result.knowledge_gaps)),
                )

            logger.info(
                f"Reflection analysis completed for session {session_id}: "
                f"sufficient={reflection_result.is_sufficient}, "
                f"confidence={reflection_result.confidence_score:.2f}"
            )

            return reflection_result, context

        except Exception as e:
            logger.error(f"Error in reflection analysis for session {session_id}: {e}")

            # End metrics session with error
            if self.config.enable_reflection_metrics and self.metrics:
                self.metrics.end_isolation_session(
                    session_id=f"reflection_{session_id}",
                    success=False,
                    error_message=str(e),
                )

            raise

    def integrate_reflection_with_plan_update(
        self,
        current_plan: Plan,
        reflection_result: ReflectionResult,
        context: ReflectionContext,
    ) -> Plan:
        """Integrate reflection results with plan updates.

        Args:
            current_plan: Current research plan
            reflection_result: Result from reflection analysis
            context: Reflection context

        Returns:
            Updated plan with reflection-driven modifications
        """
        if reflection_result.is_sufficient:
            logger.debug(
                "Reflection indicates research is sufficient, no plan updates needed"
            )
            return current_plan

        # Create new steps based on follow-up queries
        new_steps = []
        for i, query in enumerate(reflection_result.follow_up_queries):
            new_step = Step(
                need_search=True,
                title=f"Follow-up Research: {query[:50]}...",
                description=f"Research query generated by reflection analysis: {query}",
                step_type="research",
            )
            new_steps.append(new_step)

        # Update plan with new steps
        updated_steps = list(current_plan.steps) + new_steps

        # Update plan thought with reflection insights
        reflection_insight = f"\n\nReflection Analysis: {', '.join(reflection_result.recommendations) if reflection_result.recommendations else 'No specific recommendations'}"
        if reflection_result.knowledge_gaps:
            reflection_insight += f"\nIdentified Knowledge Gaps: {', '.join(reflection_result.knowledge_gaps)}"

        updated_plan = Plan(
            locale=current_plan.locale,
            has_enough_context=reflection_result.is_sufficient,
            thought=current_plan.thought + reflection_insight,
            title=current_plan.title,
            steps=updated_steps,
        )

        logger.info(
            f"Plan updated with reflection insights: added {len(new_steps)} new steps"
        )
        return updated_plan

    def finalize_reflection_session(
        self, session_id: str, success: bool = True, performance_impact: float = 0.0
    ) -> Dict[str, Any]:
        """Finalize a reflection session and clean up resources.

        Args:
            session_id: Session identifier to finalize
            success: Whether the session completed successfully
            performance_impact: Performance impact measurement

        Returns:
            Session summary dictionary
        """
        session_summary = {
            "session_id": session_id,
            "success": success,
            "performance_impact": performance_impact,
            "reflection_count": 0,
            "final_sufficient": False,
        }

        # Get reflection results
        if session_id in self.reflection_sessions:
            results = self.reflection_sessions[session_id]
            session_summary["reflection_count"] = len(results)
            if results:
                session_summary["final_sufficient"] = results[-1].is_sufficient

            # Clean up session data
            del self.reflection_sessions[session_id]

        # Clean up active reflection
        if session_id in self.active_reflections:
            del self.active_reflections[session_id]

        # Finalize metrics
        if self.config.enable_reflection_metrics and self.metrics:
            self.metrics.end_isolation_session(
                session_id=f"reflection_{session_id}",
                success=success,
                performance_impact=performance_impact,
            )

        logger.info(f"Finalized reflection session {session_id}: {session_summary}")
        return session_summary

    def _create_scenario_context(
        self, state: State, current_step: Step, agent_name: str
    ) -> ScenarioContext:
        """Create scenario context for progressive enablement."""
        state.get("current_plan")
        observations = state.get("observations", [])

        return ScenarioContext(
            task_description=current_step.description,
            step_count=len(observations),
            context_size=len(str(state)),
            parallel_execution=getattr(state, "enable_parallel_execution", False),
            has_search_results=len(state.get("resources", [])) > 0,
            has_complex_queries=current_step.need_search,
            estimated_tokens=len(str(state)) // 4,  # Rough token estimation
        )

    def _build_reflection_context(
        self, state: State, current_step: Step, session_id: str
    ) -> ReflectionContext:
        """Build reflection context from current state."""
        current_plan = state.get("current_plan")
        observations = state.get("observations", [])

        # Extract completed steps
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

        return ReflectionContext(
            research_topic=state.get("research_topic", "Unknown"),
            completed_steps=completed_steps,
            current_step=(
                {"step": current_step.title, "description": current_step.description}
                if current_step
                else None
            ),
            execution_results=observations,
            observations=observations,
            resources_found=len(state.get("resources", [])),
            total_steps=(
                len(current_plan.steps)
                if current_plan and hasattr(current_plan, "steps")
                else 1
            ),
            current_step_index=len(completed_steps),
            locale=state.get("locale", "en-US"),
            max_reflection_loops=self.config.max_reflection_iterations,
        )

    def _is_initial_research_stage(self, state: State, current_step: Optional[Step] = None) -> bool:
        """Determine if we're in the initial research stage where reflection should be skipped.
        
        Args:
            state: Current graph state
            current_step: Current step being executed
            
        Returns:
            True if in initial research stage, False otherwise
        """
        # Check if we have minimal observations
        observations = state.get("observations", [])
        if len(observations) < self.config.initial_stage_min_observations:
            return True
            
        # Check if we have no meaningful execution results
        current_plan = state.get("current_plan")
        if current_plan and hasattr(current_plan, "steps"):
            executed_steps = [step for step in current_plan.steps if step.execution_res]
            if len(executed_steps) < self.config.initial_stage_min_observations:
                return True
                
        # Check if we have minimal resources found
        resources = state.get("resources", [])
        if len(resources) < self.config.initial_stage_min_observations:
            return True
            
        # Check if total research content is minimal
        total_content_length = sum(len(str(obs)) for obs in observations)
        if total_content_length < self.config.initial_stage_min_content_length:
            return True
            
        return False

    def _is_complex_research_scenario(self, state: State, current_step: Step) -> bool:
        """Determine if the current scenario is complex enough to warrant reflection."""
        complexity_indicators = 0

        # Check for multiple resources
        if len(state.get("resources", [])) >= 3:
            complexity_indicators += 1

        # Check for long observations
        observations = state.get("observations", [])
        if any(len(obs) > 1000 for obs in observations):
            complexity_indicators += 1

        # Check for search requirements
        if current_step.need_search:
            complexity_indicators += 1

        # Check for plan iterations
        if state.get("plan_iterations", 0) > 1:
            complexity_indicators += 1

        return complexity_indicators >= 2

    def _assess_task_complexity(self, context: ReflectionContext) -> TaskComplexity:
        """Assess task complexity for metrics."""
        if self.progressive_enabler:
            scenario_context = ScenarioContext(
                task_description=context.research_topic,
                step_count=context.total_steps,
                context_size=len(str(context)),
                parallel_execution=False,
                has_search_results=context.resources_found > 0,
                has_complex_queries=True,
                estimated_tokens=len(str(context)) // 4,
            )
            return self.progressive_enabler.analyze_task_complexity(scenario_context)

        # Fallback assessment
        if context.total_steps >= 5 or context.resources_found >= 5:
            return TaskComplexity.COMPLEX
        elif context.total_steps >= 3 or context.resources_found >= 3:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get metrics about reflection integration performance."""
        active_sessions = len(self.active_reflections)
        total_sessions = len(self.reflection_sessions)

        reflection_metrics = self.reflection_agent.get_reflection_metrics()

        return {
            "integration_enabled": self.config.enable_reflection_integration,
            "active_reflection_sessions": active_sessions,
            "total_reflection_sessions": total_sessions,
            "reflection_agent_metrics": reflection_metrics,
            "progressive_reflection_enabled": self.config.enable_progressive_reflection,
            "reflection_metrics_enabled": self.config.enable_reflection_metrics,
        }
