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
    ScenarioContext,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


@dataclass
class ReflectionIntegrationConfig:
    """简化的反射集成配置。"""

    enable_reflection_integration: bool = True
    max_reflection_iterations: int = 2
    reflection_timeout: int = 30


class ReflectionIntegrator:
    """Integrates enhanced reflection with existing researcher components.

    This class provides the integration layer between the new reflection capabilities
    and existing DeerFlow components like ResearcherContextExtension and
    ResearcherProgressiveEnabler.
    """

    def __init__(
        self,
        reflection_agent: Optional[EnhancedReflectionAgent] = None,
        config: Optional[ReflectionIntegrationConfig] = None,
    ):
        """初始化简化的反射集成器。

        Args:
            reflection_agent: 增强反射代理实例
            config: 集成配置
        """
        self.reflection_agent = reflection_agent or EnhancedReflectionAgent()
        self.config = config or ReflectionIntegrationConfig()

        # 简化的集成状态
        self.active_reflections: Dict[str, ReflectionContext] = {}
        self.reflection_sessions: Dict[str, List[ReflectionResult]] = {}

        logger.info(
            f"Initialized simplified ReflectionIntegrator with integration_enabled={self.config.enable_reflection_integration}"
        )

    def should_trigger_reflection(
        self,
        state: State,
        current_step: Optional[Step] = None,
        agent_name: str = "researcher",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Determine if reflection should be triggered for the current state.

        简化版本：主要基于步骤计数和观察结果数量来决定是否触发反射。

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

        observations = state.get("observations", [])
        step_count = len(observations)

        decision_factors = {
            "integration_enabled": True,
            "step_count": step_count,
            "has_plan": state.get("current_plan") is not None,
            "has_results": step_count > 0,
        }

        logger.info(f"[DEBUG] Reflection trigger check: step_count={step_count}")

        # 简化的触发条件：只要有观察结果就触发
        if step_count > 0:
            decision_factors["trigger_reason"] = "has_observations"
            return (
                True,
                f"Step count ({step_count}) has observations for reflection",
                decision_factors,
            )

        # 没有观察结果
        decision_factors["trigger_reason"] = "no_observations"
        return (
            False,
            "No observations available for reflection",
            decision_factors,
        )

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
        # Ensure recommendations are all strings before joining
        safe_recommendations = []
        if reflection_result.recommendations:
            for rec in reflection_result.recommendations:
                if isinstance(rec, str):
                    safe_recommendations.append(rec)
                elif isinstance(rec, dict):
                    # Extract meaningful string from dict
                    if "description" in rec:
                        safe_recommendations.append(str(rec["description"]))
                    elif "recommendation" in rec:
                        safe_recommendations.append(str(rec["recommendation"]))
                    elif "action" in rec:
                        safe_recommendations.append(str(rec["action"]))
                    else:
                        safe_recommendations.append(str(rec))
                else:
                    safe_recommendations.append(str(rec))
        
        reflection_insight = f"\n\nReflection Analysis: {', '.join(safe_recommendations) if safe_recommendations else 'No specific recommendations'}"
        if reflection_result.knowledge_gaps:
            # Ensure knowledge_gaps are all strings before joining
            safe_gaps = []
            for gap in reflection_result.knowledge_gaps:
                if isinstance(gap, str):
                    safe_gaps.append(gap)
                elif isinstance(gap, dict):
                    # Extract meaningful string from dict
                    if "description" in gap:
                        safe_gaps.append(str(gap["description"]))
                    elif "gap_type" in gap:
                        safe_gaps.append(str(gap["gap_type"]))
                    else:
                        safe_gaps.append(str(gap))
                else:
                    safe_gaps.append(str(gap))
            reflection_insight += f"\nIdentified Knowledge Gaps: {', '.join(safe_gaps)}"

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

    def _is_initial_research_stage(
        self, state: State, current_step: Optional[Step] = None
    ) -> bool:
        """Determine if we're in the initial research stage where reflection should be skipped.

        Only checks if we're in the first step - all other conditions have been removed.

        Args:
            state: Current graph state
            current_step: Current step being executed

        Returns:
            True if in initial research stage (first 3 steps), False otherwise
        """
        logger.info("[DEBUG] Checking if in initial research stage...")

        # Only check: Ensure we're not in the very first few steps
        observations = state.get("observations", [])
        step_count = len(observations)
        logger.info(f"[DEBUG] Step count: {step_count}, hardcoded minimum: 3")
        if step_count < 1:  # Always skip reflection for first steps
            logger.info("[DEBUG] Step count < 1 - in initial stage")
            return True

        logger.info("[DEBUG] Not in initial research stage - reflection can proceed")
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

    def _has_actual_execution_results(
        self, state: State, current_plan: Optional[Plan]
    ) -> bool:
        """Check if there are actual execution results from completed steps.

        Args:
            state: Current graph state
            current_plan: Current research plan

        Returns:
            True if there are actual execution results, False otherwise
        """
        logger.info("[DEBUG] Checking for actual execution results...")

        # Check if we have observations with meaningful content
        observations = state.get("observations", [])
        logger.info(f"[DEBUG] Found {len(observations)} observations")
        if not observations:
            logger.info("[DEBUG] No observations found - returning False")
            return False

        # Check if we have executed steps with results
        if current_plan and hasattr(current_plan, "steps"):
            executed_steps = [step for step in current_plan.steps if step.execution_res]
            logger.info(
                f"[DEBUG] Found {len(executed_steps)} executed steps with results"
            )
            if not executed_steps:
                logger.info("[DEBUG] No executed steps with results - returning False")
                return False

            # Check if execution results exist (no length requirement)
            for i, step in enumerate(executed_steps):
                logger.info(
                    f"[DEBUG] Step {i+1}: '{step.title}' has execution result: {bool(step.execution_res)}"
                )
                if step.execution_res:
                    logger.info(
                        f"[DEBUG] Found execution result in step '{step.title}' - returning True"
                    )
                    return True
        else:
            logger.info("[DEBUG] No current plan or plan has no steps")

        # Check if observations exist (no length requirement)
        logger.info(f"[DEBUG] Found {len(observations)} observations")

        for i, obs in enumerate(observations[:3]):  # Log first 3 observations
            obs_preview = str(obs)[:100] + "..." if len(str(obs)) > 100 else str(obs)
            logger.info(f"[DEBUG] Observation {i+1} preview: {obs_preview}")

        if len(observations) < 1:
            logger.info("[DEBUG] No observations found - returning False")
            return False

        logger.info("[DEBUG] Found observations - returning True")
        return True

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
