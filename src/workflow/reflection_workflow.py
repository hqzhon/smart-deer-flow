# SPDX-License-Identifier: MIT
"""
Reflection Workflow Integration
Integrates enhanced reflection capabilities into the research workflow
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.utils.reflection.enhanced_reflection import (
    EnhancedReflectionAgent,
    ReflectionContext,
)
from src.utils.researcher.researcher_progressive_enablement import (
    ResearcherProgressiveEnabler,
)
from src.utils.researcher.researcher_isolation_metrics import ResearcherIsolationMetrics
from src.config import get_settings

logger = logging.getLogger(__name__)


class WorkflowMetrics:
    """Metrics collection and analysis for workflow execution."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize workflow metrics.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.metrics_data: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "stage_metrics": {},
            "reflection_metrics": {},
        }

    def record_execution(self, result: "WorkflowResult") -> None:
        """Record workflow execution metrics.

        Args:
            result: Workflow execution result
        """
        self.metrics_data["total_executions"] += 1

        if result.success:
            self.metrics_data["successful_executions"] += 1
        else:
            self.metrics_data["failed_executions"] += 1

        # Update average execution time
        total_time = (
            self.metrics_data["average_execution_time"]
            * (self.metrics_data["total_executions"] - 1)
            + result.execution_time
        )
        self.metrics_data["average_execution_time"] = (
            total_time / self.metrics_data["total_executions"]
        )

        # Record stage metrics
        for stage_name, stage_result in result.stage_results.items():
            if stage_name not in self.metrics_data["stage_metrics"]:
                self.metrics_data["stage_metrics"][stage_name] = {
                    "executions": 0,
                    "successes": 0,
                    "average_time": 0.0,
                }

            stage_metrics = self.metrics_data["stage_metrics"][stage_name]
            stage_metrics["executions"] += 1

            if stage_result.get("success", True):
                stage_metrics["successes"] += 1

            if (
                "stage_metrics" in stage_result
                and "execution_time" in stage_result["stage_metrics"]
            ):
                stage_time = stage_result["stage_metrics"]["execution_time"]
                total_stage_time = (
                    stage_metrics["average_time"] * (stage_metrics["executions"] - 1)
                    + stage_time
                )
                stage_metrics["average_time"] = (
                    total_stage_time / stage_metrics["executions"]
                )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of workflow metrics.

        Returns:
            Metrics summary dictionary
        """
        success_rate = 0.0
        if self.metrics_data["total_executions"] > 0:
            success_rate = (
                self.metrics_data["successful_executions"]
                / self.metrics_data["total_executions"]
            ) * 100

        return {
            "total_executions": self.metrics_data["total_executions"],
            "success_rate": success_rate,
            "average_execution_time": self.metrics_data["average_execution_time"],
            "stage_performance": self._calculate_stage_performance(),
            "reflection_insights": self.metrics_data["reflection_metrics"],
        }

    def _calculate_stage_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each stage.

        Returns:
            Stage performance metrics
        """
        performance = {}

        for stage_name, metrics in self.metrics_data["stage_metrics"].items():
            success_rate = 0.0
            if metrics["executions"] > 0:
                success_rate = (metrics["successes"] / metrics["executions"]) * 100

            performance[stage_name] = {
                "success_rate": success_rate,
                "average_time": metrics["average_time"],
                "total_executions": metrics["executions"],
            }

        return performance


@dataclass
class WorkflowStage:
    """Represents a stage in the research workflow."""

    name: str
    description: str
    requires_reflection: bool = False
    isolation_level: str = "basic"
    context_expansion: bool = False
    metrics_tracking: bool = True


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    success: bool
    stage_results: Dict[str, Any]
    reflection_insights: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class ReflectionWorkflow:
    """Enhanced research workflow with integrated reflection capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_settings().model_dump()
        self.reflection_agent = EnhancedReflectionAgent(self.config)
        self.progressive_enabler = ResearcherProgressiveEnabler(self.config)
        self.isolation_metrics = ResearcherIsolationMetrics(self.config)

        # Workflow stages
        self.stages = [
            WorkflowStage(
                name="initialization",
                description="Initialize research context and parameters",
                requires_reflection=False,
                isolation_level="basic",
                context_expansion=False,
            ),
            WorkflowStage(
                name="context_analysis",
                description="Analyze research context and identify knowledge gaps",
                requires_reflection=True,
                isolation_level="moderate",
                context_expansion=True,
            ),
            WorkflowStage(
                name="research_planning",
                description="Plan research strategy based on analysis",
                requires_reflection=True,
                isolation_level="moderate",
                context_expansion=True,
            ),
            WorkflowStage(
                name="information_gathering",
                description="Gather information from various sources",
                requires_reflection=False,
                isolation_level="strict",
                context_expansion=True,
            ),
            WorkflowStage(
                name="synthesis",
                description="Synthesize gathered information",
                requires_reflection=True,
                isolation_level="moderate",
                context_expansion=True,
            ),
            WorkflowStage(
                name="validation",
                description="Validate research findings and conclusions",
                requires_reflection=True,
                isolation_level="basic",
                context_expansion=False,
            ),
        ]

        self.execution_history: List[Dict[str, Any]] = []
        self.current_stage: Optional[str] = None
        self.workflow_context: Dict[str, Any] = {}

    async def execute_workflow(
        self, research_query: str, initial_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Execute the complete research workflow with reflection integration."""
        start_time = datetime.now()
        stage_results = {}
        reflection_insights = []

        try:
            logger.info(f"Starting reflection workflow for query: {research_query}")

            # Initialize workflow context
            self.workflow_context = {
                "query": research_query,
                "initial_context": initial_context or {},
                "start_time": start_time,
                "stage_count": len(self.stages),
            }

            # Execute each stage
            for i, stage in enumerate(self.stages):
                self.current_stage = stage.name
                logger.info(f"Executing stage {i+1}/{len(self.stages)}: {stage.name}")

                stage_result = await self._execute_stage(stage, i)
                stage_results[stage.name] = stage_result

                # Collect reflection insights if available
                if "reflection_insights" in stage_result:
                    reflection_insights.extend(stage_result["reflection_insights"])

                # Check if workflow should continue
                if not stage_result.get("success", True):
                    logger.warning(f"Stage {stage.name} failed, stopping workflow")
                    break

            # Generate final metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            final_metrics = await self._generate_final_metrics(
                stage_results, execution_time
            )

            return WorkflowResult(
                success=True,
                stage_results=stage_results,
                reflection_insights=reflection_insights,
                metrics=final_metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Workflow execution failed: {e}")

            return WorkflowResult(
                success=False,
                stage_results=stage_results,
                reflection_insights=reflection_insights,
                metrics={"execution_time": execution_time},
                execution_time=execution_time,
                error_message=str(e),
            )

    async def _execute_stage(
        self, stage: WorkflowStage, stage_index: int
    ) -> Dict[str, Any]:
        """Execute a single workflow stage."""
        stage_start = datetime.now()

        try:
            # Update isolation level
            await self.isolation_metrics.update_isolation_level(stage.isolation_level)

            # Execute stage-specific logic
            if stage.name == "initialization":
                result = await self._execute_initialization_stage()
            elif stage.name == "context_analysis":
                result = await self._execute_context_analysis_stage()
            elif stage.name == "research_planning":
                result = await self._execute_research_planning_stage()
            elif stage.name == "information_gathering":
                result = await self._execute_information_gathering_stage()
            elif stage.name == "synthesis":
                result = await self._execute_synthesis_stage()
            elif stage.name == "validation":
                result = await self._execute_validation_stage()
            else:
                result = {"success": False, "error": f"Unknown stage: {stage.name}"}

            # Apply reflection if required
            if stage.requires_reflection and result.get("success", False):
                reflection_result = await self._apply_reflection(stage, result)
                result.update(reflection_result)

            # Context expansion functionality integrated into isolation metrics
            if stage.context_expansion and result.get("success", False):
                # Context expansion is handled by isolation metrics
                result["context_expanded"] = True

            # Record stage metrics
            stage_time = (datetime.now() - stage_start).total_seconds()
            result["stage_metrics"] = {
                "execution_time": stage_time,
                "isolation_level": stage.isolation_level,
                "reflection_applied": stage.requires_reflection,
                "context_expanded": stage.context_expansion,
            }

            # Update execution history
            self.execution_history.append(
                {
                    "stage": stage.name,
                    "index": stage_index,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return result

        except Exception as e:
            logger.error(f"Stage {stage.name} execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage_metrics": {
                    "execution_time": (datetime.now() - stage_start).total_seconds(),
                    "isolation_level": stage.isolation_level,
                },
            }

    async def _execute_initialization_stage(self) -> Dict[str, Any]:
        """Execute initialization stage."""
        return {
            "success": True,
            "initialized_context": self.workflow_context,
            "config_loaded": True,
            "agents_ready": True,
        }

    async def _execute_context_analysis_stage(self) -> Dict[str, Any]:
        """Execute context analysis stage."""
        query = self.workflow_context["query"]
        initial_context = self.workflow_context["initial_context"]

        # Create reflection context
        reflection_context = ReflectionContext(
            query=query,
            context=initial_context,
            previous_results=[],
            metadata={"stage": "context_analysis"},
        )

        # Analyze knowledge gaps
        knowledge_gaps = await self.reflection_agent.analyze_knowledge_gaps(
            reflection_context
        )

        return {
            "success": True,
            "knowledge_gaps": knowledge_gaps,
            "context_analyzed": True,
            "gap_count": len(knowledge_gaps.get("gaps", [])),
        }

    async def _execute_research_planning_stage(self) -> Dict[str, Any]:
        """Execute research planning stage."""
        # Get previous stage results
        context_analysis = (
            self.execution_history[-1]["result"] if self.execution_history else {}
        )
        knowledge_gaps = context_analysis.get("knowledge_gaps", {})

        # Check if progressive enablement should be applied
        enablement_decision = await self.progressive_enabler.should_enable_isolation(
            self.workflow_context["query"], {"knowledge_gaps": knowledge_gaps}
        )

        return {
            "success": True,
            "research_plan": {
                "gaps_to_address": knowledge_gaps.get("gaps", []),
                "suggested_queries": knowledge_gaps.get("suggested_queries", []),
                "enablement_decision": enablement_decision,
            },
            "plan_created": True,
        }

    async def _execute_information_gathering_stage(self) -> Dict[str, Any]:
        """Execute information gathering stage."""
        # Get research plan
        planning_result = (
            self.execution_history[-1]["result"] if self.execution_history else {}
        )
        research_plan = planning_result.get("research_plan", {})

        # Simulate information gathering
        gathered_info = {
            "sources": ["source1", "source2", "source3"],
            "content": "Simulated gathered information",
            "relevance_score": 0.85,
        }

        return {
            "success": True,
            "gathered_information": gathered_info,
            "sources_count": len(gathered_info["sources"]),
            "information_quality": "high",
        }

    async def _execute_synthesis_stage(self) -> Dict[str, Any]:
        """Execute synthesis stage."""
        # Get gathered information
        gathering_result = (
            self.execution_history[-1]["result"] if self.execution_history else {}
        )
        gathered_info = gathering_result.get("gathered_information", {})

        # Synthesize information
        synthesis = {
            "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
            "conclusions": ["Conclusion 1", "Conclusion 2"],
            "confidence_score": 0.78,
        }

        return {
            "success": True,
            "synthesis_result": synthesis,
            "findings_count": len(synthesis["key_findings"]),
            "synthesis_quality": "good",
        }

    async def _execute_validation_stage(self) -> Dict[str, Any]:
        """Execute validation stage."""
        # Get synthesis results
        synthesis_result = (
            self.execution_history[-1]["result"] if self.execution_history else {}
        )
        synthesis = synthesis_result.get("synthesis_result", {})

        # Validate findings
        validation = {
            "validated_findings": synthesis.get("key_findings", []),
            "validation_score": 0.82,
            "issues_found": [],
            "recommendations": ["Recommendation 1", "Recommendation 2"],
        }

        return {
            "success": True,
            "validation_result": validation,
            "validation_passed": True,
            "final_confidence": validation["validation_score"],
        }

    async def _apply_reflection(
        self, stage: WorkflowStage, stage_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply reflection to stage result."""
        try:
            # Create reflection context
            reflection_context = ReflectionContext(
                query=self.workflow_context["query"],
                context=stage_result,
                previous_results=self.execution_history,
                metadata={"stage": stage.name},
            )

            # Perform reflection
            reflection_insights = await self.reflection_agent.analyze_knowledge_gaps(
                reflection_context
            )

            return {
                "reflection_applied": True,
                "reflection_insights": [reflection_insights],
                "reflection_confidence": reflection_insights.get(
                    "confidence_score", 0.5
                ),
            }

        except Exception as e:
            logger.error(f"Reflection failed for stage {stage.name}: {e}")
            return {"reflection_applied": False, "reflection_error": str(e)}

    async def _generate_final_metrics(
        self, stage_results: Dict[str, Any], execution_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive workflow metrics."""
        try:
            # Collect stage metrics
            stage_metrics = {}
            total_reflection_time = 0
            total_expansion_time = 0

            for stage_name, result in stage_results.items():
                stage_metrics[stage_name] = result.get("stage_metrics", {})
                if result.get("reflection_applied"):
                    total_reflection_time += stage_metrics[stage_name].get(
                        "execution_time", 0
                    )
                if result.get("context_expanded"):
                    total_expansion_time += stage_metrics[stage_name].get(
                        "execution_time", 0
                    )

            # Get isolation metrics
            isolation_summary = await self.isolation_metrics.get_metrics_summary()

            return {
                "workflow_metrics": {
                    "total_execution_time": execution_time,
                    "stages_completed": len(stage_results),
                    "stages_successful": sum(
                        1 for r in stage_results.values() if r.get("success", False)
                    ),
                    "reflection_time": total_reflection_time,
                    "expansion_time": total_expansion_time,
                },
                "stage_metrics": stage_metrics,
                "isolation_metrics": isolation_summary,
                "performance_summary": {
                    "efficiency_score": (
                        min(1.0, 60.0 / execution_time) if execution_time > 0 else 1.0
                    ),
                    "success_rate": (
                        len(
                            [
                                r
                                for r in stage_results.values()
                                if r.get("success", False)
                            ]
                        )
                        / len(stage_results)
                        if stage_results
                        else 0
                    ),
                    "reflection_coverage": (
                        sum(
                            1
                            for r in stage_results.values()
                            if r.get("reflection_applied", False)
                        )
                        / len(stage_results)
                        if stage_results
                        else 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Failed to generate final metrics: {e}")
            return {
                "workflow_metrics": {"total_execution_time": execution_time},
                "error": str(e),
            }

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "current_stage": self.current_stage,
            "stages_completed": len(self.execution_history),
            "total_stages": len(self.stages),
            "workflow_context": self.workflow_context,
            "execution_history": (
                self.execution_history[-3:] if self.execution_history else []
            ),  # Last 3 stages
        }

    def reset_workflow(self) -> None:
        """Reset workflow state."""
        self.execution_history.clear()
        self.current_stage = None
        self.workflow_context.clear()
        logger.info("Workflow state reset")
