# -*- coding: utf-8 -*-
"""
Reflection Integration - Deep integration with workflow.

Implements automatic conversion of reflection results into new research steps,
following the Reflexion pattern for knowledge gap remediation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from src.utils.reflection.reflexion_models import (
    ReflexionResult,
    KnowledgeGap,
    KnowledgeGapCategory,
)

logger = logging.getLogger(__name__)


@dataclass
class FollowUpStep:
    """A follow-up research step generated from reflection."""

    title: str
    description: str
    need_search: bool = True
    step_type: str = "research"
    priority: int = 3
    source_gap: str = ""  # Description of the originating knowledge gap


class ReflectionStepGenerator:
    """Generates research steps from reflection results.

    This class bridges the gap between reflection analysis and workflow execution
    by automatically converting identified knowledge gaps into actionable research steps.
    """

    def __init__(
        self,
        max_follow_up_steps: int = 3,
        min_priority_threshold: int = 3,
    ):
        self.max_follow_up_steps = max_follow_up_steps
        self.min_priority_threshold = min_priority_threshold

    def generate_steps_from_reflection(
        self,
        reflection_result: ReflexionResult,
        current_plan_steps: List[Any],
        research_topic: str,
    ) -> List[FollowUpStep]:
        """Generate follow-up research steps from reflection result.

        Args:
            reflection_result: The reflection analysis result
            current_plan_steps: Existing steps in the current plan
            research_topic: The research topic for context

        Returns:
            List of FollowUpStep objects to add to the plan
        """
        if reflection_result.is_sufficient:
            logger.info("Research is sufficient, no follow-up steps needed")
            return []

        existing_titles = set()
        for step in current_plan_steps:
            if hasattr(step, "title"):
                existing_titles.add(step.title.lower())
            if hasattr(step, "description"):
                existing_titles.add(step.description.lower()[:50])

        follow_up_steps = []
        prioritized_gaps = reflection_result.get_top_gaps(self.max_follow_up_steps)

        for gap in prioritized_gaps:
            if gap.priority < self.min_priority_threshold:
                logger.debug(f"Skipping low-priority gap: {gap.description[:50]}")
                continue

            step = self._create_step_from_gap(gap, existing_titles, research_topic)
            if step:
                follow_up_steps.append(step)
                existing_titles.add(step.title.lower())

        logger.info(f"Generated {len(follow_up_steps)} follow-up steps from reflection")
        return follow_up_steps

    def _create_step_from_gap(
        self,
        gap: KnowledgeGap,
        existing_titles: set,
        research_topic: str,
    ) -> Optional[FollowUpStep]:
        """Create a follow-up step from a knowledge gap.

        Args:
            gap: The knowledge gap to address
            existing_titles: Set of existing step titles (for deduplication)
            research_topic: The research topic for context

        Returns:
            FollowUpStep or None if the gap is already addressed
        """
        title = self._generate_step_title(gap)
        if title.lower() in existing_titles:
            logger.debug(f"Step already exists: {title}")
            return None

        description = self._generate_step_description(gap, research_topic)

        need_search = gap.category in [
            KnowledgeGapCategory.FACTUAL,
            KnowledgeGapCategory.TEMPORAL,
            KnowledgeGapCategory.CONTEXTUAL,
        ]

        return FollowUpStep(
            title=title,
            description=description,
            need_search=need_search,
            step_type="research",
            priority=gap.priority,
            source_gap=gap.description,
        )

    def _generate_step_title(self, gap: KnowledgeGap) -> str:
        """Generate a concise step title from a knowledge gap."""
        title = f"Follow-up: {gap.description[:50]}"
        if len(gap.description) > 50:
            title += "..."
        return title

    def _generate_step_description(
        self,
        gap: KnowledgeGap,
        research_topic: str,
    ) -> str:
        """Generate a detailed step description from a knowledge gap."""
        category_guidance = {
            KnowledgeGapCategory.FACTUAL: "Search for specific data, statistics, or verifiable facts.",
            KnowledgeGapCategory.CONTEXTUAL: "Research background information and related context.",
            KnowledgeGapCategory.ANALYTICAL: "Analyze and interpret the available information.",
            KnowledgeGapCategory.TEMPORAL: "Find recent developments and historical timeline.",
            KnowledgeGapCategory.METHODOLOGICAL: "Research methods, approaches, and best practices.",
        }

        guidance = category_guidance.get(gap.category, "Conduct additional research.")

        description = f"""Research Task: {gap.description}

Context: This addresses a knowledge gap identified in the research on "{research_topic}".

Suggested Query: {gap.suggested_query}

Guidance: {guidance}

Priority: {gap.priority}/5"""

        return description


def integrate_reflection_with_workflow(
    reflection_result: ReflexionResult,
    current_plan: Any,
    research_topic: str,
    max_steps: int = 3,
) -> Tuple[List[FollowUpStep], Dict[str, Any]]:
    """Integrate reflection result with workflow by generating follow-up steps.

    This is the main entry point for reflection-workflow integration.

    Args:
        reflection_result: The reflection analysis result
        current_plan: The current research plan
        research_topic: The research topic
        max_steps: Maximum number of follow-up steps to generate

    Returns:
        Tuple of (follow_up_steps, integration_metadata)
    """
    generator = ReflectionStepGenerator(max_follow_up_steps=max_steps)

    current_steps = []
    if current_plan and hasattr(current_plan, "steps"):
        current_steps = list(current_plan.steps)

    follow_up_steps = generator.generate_steps_from_reflection(
        reflection_result=reflection_result,
        current_plan_steps=current_steps,
        research_topic=research_topic,
    )

    metadata = {
        "reflection_sufficient": reflection_result.is_sufficient,
        "gaps_identified": len(reflection_result.knowledge_gaps),
        "steps_generated": len(follow_up_steps),
        "confidence_score": reflection_result.confidence_score,
        "gap_categories": [g.category.value for g in reflection_result.knowledge_gaps],
    }

    return follow_up_steps, metadata


def should_continue_research(
    reflection_result: ReflexionResult,
    current_loop: int,
    max_loops: int,
    confidence_threshold: float = 0.8,
) -> Tuple[bool, str]:
    """Determine if research should continue based on reflection result.

    Args:
        reflection_result: The reflection analysis result
        current_loop: Current reflection loop count
        max_loops: Maximum allowed loops
        confidence_threshold: Minimum confidence to consider research sufficient

    Returns:
        Tuple of (should_continue, reason)
    """
    if current_loop >= max_loops:
        return False, "max_loops_reached"

    if reflection_result.is_sufficient:
        return False, "research_sufficient"

    if reflection_result.confidence_score >= confidence_threshold:
        return False, "confidence_threshold_met"

    if not reflection_result.knowledge_gaps:
        return False, "no_gaps_identified"

    return True, "knowledge_gaps_exist"


def create_step_dict_from_follow_up(step: FollowUpStep) -> Dict[str, Any]:
    """Convert a FollowUpStep to a dictionary for plan integration.

    Args:
        step: The FollowUpStep to convert

    Returns:
        Dictionary representation suitable for plan step creation
    """
    return {
        "title": step.title,
        "description": step.description,
        "need_search": step.need_search,
        "step_type": step.step_type,
    }
