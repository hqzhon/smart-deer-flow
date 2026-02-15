# -*- coding: utf-8 -*-
"""
Smart Reflection Controller - Intelligent loop control for reflection.

Implements dynamic reflection loop control based on:
- Confidence scores
- Research complexity
- Improvement trends
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .reflexion_models import (
    ReflexionResult,
    SmartReflectionControllerConfig,
    KnowledgeGap,
)

logger = logging.getLogger(__name__)


@dataclass
class ReflectionHistory:
    """History entry for reflection analysis."""

    loop_count: int
    confidence_score: float
    is_sufficient: bool
    gaps_count: int
    improvement: float = 0.0


class SmartReflectionController:
    """Smart reflection loop controller with dynamic adjustment.

    Controls reflection loops based on:
    1. Minimum/maximum loop constraints
    2. Confidence threshold for early termination
    3. Improvement trend analysis
    4. Research complexity estimation
    """

    def __init__(self, config: Optional[SmartReflectionControllerConfig] = None):
        self.config = config or SmartReflectionControllerConfig()
        self.history: List[ReflectionHistory] = []
        self._loop_count = 0

    def should_continue_reflection(
        self, current_result: ReflexionResult
    ) -> Tuple[bool, str]:
        """Determine if reflection should continue.

        Args:
            current_result: Current reflection result

        Returns:
            Tuple of (should_continue, reason)
        """
        self._loop_count += 1

        improvement = 0.0
        if self.history:
            prev_confidence = self.history[-1].confidence_score
            improvement = current_result.confidence_score - prev_confidence

        history_entry = ReflectionHistory(
            loop_count=self._loop_count,
            confidence_score=current_result.confidence_score,
            is_sufficient=current_result.is_sufficient,
            gaps_count=len(current_result.knowledge_gaps),
            improvement=improvement,
        )
        self.history.append(history_entry)

        if self._loop_count < self.config.min_loops:
            return True, "minimum_loops_not_reached"

        if current_result.confidence_score >= self.config.confidence_threshold:
            return False, "confidence_threshold_met"

        if current_result.is_sufficient:
            return False, "research_sufficient"

        if len(self.history) >= 2:
            if improvement < self.config.improvement_threshold:
                return False, "diminishing_returns"

        if self._loop_count >= self.config.max_loops:
            return False, "max_loops_reached"

        return True, "continue_reflection"

    def estimate_required_loops(
        self, research_topic: str, step_count: int, research_steps: int
    ) -> int:
        """Estimate required reflection loops based on research complexity.

        Args:
            research_topic: The research topic
            step_count: Total number of steps in the plan
            research_steps: Number of steps requiring research

        Returns:
            Estimated number of reflection loops
        """
        complexity_score = 0.0

        complexity_score += min(step_count * 0.3, 2.0)

        topic_words = len(research_topic.split())
        complexity_score += min(topic_words * 0.05, 1.0)

        complexity_score += min(research_steps * 0.2, 1.0)

        recommended_loops = max(
            self.config.min_loops, min(int(complexity_score) + 1, self.config.max_loops)
        )

        logger.info(
            f"Estimated reflection loops: {recommended_loops} "
            f"(complexity_score={complexity_score:.2f}, "
            f"steps={step_count}, research_steps={research_steps})"
        )

        return recommended_loops

    def get_loop_stats(self) -> dict:
        """Get statistics about reflection loops."""
        if not self.history:
            return {
                "total_loops": 0,
                "final_confidence": 0.0,
                "total_improvement": 0.0,
            }

        return {
            "total_loops": self._loop_count,
            "final_confidence": self.history[-1].confidence_score,
            "total_improvement": sum(h.improvement for h in self.history),
            "avg_improvement": sum(h.improvement for h in self.history)
            / len(self.history),
            "sufficient_at_loop": next(
                (h.loop_count for h in self.history if h.is_sufficient), None
            ),
        }

    def reset(self) -> None:
        """Reset the controller for a new reflection session."""
        self.history.clear()
        self._loop_count = 0


class KnowledgeGapPrioritizer:
    """Prioritizes knowledge gaps for focused research."""

    def __init__(self, max_gaps: int = 5):
        self.max_gaps = max_gaps

    def prioritize(
        self,
        gaps: List[KnowledgeGap],
        research_topic: str,
        completed_queries: List[str] = None,
    ) -> List[KnowledgeGap]:
        """Prioritize knowledge gaps based on multiple factors.

        Args:
            gaps: List of knowledge gaps
            research_topic: The research topic for context
            completed_queries: Already executed queries to avoid duplicates

        Returns:
            Prioritized list of knowledge gaps
        """
        if not gaps:
            return []

        completed_queries = completed_queries or []

        scored_gaps = []
        for gap in gaps:
            score = self._calculate_priority_score(
                gap, research_topic, completed_queries
            )
            scored_gaps.append((gap, score))

        scored_gaps.sort(key=lambda x: -x[1])

        return [gap for gap, _ in scored_gaps[: self.max_gaps]]

    def _calculate_priority_score(
        self,
        gap: KnowledgeGap,
        research_topic: str,
        completed_queries: List[str],
    ) -> float:
        """Calculate priority score for a knowledge gap."""
        score = float(gap.priority)

        score += gap.impact_score * 2

        topic_keywords = set(research_topic.lower().split())
        gap_keywords = set(gap.description.lower().split())
        relevance = len(topic_keywords & gap_keywords) / max(len(topic_keywords), 1)
        score += relevance

        if gap.suggested_query.lower() in [q.lower() for q in completed_queries]:
            score *= 0.3

        category_weights = {
            "factual": 1.2,
            "contextual": 1.0,
            "analytical": 0.9,
            "temporal": 1.1,
            "methodological": 0.8,
        }
        score *= category_weights.get(gap.category.value, 1.0)

        return score
