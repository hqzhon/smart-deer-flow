# -*- coding: utf-8 -*-
"""
Researcher Progressive Enablement - Phase 3 Implementation

Provides intelligent scenario detection and progressive enablement of context isolation
for researcher nodes based on task complexity, performance metrics, and user feedback.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class IsolationStrategy(Enum):
    """Isolation strategy types"""

    DISABLED = "disabled"
    SELECTIVE = "selective"
    ADAPTIVE = "adaptive"
    ALWAYS_ON = "always_on"


@dataclass
class ScenarioContext:
    """Context information for scenario analysis"""

    task_description: str
    step_count: int
    context_size: int
    parallel_execution: bool
    has_search_results: bool
    has_complex_queries: bool
    estimated_tokens: int
    user_feedback_score: Optional[float] = None
    previous_isolation_success: Optional[bool] = None


class ResearcherProgressiveEnabler:
    """Progressive enablement system for researcher context isolation"""

    def __init__(self, config: Optional[Any] = None):
        self.scenario_history: List[Tuple[ScenarioContext, bool, float]] = (
            []
        )  # (context, enabled, performance)
        self.user_feedback: Dict[str, float] = {}  # scenario_id -> satisfaction_score
        self.performance_baseline: Optional[float] = None
        self.config = config

        # Learning parameters
        self.min_samples_for_learning = 5
        self.performance_threshold = 0.1  # 10% performance degradation threshold
        self.token_savings_threshold = (
            1000  # Minimum token savings to consider worthwhile
        )

        # Phase 2 - GFLQ Integration: Initialize reflection agent if enabled
        self.reflection_agent = None
        if config and getattr(config, "enabled", True):
            try:
                from src.utils.reflection.enhanced_reflection import (
                    EnhancedReflectionAgent,
                )

                self.reflection_agent = EnhancedReflectionAgent(config)
                logger.info(
                    "Enhanced reflection agent initialized for progressive enablement"
                )
            except ImportError as e:
                logger.warning(f"Failed to initialize reflection agent: {e}")

        logger.info("Initialized ResearcherProgressiveEnabler")

    def analyze_task_complexity(self, scenario: ScenarioContext) -> TaskComplexity:
        """Analyze task complexity based on scenario context"""
        complexity_score = 0

        # Step count factor
        if scenario.step_count >= 5:
            complexity_score += 3
        elif scenario.step_count >= 3:
            complexity_score += 2
        elif scenario.step_count >= 2:
            complexity_score += 1

        # Context size factor
        if scenario.context_size >= 10000:
            complexity_score += 3
        elif scenario.context_size >= 5000:
            complexity_score += 2
        elif scenario.context_size >= 2000:
            complexity_score += 1

        # Parallel execution factor
        if scenario.parallel_execution:
            complexity_score += 2

        # Search and query complexity
        if scenario.has_search_results:
            complexity_score += 1
        if scenario.has_complex_queries:
            complexity_score += 2

        # Token estimation factor
        if scenario.estimated_tokens >= 20000:
            complexity_score += 3
        elif scenario.estimated_tokens >= 10000:
            complexity_score += 2
        elif scenario.estimated_tokens >= 5000:
            complexity_score += 1

        # Task description analysis
        description_lower = scenario.task_description.lower()
        complex_keywords = [
            "analyze",
            "compare",
            "synthesize",
            "comprehensive",
            "detailed",
            "multiple sources",
            "cross-reference",
            "in-depth",
            "thorough",
        ]

        for keyword in complex_keywords:
            if keyword in description_lower:
                complexity_score += 1

        # Map score to complexity level
        if complexity_score >= 10:
            return TaskComplexity.VERY_COMPLEX
        elif complexity_score >= 7:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 4:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE

    def should_enable_isolation(
        self, scenario: ScenarioContext
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Determine if isolation should be enabled for the given scenario"""
        from .researcher_isolation_metrics import get_isolation_metrics

        # Get configuration from settings
        try:
            from src.config import get_settings

            config = get_settings()
        except ImportError:
            config = None
        metrics = get_isolation_metrics()

        # Get current strategy
        strategy = self._get_current_strategy(config)

        # Analyze task complexity
        complexity = self.analyze_task_complexity(scenario)

        decision_factors = {
            "strategy": strategy.value,
            "complexity": complexity.value,
            "step_count": scenario.step_count,
            "context_size": scenario.context_size,
            "parallel_execution": scenario.parallel_execution,
            "estimated_tokens": scenario.estimated_tokens,
        }

        # Strategy-based decision
        if strategy == IsolationStrategy.DISABLED:
            return False, "Isolation disabled by strategy", decision_factors

        elif strategy == IsolationStrategy.ALWAYS_ON:
            return True, "Isolation always enabled by strategy", decision_factors

        elif strategy == IsolationStrategy.SELECTIVE:
            return self._selective_enablement(
                scenario, complexity, config, decision_factors
            )

        elif strategy == IsolationStrategy.ADAPTIVE:
            return self._adaptive_enablement(
                scenario, complexity, config, metrics, decision_factors
            )

        else:
            # Default to selective
            return self._selective_enablement(
                scenario, complexity, config, decision_factors
            )

    def _get_current_strategy(self, config: Optional[Any]) -> IsolationStrategy:
        """Get current isolation strategy"""
        if not config:
            return IsolationStrategy.SELECTIVE

        if not config.enable_researcher_isolation:
            return IsolationStrategy.DISABLED

        if not config.researcher_auto_isolation:
            return IsolationStrategy.ALWAYS_ON

        # Determine strategy based on metrics and configuration
        from .researcher_isolation_metrics import get_isolation_metrics

        metrics = get_isolation_metrics()

        if metrics.isolation_sessions < self.min_samples_for_learning:
            return IsolationStrategy.SELECTIVE

        # Use adaptive strategy if we have enough data
        return IsolationStrategy.ADAPTIVE

    def _selective_enablement(
        self,
        scenario: ScenarioContext,
        complexity: TaskComplexity,
        config: Optional[Any],
        decision_factors: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Selective enablement based on predefined rules"""
        threshold = config.researcher_isolation_threshold if config else 3
        max_context = config.researcher_max_local_context if config else 5000

        # Rule-based decisions
        if scenario.step_count >= threshold:
            decision_factors["trigger"] = "step_count_threshold"
            return (
                True,
                f"Step count ({scenario.step_count}) exceeds threshold ({threshold})",
                decision_factors,
            )

        if scenario.context_size >= max_context:
            decision_factors["trigger"] = "context_size_threshold"
            return (
                True,
                f"Context size ({scenario.context_size}) exceeds threshold ({max_context})",
                decision_factors,
            )

        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            decision_factors["trigger"] = "high_complexity"
            return True, f"High task complexity ({complexity.value})", decision_factors

        if scenario.parallel_execution and scenario.step_count >= 2:
            decision_factors["trigger"] = "parallel_execution"
            return True, "Parallel execution with multiple steps", decision_factors

        if scenario.estimated_tokens >= 15000:
            decision_factors["trigger"] = "high_token_estimate"
            return (
                True,
                f"High token estimate ({scenario.estimated_tokens})",
                decision_factors,
            )

        decision_factors["trigger"] = "no_trigger_met"
        return False, "No selective enablement criteria met", decision_factors

    def _adaptive_enablement(
        self,
        scenario: ScenarioContext,
        complexity: TaskComplexity,
        config: Optional[Any],
        metrics: Any,
        decision_factors: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Adaptive enablement based on historical performance and learning"""
        # Phase 2 - GFLQ Integration: Add reflection-driven decision making
        if self.reflection_agent:
            try:
                # Create reflection context for analysis
                from src.utils.reflection.enhanced_reflection import ReflectionContext
                import asyncio

                reflection_context = ReflectionContext(
                    research_topic=scenario.task_description,
                    completed_steps=[],
                    execution_results=[],
                    total_steps=scenario.step_count,
                    current_step_index=0,
                    locale="en-US",
                )

                # Use analyze_knowledge_gaps for full reflection analysis
                reflection_result = asyncio.run(
                    self.reflection_agent.analyze_knowledge_gaps(reflection_context)
                )
                decision_factors["reflection_insights"] = {
                    "is_sufficient": reflection_result.is_sufficient,
                    "knowledge_gaps_count": len(reflection_result.knowledge_gaps),
                    "follow_up_queries_count": len(reflection_result.follow_up_queries),
                    "confidence_score": reflection_result.confidence_score or 0.0,
                }

                # If reflection indicates insufficient information, enable deep research
                if not reflection_result.is_sufficient:
                    decision_factors["trigger"] = "reflection_insufficient_knowledge"
                    decision_factors["suggested_queries"] = (
                        reflection_result.follow_up_queries[:3]
                    )  # Limit to top 3
                    return (
                        True,
                        "Reflection analysis indicates knowledge gaps requiring deep research",
                        decision_factors,
                    )

            except Exception as e:
                logger.warning(f"Reflection analysis failed: {e}")

        # Get historical performance for similar scenarios
        similar_scenarios = self._find_similar_scenarios(scenario)

        decision_factors["similar_scenarios_count"] = len(similar_scenarios)

        if similar_scenarios:
            # Calculate average performance and success rate for similar scenarios
            isolation_enabled_scenarios = [
                s for s in similar_scenarios if s[1]
            ]  # enabled=True
            isolation_disabled_scenarios = [
                s for s in similar_scenarios if not s[1]
            ]  # enabled=False

            if isolation_enabled_scenarios and isolation_disabled_scenarios:
                avg_perf_enabled = sum(s[2] for s in isolation_enabled_scenarios) / len(
                    isolation_enabled_scenarios
                )
                avg_perf_disabled = sum(
                    s[2] for s in isolation_disabled_scenarios
                ) / len(isolation_disabled_scenarios)

                decision_factors["avg_perf_enabled"] = avg_perf_enabled
                decision_factors["avg_perf_disabled"] = avg_perf_disabled

                # Enable if isolation shows better performance
                if avg_perf_enabled < avg_perf_disabled * (
                    1 + self.performance_threshold
                ):
                    decision_factors["trigger"] = "adaptive_performance_better"
                    return (
                        True,
                        "Historical data shows isolation improves performance",
                        decision_factors,
                    )

        # Check current metrics
        health = metrics.get_isolation_health()

        if (
            health["estimated_token_savings"]
            / max(health["total_isolation_sessions"], 1)
            >= self.token_savings_threshold
        ):
            # Good token savings, enable for complex scenarios
            if complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
                decision_factors["trigger"] = "adaptive_good_savings"
                return (
                    True,
                    "Good historical token savings for complex tasks",
                    decision_factors,
                )

        # Check recent success rate
        if health["recent_success_rate"] >= 0.9:
            # High success rate, be more aggressive
            if scenario.step_count >= 2 or scenario.context_size >= 3000:
                decision_factors["trigger"] = "adaptive_high_success_rate"
                return (
                    True,
                    "High recent success rate allows aggressive enablement",
                    decision_factors,
                )

        # Fall back to selective enablement
        return self._selective_enablement(
            scenario, complexity, config, decision_factors
        )

    def _find_similar_scenarios(
        self, scenario: ScenarioContext
    ) -> List[Tuple[ScenarioContext, bool, float]]:
        """Find historically similar scenarios"""
        similar = []

        for hist_scenario, enabled, performance in self.scenario_history:
            similarity_score = self._calculate_similarity(scenario, hist_scenario)

            # Consider scenarios with similarity > 0.7 as similar
            if similarity_score > 0.7:
                similar.append((hist_scenario, enabled, performance))

        return similar

    def _calculate_similarity(
        self, scenario1: ScenarioContext, scenario2: ScenarioContext
    ) -> float:
        """Calculate similarity between two scenarios"""
        similarity = 0.0
        factors = 0

        # Step count similarity
        step_diff = abs(scenario1.step_count - scenario2.step_count)
        step_similarity = max(
            0, 1 - step_diff / 5
        )  # Normalize by max expected difference
        similarity += step_similarity
        factors += 1

        # Context size similarity
        context_diff = abs(scenario1.context_size - scenario2.context_size)
        context_similarity = max(
            0, 1 - context_diff / 10000
        )  # Normalize by max expected difference
        similarity += context_similarity
        factors += 1

        # Boolean features
        if scenario1.parallel_execution == scenario2.parallel_execution:
            similarity += 1
        factors += 1

        if scenario1.has_search_results == scenario2.has_search_results:
            similarity += 1
        factors += 1

        if scenario1.has_complex_queries == scenario2.has_complex_queries:
            similarity += 1
        factors += 1

        # Token estimation similarity
        token_diff = abs(scenario1.estimated_tokens - scenario2.estimated_tokens)
        token_similarity = max(
            0, 1 - token_diff / 20000
        )  # Normalize by max expected difference
        similarity += token_similarity
        factors += 1

        return similarity / factors if factors > 0 else 0.0

    def record_scenario_outcome(
        self,
        scenario: ScenarioContext,
        isolation_enabled: bool,
        execution_time: float,
        token_savings: int = 0,
        user_satisfaction: Optional[float] = None,
    ):
        """Record the outcome of a scenario for learning"""
        # Calculate performance score (lower is better)
        performance_score = execution_time

        # Adjust for token savings (positive impact)
        if token_savings > 0:
            performance_score *= 1 - min(
                0.3, token_savings / 10000
            )  # Up to 30% improvement

        # Record scenario
        self.scenario_history.append((scenario, isolation_enabled, performance_score))

        # Record user feedback if provided
        if user_satisfaction is not None:
            scenario_id = self._generate_scenario_id(scenario)
            self.user_feedback[scenario_id] = user_satisfaction

        # Update baseline if this is a non-isolation scenario
        if not isolation_enabled:
            if self.performance_baseline is None:
                self.performance_baseline = performance_score
            else:
                # Running average
                self.performance_baseline = (
                    self.performance_baseline * 0.8 + performance_score * 0.2
                )

        # Limit history size
        if len(self.scenario_history) > 1000:
            self.scenario_history = self.scenario_history[-500:]  # Keep recent 500

        logger.debug(
            f"Recorded scenario outcome: isolation={isolation_enabled}, "
            f"performance={performance_score:.2f}, token_savings={token_savings}"
        )

    def _generate_scenario_id(self, scenario: ScenarioContext) -> str:
        """Generate a unique ID for a scenario"""
        import hashlib

        scenario_str = f"{scenario.task_description[:100]}_{scenario.step_count}_{scenario.context_size}_{scenario.parallel_execution}"
        return hashlib.md5(scenario_str.encode()).hexdigest()[:8]

    def get_enablement_statistics(self) -> Dict[str, Any]:
        """Get statistics about enablement decisions"""
        if not self.scenario_history:
            return {"message": "No scenario history available"}

        enabled_scenarios = [s for s in self.scenario_history if s[1]]
        disabled_scenarios = [s for s in self.scenario_history if not s[1]]

        stats = {
            "total_scenarios": len(self.scenario_history),
            "isolation_enabled_count": len(enabled_scenarios),
            "isolation_disabled_count": len(disabled_scenarios),
            "enablement_rate": len(enabled_scenarios) / len(self.scenario_history),
        }

        if enabled_scenarios:
            stats["avg_performance_enabled"] = sum(
                s[2] for s in enabled_scenarios
            ) / len(enabled_scenarios)

        if disabled_scenarios:
            stats["avg_performance_disabled"] = sum(
                s[2] for s in disabled_scenarios
            ) / len(disabled_scenarios)

        if enabled_scenarios and disabled_scenarios:
            stats["performance_improvement"] = (
                stats["avg_performance_disabled"] - stats["avg_performance_enabled"]
            ) / stats["avg_performance_disabled"]

        stats["user_feedback_count"] = len(self.user_feedback)
        if self.user_feedback:
            stats["avg_user_satisfaction"] = sum(self.user_feedback.values()) / len(
                self.user_feedback
            )

        return stats

    def optimize_thresholds(self) -> Dict[str, Any]:
        """Optimize enablement thresholds based on historical data"""
        if len(self.scenario_history) < self.min_samples_for_learning:
            return {"message": "Insufficient data for threshold optimization"}

        # Analyze optimal thresholds
        step_count_analysis = self._analyze_threshold_performance("step_count")
        context_size_analysis = self._analyze_threshold_performance("context_size")

        recommendations = {
            "current_data_points": len(self.scenario_history),
            "step_count_analysis": step_count_analysis,
            "context_size_analysis": context_size_analysis,
            "recommendations": [],
        }

        # Generate recommendations
        if step_count_analysis.get("optimal_threshold"):
            recommendations["recommendations"].append(
                f"Consider setting step count threshold to {step_count_analysis['optimal_threshold']}"
            )

        if context_size_analysis.get("optimal_threshold"):
            recommendations["recommendations"].append(
                f"Consider setting context size threshold to {context_size_analysis['optimal_threshold']}"
            )

        return recommendations

    def _analyze_threshold_performance(self, attribute: str) -> Dict[str, Any]:
        """Analyze performance for different threshold values of a given attribute"""
        # Group scenarios by attribute value ranges
        ranges = {}

        for scenario, enabled, performance in self.scenario_history:
            value = getattr(scenario, attribute)

            # Create range buckets
            if attribute == "step_count":
                bucket = (value // 2) * 2  # Group by 2s
            elif attribute == "context_size":
                bucket = (value // 1000) * 1000  # Group by 1000s
            else:
                bucket = value

            if bucket not in ranges:
                ranges[bucket] = {"enabled": [], "disabled": []}

            if enabled:
                ranges[bucket]["enabled"].append(performance)
            else:
                ranges[bucket]["disabled"].append(performance)

        # Find optimal threshold
        best_threshold = None
        best_improvement = 0

        for threshold, data in ranges.items():
            if data["enabled"] and data["disabled"]:
                avg_enabled = sum(data["enabled"]) / len(data["enabled"])
                avg_disabled = sum(data["disabled"]) / len(data["disabled"])
                improvement = (avg_disabled - avg_enabled) / avg_disabled

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_threshold = threshold

        return {
            "ranges_analyzed": len(ranges),
            "optimal_threshold": best_threshold,
            "best_improvement": best_improvement,
            "threshold_data": ranges,
        }


# Global progressive enabler instance
_global_enabler: Optional[ResearcherProgressiveEnabler] = None


def get_progressive_enabler() -> ResearcherProgressiveEnabler:
    """Get global progressive enabler instance"""
    global _global_enabler
    if _global_enabler is None:
        _global_enabler = ResearcherProgressiveEnabler()
    return _global_enabler


def reset_progressive_enabler():
    """Reset global progressive enabler (for testing)"""
    global _global_enabler
    _global_enabler = None
