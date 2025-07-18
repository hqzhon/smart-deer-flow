# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Reflection tools and utilities for enhanced research analysis.
Provides helper functions for reflection result processing and analysis.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from src.utils.reflection.enhanced_reflection import ReflectionResult
from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ReflectionSession:
    """
    Represents a reflection analysis session.
    """

    session_id: str
    research_topic: str
    step_count: int
    timestamp: datetime
    reflection_result: Optional[ReflectionResult] = None
    context_size: int = 0
    complexity_score: float = 0.0
    isolation_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary format."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        if self.reflection_result:
            result["reflection_result"] = self.reflection_result.model_dump()
        return result


class ReflectionAnalyzer:
    """
    Utility class for analyzing reflection results and trends.
    """

    def __init__(self, config=None):
        # Get configuration from settings
        try:
            settings = get_settings()
            self.config = settings
        except Exception:
            self.config = None
        self.session_history: List[ReflectionSession] = []

    def analyze_sufficiency_trend(
        self, sessions: List[ReflectionSession]
    ) -> Dict[str, Any]:
        """
        Analyze the trend of research sufficiency over multiple sessions.

        Args:
            sessions: List of reflection sessions to analyze

        Returns:
            Dict containing trend analysis results
        """
        if not sessions:
            return {
                "trend": "no_data",
                "confidence": 0.0,
                "recommendation": "insufficient_data",
            }

        # Extract confidence scores
        scores = []
        for session in sessions:
            if (
                session.reflection_result
                and session.reflection_result.confidence_score is not None
            ):
                scores.append(session.reflection_result.confidence_score)

        if not scores:
            return {
                "trend": "no_scores",
                "confidence": 0.0,
                "recommendation": "enable_scoring",
            }

        # Calculate trend
        if len(scores) < 2:
            trend = "insufficient_data"
        elif len(scores) == 2:
            # Simple comparison for two scores
            if scores[1] > scores[0] + 0.1:
                trend = "improving"
            elif scores[1] < scores[0] - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            # For more than 2 scores, compare recent vs earlier
            mid_point = len(scores) // 2
            recent_avg = sum(scores[mid_point:]) / len(scores[mid_point:])
            earlier_avg = sum(scores[:mid_point]) / len(scores[:mid_point])

            if recent_avg > earlier_avg + 0.1:
                trend = "improving"
            elif recent_avg < earlier_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"

        # Generate recommendation
        latest_score = scores[-1]
        # Get reflection confidence threshold from config or use default
        threshold = 0.7  # default value
        if self.config:
            try:
                reflection_config = getattr(self.config, "reflection", None)
                if reflection_config:
                    threshold = getattr(reflection_config, "confidence_threshold", 0.7)
            except Exception:
                pass

        if latest_score >= threshold:
            if trend == "improving":
                recommendation = "continue_current_approach"
            else:
                recommendation = "maintain_quality"
        else:
            if trend == "declining":
                recommendation = "investigate_quality_issues"
            else:
                recommendation = "increase_research_depth"

        return {
            "trend": trend,
            "confidence": latest_score,
            "average_confidence": sum(scores) / len(scores),
            "recommendation": recommendation,
            "score_history": scores,
        }

    def identify_recurring_gaps(self, sessions: List[ReflectionSession]) -> List[str]:
        """
        Identify knowledge gaps that appear repeatedly across sessions.

        Args:
            sessions: List of reflection sessions to analyze

        Returns:
            List of recurring knowledge gaps
        """
        gap_frequency = {}

        for session in sessions:
            if session.reflection_result and session.reflection_result.knowledge_gaps:
                for gap in session.reflection_result.knowledge_gaps:
                    gap_lower = gap.lower().strip()
                    gap_frequency[gap_lower] = gap_frequency.get(gap_lower, 0) + 1

        # Return gaps that appear in more than one session
        recurring_gaps = [gap for gap, freq in gap_frequency.items() if freq > 1]

        return sorted(recurring_gaps, key=lambda x: gap_frequency[x], reverse=True)

    def generate_research_recommendations(
        self, current_session: ReflectionSession, history: List[ReflectionSession]
    ) -> List[str]:
        """
        Generate actionable research recommendations based on reflection analysis.

        Args:
            current_session: Current reflection session
            history: Historical reflection sessions

        Returns:
            List of research recommendations
        """
        recommendations = []

        if not current_session.reflection_result:
            return ["Enable reflection analysis to get recommendations"]

        result = current_session.reflection_result

        # Sufficiency-based recommendations
        if not result.is_sufficient:
            if result.confidence_score < 0.3:
                recommendations.append(
                    "Research appears incomplete. Consider expanding scope and depth."
                )
            elif result.confidence_score < 0.6:
                recommendations.append(
                    "Research shows moderate progress. Focus on identified knowledge gaps."
                )
            else:
                recommendations.append(
                    "Research is nearly complete. Address remaining specific gaps."
                )

        # Gap-based recommendations
        if result.knowledge_gaps:
            recurring_gaps = self.identify_recurring_gaps(history + [current_session])
            for gap in result.knowledge_gaps[:3]:  # Top 3 gaps
                if gap.lower() in [rg.lower() for rg in recurring_gaps]:
                    recommendations.append(f"Priority: Address recurring gap - {gap}")
                else:
                    recommendations.append(f"Investigate: {gap}")

        # Follow-up query recommendations
        if result.follow_up_queries:
            recommendations.append(
                f"Execute {len(result.follow_up_queries)} follow-up queries for comprehensive coverage"
            )

        # Trend-based recommendations
        if len(history) >= 2:
            trend_analysis = self.analyze_sufficiency_trend(history + [current_session])
            if trend_analysis["trend"] == "declining":
                recommendations.append(
                    "Quality trend declining. Review research methodology and sources."
                )
            elif trend_analysis["trend"] == "improving":
                recommendations.append(
                    "Quality trend positive. Continue current research approach."
                )

        return recommendations[:5]  # Limit to top 5 recommendations


class ReflectionTools:
    """
    Main utility class providing reflection tools and analysis capabilities.
    """

    def __init__(self, config=None):
        """Initialize reflection tools.

        Args:
            config: Configuration instance (optional, will use get_settings() if not provided)
        """
        try:
            self.config = get_settings() if config is None else config
        except Exception:
            self.config = None
        self.analyzer = ReflectionAnalyzer(self.config)
        self.metrics = ReflectionMetrics()
        self.session_history: List[ReflectionSession] = []

    def create_session(
        self, session_id: str, research_topic: str, step_count: int
    ) -> ReflectionSession:
        """Create a new reflection session.

        Args:
            session_id: Unique session identifier
            research_topic: Topic being researched
            step_count: Number of research steps

        Returns:
            New ReflectionSession instance
        """
        session = ReflectionSession(
            session_id=session_id,
            research_topic=research_topic,
            step_count=step_count,
            timestamp=datetime.now(),
        )
        self.session_history.append(session)
        return session

    def analyze_session(self, session: ReflectionSession) -> Dict[str, Any]:
        """Analyze a reflection session.

        Args:
            session: Session to analyze

        Returns:
            Analysis results
        """
        return {
            "sufficiency_trend": self.analyzer.analyze_sufficiency_trend([session]),
            "recurring_gaps": self.analyzer.identify_recurring_gaps([session]),
            "recommendations": self.analyzer.generate_research_recommendations(
                session, self.session_history[:-1]
            ),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get reflection metrics.

        Returns:
            Metrics dictionary
        """
        return self.metrics.get_summary_metrics()


class ReflectionMetrics:
    """
    Utility class for collecting and analyzing reflection performance metrics.
    """

    def __init__(self):
        self.metrics_data: Dict[str, List[float]] = {
            "confidence_scores": [],
            "gap_counts": [],
            "query_counts": [],
            "session_durations": [],
            "complexity_scores": [],
        }

    def record_session_metrics(self, session: ReflectionSession, duration_ms: float):
        """
        Record metrics from a reflection session.

        Args:
            session: Reflection session to record
            duration_ms: Session duration in milliseconds
        """
        if session.reflection_result:
            result = session.reflection_result

            if result.confidence_score is not None:
                self.metrics_data["confidence_scores"].append(result.confidence_score)

            self.metrics_data["gap_counts"].append(len(result.knowledge_gaps or []))
            self.metrics_data["query_counts"].append(
                len(result.follow_up_queries or [])
            )

        self.metrics_data["session_durations"].append(duration_ms)
        self.metrics_data["complexity_scores"].append(session.complexity_score)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of reflection performance metrics.

        Returns:
            Dict containing performance summary
        """
        summary = {}

        for metric_name, values in self.metrics_data.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0,
                }
            else:
                summary[metric_name] = {
                    "count": 0,
                    "average": 0,
                    "min": 0,
                    "max": 0,
                    "latest": 0,
                }

        return summary

    def export_metrics(self) -> str:
        """
        Export metrics data as JSON string.

        Returns:
            JSON string containing all metrics data
        """
        export_data = {
            "metrics": self.metrics_data,
            "summary": self.get_performance_summary(),
            "export_timestamp": datetime.now().isoformat(),
        }

        return json.dumps(export_data, indent=2)


def parse_reflection_result(raw_output: str) -> Optional[ReflectionResult]:
    """
    Parse raw model output into a ReflectionResult object.

    Args:
        raw_output: Raw output from reflection model

    Returns:
        Parsed ReflectionResult or None if parsing fails
    """
    logger.info(
        f"Parsing reflection result from raw output (length: {len(raw_output)})"
    )
    logger.debug(f"Raw reflection output: {raw_output}")

    try:
        # Try to extract JSON from the output
        if "{" in raw_output and "}" in raw_output:
            start_idx = raw_output.find("{")
            end_idx = raw_output.rfind("}") + 1
            json_str = raw_output[start_idx:end_idx]

            logger.debug(f"Extracted JSON string: {json_str}")

            data = json.loads(json_str)
            logger.info("Successfully parsed JSON data from reflection output")
            logger.debug(f"Parsed reflection data: {data}")

            # Create ReflectionResult with validation
            result = ReflectionResult(
                is_sufficient=data.get("is_sufficient", False),
                confidence_score=data.get("confidence_score"),
                knowledge_gaps=data.get("knowledge_gaps", []),
                follow_up_queries=data.get("follow_up_queries", []),
                quality_assessment=data.get("quality_assessment", {}),
                recommendations=data.get("recommendations", []),
                priority_areas=data.get("priority_areas", []),
            )

            # Log parsed result details
            logger.info(
                f"Reflection result parsed: sufficient={result.is_sufficient}, confidence={result.confidence_score}, gaps={len(result.knowledge_gaps)}, queries={len(result.follow_up_queries)}"
            )

            return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse reflection result: {e}")
        logger.debug(f"Raw output: {raw_output}")

    logger.warning("Could not parse reflection result, returning None")
    return None


def calculate_research_complexity(
    research_topic: str, step_count: int, context_size: int, findings_count: int
) -> float:
    """
    Calculate research complexity score based on various factors.

    Args:
        research_topic: The research topic
        step_count: Number of research steps
        context_size: Size of research context
        findings_count: Number of findings collected

    Returns:
        Complexity score between 0.0 and 1.0
    """
    # Base complexity from topic length and keywords
    topic_complexity = min(len(research_topic.split()) / 20.0, 0.3)

    # Step complexity (more steps = higher complexity)
    step_complexity = min(step_count / 10.0, 0.3)

    # Context complexity (larger context = higher complexity)
    context_complexity = min(context_size / 10000.0, 0.2)

    # Findings complexity (more findings = higher complexity)
    findings_complexity = min(findings_count / 20.0, 0.2)

    total_complexity = (
        topic_complexity + step_complexity + context_complexity + findings_complexity
    )

    return min(total_complexity, 1.0)
