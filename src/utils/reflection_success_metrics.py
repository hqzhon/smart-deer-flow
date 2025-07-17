# -*- coding: utf-8 -*-
"""
Reflection Success Metrics and Validation

This module implements success metrics and acceptance criteria for Phase 5 reflection integration,
including automated validation, reporting, and continuous monitoring.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricStatus(Enum):
    """Status of success metrics."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILING = "failing"


class ValidationResult(Enum):
    """Validation result status."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class SuccessThresholds:
    """Thresholds for success metrics."""

    # Technical Performance Thresholds
    max_error_rate: float = 0.01  # 1%
    max_response_time_p95: float = 2.0  # 2 seconds
    max_memory_overhead: float = 0.15  # 15%
    max_cpu_overhead: float = 0.10  # 10%
    min_cache_hit_rate: float = 0.80  # 80%

    # Quality Thresholds
    min_reflection_accuracy: float = 0.85  # 85%
    min_knowledge_gap_detection: float = 0.90  # 90%
    min_query_improvement_rate: float = 0.75  # 75%
    max_false_positive_rate: float = 0.05  # 5%

    # User Experience Thresholds
    min_user_satisfaction: float = 0.80  # 80%
    min_task_completion_rate: float = 0.95  # 95%
    max_user_reported_issues: int = 5  # per 1000 users
    min_feature_adoption_rate: float = 0.60  # 60%

    # Business Impact Thresholds
    min_research_efficiency_gain: float = 0.20  # 20%
    min_query_success_improvement: float = 0.15  # 15%
    max_support_ticket_increase: float = 0.05  # 5%
    min_user_retention_rate: float = 0.95  # 95%


@dataclass
class MetricMeasurement:
    """Individual metric measurement."""

    name: str
    value: float
    threshold: float
    status: MetricStatus
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def is_passing(self) -> bool:
        """Check if metric is passing."""
        return self.status in [
            MetricStatus.EXCELLENT,
            MetricStatus.GOOD,
            MetricStatus.ACCEPTABLE,
        ]


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    validation_id: str
    timestamp: datetime
    overall_result: ValidationResult
    technical_metrics: List[MetricMeasurement]
    quality_metrics: List[MetricMeasurement]
    user_experience_metrics: List[MetricMeasurement]
    business_metrics: List[MetricMeasurement]
    summary: Dict[str, Any]
    recommendations: List[str]
    next_validation: Optional[datetime] = None


class ReflectionSuccessValidator:
    """Validates success metrics for reflection system."""

    def __init__(self, thresholds: Optional[SuccessThresholds] = None):
        self.thresholds = thresholds or SuccessThresholds()
        self.validation_history: List[ValidationReport] = []

        logger.info("Reflection Success Validator initialized")

    def validate_technical_performance(
        self, metrics: Dict[str, Any]
    ) -> List[MetricMeasurement]:
        """Validate technical performance metrics."""
        measurements = []

        # Error Rate
        error_rate = metrics.get("error_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="error_rate",
                value=error_rate,
                threshold=self.thresholds.max_error_rate,
                status=self._evaluate_error_rate(error_rate),
                details={
                    "description": "System error rate during reflection operations"
                },
            )
        )

        # Response Time P95
        response_time_p95 = metrics.get("response_time_p95", 0.0)
        measurements.append(
            MetricMeasurement(
                name="response_time_p95",
                value=response_time_p95,
                threshold=self.thresholds.max_response_time_p95,
                status=self._evaluate_response_time(response_time_p95),
                details={
                    "description": "95th percentile response time for reflection operations"
                },
            )
        )

        # Memory Overhead
        memory_overhead = metrics.get("memory_overhead", 0.0)
        measurements.append(
            MetricMeasurement(
                name="memory_overhead",
                value=memory_overhead,
                threshold=self.thresholds.max_memory_overhead,
                status=self._evaluate_overhead(
                    memory_overhead, self.thresholds.max_memory_overhead
                ),
                details={
                    "description": "Additional memory usage due to reflection system"
                },
            )
        )

        # CPU Overhead
        cpu_overhead = metrics.get("cpu_overhead", 0.0)
        measurements.append(
            MetricMeasurement(
                name="cpu_overhead",
                value=cpu_overhead,
                threshold=self.thresholds.max_cpu_overhead,
                status=self._evaluate_overhead(
                    cpu_overhead, self.thresholds.max_cpu_overhead
                ),
                details={
                    "description": "Additional CPU usage due to reflection system"
                },
            )
        )

        # Cache Hit Rate
        cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="cache_hit_rate",
                value=cache_hit_rate,
                threshold=self.thresholds.min_cache_hit_rate,
                status=self._evaluate_cache_hit_rate(cache_hit_rate),
                details={"description": "Reflection result cache hit rate"},
            )
        )

        return measurements

    def validate_quality_metrics(
        self, metrics: Dict[str, Any]
    ) -> List[MetricMeasurement]:
        """Validate quality metrics."""
        measurements = []

        # Reflection Accuracy
        reflection_accuracy = metrics.get("reflection_accuracy", 0.0)
        measurements.append(
            MetricMeasurement(
                name="reflection_accuracy",
                value=reflection_accuracy,
                threshold=self.thresholds.min_reflection_accuracy,
                status=self._evaluate_accuracy(
                    reflection_accuracy, self.thresholds.min_reflection_accuracy
                ),
                details={
                    "description": "Accuracy of reflection analysis and recommendations"
                },
            )
        )

        # Knowledge Gap Detection
        gap_detection_rate = metrics.get("knowledge_gap_detection_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="knowledge_gap_detection_rate",
                value=gap_detection_rate,
                threshold=self.thresholds.min_knowledge_gap_detection,
                status=self._evaluate_accuracy(
                    gap_detection_rate, self.thresholds.min_knowledge_gap_detection
                ),
                details={
                    "description": "Rate of successful knowledge gap identification"
                },
            )
        )

        # Query Improvement Rate
        query_improvement_rate = metrics.get("query_improvement_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="query_improvement_rate",
                value=query_improvement_rate,
                threshold=self.thresholds.min_query_improvement_rate,
                status=self._evaluate_accuracy(
                    query_improvement_rate, self.thresholds.min_query_improvement_rate
                ),
                details={
                    "description": "Rate of successful query improvements through reflection"
                },
            )
        )

        # False Positive Rate
        false_positive_rate = metrics.get("false_positive_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="false_positive_rate",
                value=false_positive_rate,
                threshold=self.thresholds.max_false_positive_rate,
                status=self._evaluate_false_positive_rate(false_positive_rate),
                details={"description": "Rate of incorrect reflection recommendations"},
            )
        )

        return measurements

    def validate_user_experience(
        self, metrics: Dict[str, Any]
    ) -> List[MetricMeasurement]:
        """Validate user experience metrics."""
        measurements = []

        # User Satisfaction
        user_satisfaction = metrics.get("user_satisfaction", 0.0)
        measurements.append(
            MetricMeasurement(
                name="user_satisfaction",
                value=user_satisfaction,
                threshold=self.thresholds.min_user_satisfaction,
                status=self._evaluate_satisfaction(user_satisfaction),
                details={
                    "description": "Overall user satisfaction with reflection features"
                },
            )
        )

        # Task Completion Rate
        task_completion_rate = metrics.get("task_completion_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="task_completion_rate",
                value=task_completion_rate,
                threshold=self.thresholds.min_task_completion_rate,
                status=self._evaluate_completion_rate(task_completion_rate),
                details={
                    "description": "Rate of successful task completion with reflection assistance"
                },
            )
        )

        # User Reported Issues
        user_reported_issues = metrics.get("user_reported_issues_per_1000", 0)
        measurements.append(
            MetricMeasurement(
                name="user_reported_issues_per_1000",
                value=user_reported_issues,
                threshold=self.thresholds.max_user_reported_issues,
                status=self._evaluate_user_issues(user_reported_issues),
                details={
                    "description": "Number of user-reported issues per 1000 users"
                },
            )
        )

        # Feature Adoption Rate
        feature_adoption_rate = metrics.get("feature_adoption_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="feature_adoption_rate",
                value=feature_adoption_rate,
                threshold=self.thresholds.min_feature_adoption_rate,
                status=self._evaluate_adoption_rate(feature_adoption_rate),
                details={
                    "description": "Rate of reflection feature adoption among users"
                },
            )
        )

        return measurements

    def validate_business_impact(
        self, metrics: Dict[str, Any]
    ) -> List[MetricMeasurement]:
        """Validate business impact metrics."""
        measurements = []

        # Research Efficiency Gain
        efficiency_gain = metrics.get("research_efficiency_gain", 0.0)
        measurements.append(
            MetricMeasurement(
                name="research_efficiency_gain",
                value=efficiency_gain,
                threshold=self.thresholds.min_research_efficiency_gain,
                status=self._evaluate_efficiency_gain(efficiency_gain),
                details={
                    "description": "Improvement in research efficiency due to reflection"
                },
            )
        )

        # Query Success Improvement
        query_success_improvement = metrics.get("query_success_improvement", 0.0)
        measurements.append(
            MetricMeasurement(
                name="query_success_improvement",
                value=query_success_improvement,
                threshold=self.thresholds.min_query_success_improvement,
                status=self._evaluate_success_improvement(query_success_improvement),
                details={"description": "Improvement in query success rate"},
            )
        )

        # Support Ticket Increase
        support_ticket_increase = metrics.get("support_ticket_increase", 0.0)
        measurements.append(
            MetricMeasurement(
                name="support_ticket_increase",
                value=support_ticket_increase,
                threshold=self.thresholds.max_support_ticket_increase,
                status=self._evaluate_support_tickets(support_ticket_increase),
                details={"description": "Change in support ticket volume"},
            )
        )

        # User Retention Rate
        user_retention_rate = metrics.get("user_retention_rate", 0.0)
        measurements.append(
            MetricMeasurement(
                name="user_retention_rate",
                value=user_retention_rate,
                threshold=self.thresholds.min_user_retention_rate,
                status=self._evaluate_retention_rate(user_retention_rate),
                details={"description": "User retention rate with reflection features"},
            )
        )

        return measurements

    def _evaluate_error_rate(self, error_rate: float) -> MetricStatus:
        """Evaluate error rate status."""
        if error_rate <= self.thresholds.max_error_rate * 0.5:
            return MetricStatus.EXCELLENT
        elif error_rate <= self.thresholds.max_error_rate * 0.75:
            return MetricStatus.GOOD
        elif error_rate <= self.thresholds.max_error_rate:
            return MetricStatus.ACCEPTABLE
        elif error_rate <= self.thresholds.max_error_rate * 1.5:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_response_time(self, response_time: float) -> MetricStatus:
        """Evaluate response time status."""
        if response_time <= self.thresholds.max_response_time_p95 * 0.5:
            return MetricStatus.EXCELLENT
        elif response_time <= self.thresholds.max_response_time_p95 * 0.75:
            return MetricStatus.GOOD
        elif response_time <= self.thresholds.max_response_time_p95:
            return MetricStatus.ACCEPTABLE
        elif response_time <= self.thresholds.max_response_time_p95 * 1.5:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_overhead(self, overhead: float, threshold: float) -> MetricStatus:
        """Evaluate overhead status."""
        if overhead <= threshold * 0.5:
            return MetricStatus.EXCELLENT
        elif overhead <= threshold * 0.75:
            return MetricStatus.GOOD
        elif overhead <= threshold:
            return MetricStatus.ACCEPTABLE
        elif overhead <= threshold * 1.5:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_cache_hit_rate(self, hit_rate: float) -> MetricStatus:
        """Evaluate cache hit rate status."""
        if hit_rate >= self.thresholds.min_cache_hit_rate * 1.1:
            return MetricStatus.EXCELLENT
        elif hit_rate >= self.thresholds.min_cache_hit_rate * 1.05:
            return MetricStatus.GOOD
        elif hit_rate >= self.thresholds.min_cache_hit_rate:
            return MetricStatus.ACCEPTABLE
        elif hit_rate >= self.thresholds.min_cache_hit_rate * 0.9:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_accuracy(self, accuracy: float, threshold: float) -> MetricStatus:
        """Evaluate accuracy-based metrics."""
        if accuracy >= threshold * 1.1:
            return MetricStatus.EXCELLENT
        elif accuracy >= threshold * 1.05:
            return MetricStatus.GOOD
        elif accuracy >= threshold:
            return MetricStatus.ACCEPTABLE
        elif accuracy >= threshold * 0.9:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_false_positive_rate(self, fp_rate: float) -> MetricStatus:
        """Evaluate false positive rate."""
        if fp_rate <= self.thresholds.max_false_positive_rate * 0.5:
            return MetricStatus.EXCELLENT
        elif fp_rate <= self.thresholds.max_false_positive_rate * 0.75:
            return MetricStatus.GOOD
        elif fp_rate <= self.thresholds.max_false_positive_rate:
            return MetricStatus.ACCEPTABLE
        elif fp_rate <= self.thresholds.max_false_positive_rate * 1.5:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_satisfaction(self, satisfaction: float) -> MetricStatus:
        """Evaluate user satisfaction."""
        if satisfaction >= 0.95:
            return MetricStatus.EXCELLENT
        elif satisfaction >= 0.90:
            return MetricStatus.GOOD
        elif satisfaction >= self.thresholds.min_user_satisfaction:
            return MetricStatus.ACCEPTABLE
        elif satisfaction >= self.thresholds.min_user_satisfaction * 0.9:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_completion_rate(self, completion_rate: float) -> MetricStatus:
        """Evaluate task completion rate."""
        if completion_rate >= 0.98:
            return MetricStatus.EXCELLENT
        elif completion_rate >= 0.97:
            return MetricStatus.GOOD
        elif completion_rate >= self.thresholds.min_task_completion_rate:
            return MetricStatus.ACCEPTABLE
        elif completion_rate >= self.thresholds.min_task_completion_rate * 0.95:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_user_issues(self, issues_per_1000: int) -> MetricStatus:
        """Evaluate user reported issues."""
        if issues_per_1000 <= self.thresholds.max_user_reported_issues * 0.5:
            return MetricStatus.EXCELLENT
        elif issues_per_1000 <= self.thresholds.max_user_reported_issues * 0.75:
            return MetricStatus.GOOD
        elif issues_per_1000 <= self.thresholds.max_user_reported_issues:
            return MetricStatus.ACCEPTABLE
        elif issues_per_1000 <= self.thresholds.max_user_reported_issues * 1.5:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_adoption_rate(self, adoption_rate: float) -> MetricStatus:
        """Evaluate feature adoption rate."""
        if adoption_rate >= 0.85:
            return MetricStatus.EXCELLENT
        elif adoption_rate >= 0.75:
            return MetricStatus.GOOD
        elif adoption_rate >= self.thresholds.min_feature_adoption_rate:
            return MetricStatus.ACCEPTABLE
        elif adoption_rate >= self.thresholds.min_feature_adoption_rate * 0.8:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_efficiency_gain(self, efficiency_gain: float) -> MetricStatus:
        """Evaluate research efficiency gain."""
        if efficiency_gain >= 0.40:
            return MetricStatus.EXCELLENT
        elif efficiency_gain >= 0.30:
            return MetricStatus.GOOD
        elif efficiency_gain >= self.thresholds.min_research_efficiency_gain:
            return MetricStatus.ACCEPTABLE
        elif efficiency_gain >= self.thresholds.min_research_efficiency_gain * 0.75:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_success_improvement(self, improvement: float) -> MetricStatus:
        """Evaluate query success improvement."""
        if improvement >= 0.30:
            return MetricStatus.EXCELLENT
        elif improvement >= 0.25:
            return MetricStatus.GOOD
        elif improvement >= self.thresholds.min_query_success_improvement:
            return MetricStatus.ACCEPTABLE
        elif improvement >= self.thresholds.min_query_success_improvement * 0.75:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_support_tickets(self, increase: float) -> MetricStatus:
        """Evaluate support ticket increase."""
        if increase <= 0:
            return MetricStatus.EXCELLENT
        elif increase <= self.thresholds.max_support_ticket_increase * 0.5:
            return MetricStatus.GOOD
        elif increase <= self.thresholds.max_support_ticket_increase:
            return MetricStatus.ACCEPTABLE
        elif increase <= self.thresholds.max_support_ticket_increase * 1.5:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def _evaluate_retention_rate(self, retention_rate: float) -> MetricStatus:
        """Evaluate user retention rate."""
        if retention_rate >= 0.98:
            return MetricStatus.EXCELLENT
        elif retention_rate >= 0.97:
            return MetricStatus.GOOD
        elif retention_rate >= self.thresholds.min_user_retention_rate:
            return MetricStatus.ACCEPTABLE
        elif retention_rate >= self.thresholds.min_user_retention_rate * 0.95:
            return MetricStatus.POOR
        else:
            return MetricStatus.FAILING

    def perform_comprehensive_validation(
        self, all_metrics: Dict[str, Any]
    ) -> ValidationReport:
        """Perform comprehensive validation of all metrics."""
        validation_id = f"validation_{int(time.time())}"
        timestamp = datetime.now()

        # Validate each category
        technical_metrics = self.validate_technical_performance(
            all_metrics.get("technical", {})
        )
        quality_metrics = self.validate_quality_metrics(all_metrics.get("quality", {}))
        user_experience_metrics = self.validate_user_experience(
            all_metrics.get("user_experience", {})
        )
        business_metrics = self.validate_business_impact(
            all_metrics.get("business", {})
        )

        # Determine overall result
        all_measurements = (
            technical_metrics
            + quality_metrics
            + user_experience_metrics
            + business_metrics
        )
        overall_result = self._determine_overall_result(all_measurements)

        # Generate summary
        summary = self._generate_validation_summary(all_measurements)

        # Generate recommendations
        recommendations = self._generate_recommendations(all_measurements)

        # Create validation report
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=timestamp,
            overall_result=overall_result,
            technical_metrics=technical_metrics,
            quality_metrics=quality_metrics,
            user_experience_metrics=user_experience_metrics,
            business_metrics=business_metrics,
            summary=summary,
            recommendations=recommendations,
            next_validation=timestamp + timedelta(hours=24),
        )

        # Store in history
        self.validation_history.append(report)

        # Keep only recent history (last 100 validations)
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]

        logger.info(f"Comprehensive validation completed: {overall_result.value}")

        return report

    def _determine_overall_result(
        self, measurements: List[MetricMeasurement]
    ) -> ValidationResult:
        """Determine overall validation result."""
        if not measurements:
            return ValidationResult.PENDING

        failing_count = sum(1 for m in measurements if m.status == MetricStatus.FAILING)
        poor_count = sum(1 for m in measurements if m.status == MetricStatus.POOR)

        total_count = len(measurements)

        # If more than 10% are failing, overall result is failed
        if failing_count / total_count > 0.1:
            return ValidationResult.FAILED

        # If more than 25% are poor or failing, result is warning
        if (failing_count + poor_count) / total_count > 0.25:
            return ValidationResult.WARNING

        # If any critical metrics are failing, result is failed
        critical_metrics = ["error_rate", "user_satisfaction", "task_completion_rate"]
        for measurement in measurements:
            if (
                measurement.name in critical_metrics
                and measurement.status == MetricStatus.FAILING
            ):
                return ValidationResult.FAILED

        return ValidationResult.PASSED

    def _generate_validation_summary(
        self, measurements: List[MetricMeasurement]
    ) -> Dict[str, Any]:
        """Generate validation summary."""
        if not measurements:
            return {"status": "no_metrics"}

        status_counts = {}
        for status in MetricStatus:
            status_counts[status.value] = sum(
                1 for m in measurements if m.status == status
            )

        total_count = len(measurements)
        passing_count = sum(1 for m in measurements if m.is_passing())

        return {
            "total_metrics": total_count,
            "passing_metrics": passing_count,
            "passing_percentage": (
                (passing_count / total_count) * 100 if total_count > 0 else 0
            ),
            "status_distribution": status_counts,
            "critical_failures": [
                m.name
                for m in measurements
                if m.status == MetricStatus.FAILING
                and m.name
                in ["error_rate", "user_satisfaction", "task_completion_rate"]
            ],
        }

    def _generate_recommendations(
        self, measurements: List[MetricMeasurement]
    ) -> List[str]:
        """Generate recommendations based on metric results."""
        recommendations = []

        for measurement in measurements:
            if measurement.status in [MetricStatus.FAILING, MetricStatus.POOR]:
                if measurement.name == "error_rate":
                    recommendations.append(
                        "High error rate detected. Review error logs and implement additional error handling."
                    )
                elif measurement.name == "response_time_p95":
                    recommendations.append(
                        "High response times detected. Consider optimizing reflection algorithms or adding caching."
                    )
                elif measurement.name == "memory_overhead":
                    recommendations.append(
                        "High memory overhead detected. Review memory usage patterns and implement optimization."
                    )
                elif measurement.name == "cpu_overhead":
                    recommendations.append(
                        "High CPU overhead detected. Consider optimizing computational complexity."
                    )
                elif measurement.name == "cache_hit_rate":
                    recommendations.append(
                        "Low cache hit rate detected. Review caching strategy and cache invalidation policies."
                    )
                elif measurement.name == "reflection_accuracy":
                    recommendations.append(
                        "Low reflection accuracy detected. Review and retrain reflection models."
                    )
                elif measurement.name == "user_satisfaction":
                    recommendations.append(
                        "Low user satisfaction detected. Conduct user research and improve UX."
                    )
                elif measurement.name == "feature_adoption_rate":
                    recommendations.append(
                        "Low feature adoption detected. Improve feature discoverability and user onboarding."
                    )

        # Add general recommendations
        if not recommendations:
            recommendations.append(
                "All metrics are performing well. Continue monitoring and maintain current practices."
            )

        return recommendations

    def get_validation_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get validation trends over specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_validations = [
            v for v in self.validation_history if v.timestamp >= cutoff_date
        ]

        if not recent_validations:
            return {"status": "no_data"}

        # Calculate trends
        trends = {}

        # Overall result trend
        results = [v.overall_result.value for v in recent_validations]
        trends["overall_result_trend"] = self._calculate_trend(results)

        # Metric-specific trends
        metric_names = set()
        for validation in recent_validations:
            all_metrics = (
                validation.technical_metrics
                + validation.quality_metrics
                + validation.user_experience_metrics
                + validation.business_metrics
            )
            metric_names.update(m.name for m in all_metrics)

        for metric_name in metric_names:
            values = []
            for validation in recent_validations:
                all_metrics = (
                    validation.technical_metrics
                    + validation.quality_metrics
                    + validation.user_experience_metrics
                    + validation.business_metrics
                )
                for metric in all_metrics:
                    if metric.name == metric_name:
                        values.append(metric.value)
                        break

            if len(values) >= 2:
                trends[f"{metric_name}_trend"] = self._calculate_numeric_trend(values)

        return {
            "period_days": days,
            "validation_count": len(recent_validations),
            "trends": trends,
            "latest_validation": (
                recent_validations[-1].validation_id if recent_validations else None
            ),
        }

    def _calculate_trend(self, values: List[str]) -> str:
        """Calculate trend for categorical values."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple trend based on recent vs older values
        recent_half = values[len(values) // 2 :]
        older_half = values[: len(values) // 2]

        recent_good = sum(
            1 for v in recent_half if v in ["passed", "excellent", "good"]
        )
        older_good = sum(1 for v in older_half if v in ["passed", "excellent", "good"])

        recent_ratio = recent_good / len(recent_half) if recent_half else 0
        older_ratio = older_good / len(older_half) if older_half else 0

        if recent_ratio > older_ratio + 0.1:
            return "improving"
        elif recent_ratio < older_ratio - 0.1:
            return "declining"
        else:
            return "stable"

    def _calculate_numeric_trend(self, values: List[float]) -> str:
        """Calculate trend for numeric values."""
        if len(values) < 3:
            return "insufficient_data"

        # Calculate linear regression slope
        n = len(values)
        x_values = list(range(n))

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Determine trend based on slope
        if abs(slope) < 0.01:  # Small threshold for stability
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def export_validation_report(
        self, report: ValidationReport, output_file: Optional[str] = None
    ) -> str:
        """Export validation report to JSON file."""
        if output_file is None:
            output_file = f"validation_report_{report.validation_id}.json"

        # Convert report to serializable format
        report_data = {
            "validation_id": report.validation_id,
            "timestamp": report.timestamp.isoformat(),
            "overall_result": report.overall_result.value,
            "technical_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status.value,
                    "timestamp": m.timestamp.isoformat(),
                    "details": m.details,
                }
                for m in report.technical_metrics
            ],
            "quality_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status.value,
                    "timestamp": m.timestamp.isoformat(),
                    "details": m.details,
                }
                for m in report.quality_metrics
            ],
            "user_experience_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status.value,
                    "timestamp": m.timestamp.isoformat(),
                    "details": m.details,
                }
                for m in report.user_experience_metrics
            ],
            "business_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status.value,
                    "timestamp": m.timestamp.isoformat(),
                    "details": m.details,
                }
                for m in report.business_metrics
            ],
            "summary": report.summary,
            "recommendations": report.recommendations,
            "next_validation": (
                report.next_validation.isoformat() if report.next_validation else None
            ),
        }

        # Write to file
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Validation report exported to {output_file}")
        return output_file


# Global validator instance
_success_validator: Optional[ReflectionSuccessValidator] = None


def get_success_validator(
    thresholds: Optional[SuccessThresholds] = None,
) -> ReflectionSuccessValidator:
    """Get or create success validator instance."""
    global _success_validator

    if _success_validator is None:
        _success_validator = ReflectionSuccessValidator(thresholds)

    return _success_validator
