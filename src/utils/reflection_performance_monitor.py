# -*- coding: utf-8 -*-
"""
Reflection Performance Monitor

This module implements performance monitoring for the enhanced reflection system,
tracking key metrics, success indicators, and system health for Phase 5 integration.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import threading

logger = logging.getLogger(__name__)


@dataclass
class ReflectionMetrics:
    """Reflection performance metrics."""

    # Technical indicators
    reflection_accuracy: float = 0.0
    knowledge_gap_identification_rate: float = 0.0
    system_response_time_increase: float = 0.0
    error_rate: float = 0.0

    # Business indicators
    research_completeness_score: float = 0.0
    user_satisfaction_score: float = 0.0
    query_success_rate: float = 0.0
    average_research_time_reduction: float = 0.0

    # Operational metrics
    cache_hit_rate: float = 0.0
    reflection_loop_count: int = 0
    follow_up_query_effectiveness: float = 0.0
    resource_utilization: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReflectionEvent:
    """Individual reflection event for tracking."""

    event_id: str
    event_type: str  # 'knowledge_gap_analysis', 'sufficiency_assessment', 'follow_up_generation'
    node_type: str  # 'researcher', 'planner'
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    confidence: float = 0.0
    sufficiency_score: float = 0.0
    gaps_identified: int = 0
    follow_up_queries_generated: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflectionPerformanceMonitor:
    """Monitor and track reflection system performance."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: deque = deque(maxlen=1000)
        self.events_history: deque = deque(maxlen=5000)
        self.current_metrics = ReflectionMetrics()
        self.monitoring_enabled = config.get("enable_reflection_monitoring", True)
        self.metrics_lock = threading.Lock()

        # Performance thresholds from plan
        self.target_thresholds = {
            "reflection_accuracy": 0.85,
            "knowledge_gap_identification_rate": 0.80,
            "system_response_time_increase": 0.20,
            "error_rate": 0.02,
            "research_completeness_score_improvement": 0.30,
            "user_satisfaction_improvement": 0.25,
            "query_success_rate_improvement": 0.20,
            "average_research_time_reduction": 0.15,
        }

        # Trend tracking
        self.trend_window = timedelta(hours=24)
        self.alert_callbacks: List[callable] = []

        logger.info("Reflection Performance Monitor initialized")

    def record_reflection_event(self, event: ReflectionEvent) -> None:
        """Record a reflection event."""
        if not self.monitoring_enabled:
            return

        with self.metrics_lock:
            event.end_time = datetime.now()
            self.events_history.append(event)

            # Update real-time metrics
            self._update_real_time_metrics(event)

            logger.debug(f"Recorded reflection event: {event.event_id}")

    def _update_real_time_metrics(self, event: ReflectionEvent) -> None:
        """Update real-time metrics based on new event."""
        try:
            # Calculate recent events metrics
            recent_events = self._get_recent_events(hours=1)

            if recent_events:
                # Technical metrics
                successful_events = [e for e in recent_events if e.success]
                self.current_metrics.reflection_accuracy = len(successful_events) / len(
                    recent_events
                )

                gap_events = [e for e in recent_events if e.gaps_identified > 0]
                self.current_metrics.knowledge_gap_identification_rate = len(
                    gap_events
                ) / len(recent_events)

                error_events = [e for e in recent_events if e.error_message]
                self.current_metrics.error_rate = len(error_events) / len(recent_events)

                # Performance metrics
                if successful_events:
                    avg_confidence = sum(e.confidence for e in successful_events) / len(
                        successful_events
                    )
                    self.current_metrics.research_completeness_score = avg_confidence

                    avg_sufficiency = sum(
                        e.sufficiency_score for e in successful_events
                    ) / len(successful_events)
                    self.current_metrics.user_satisfaction_score = avg_sufficiency

                # Cache and efficiency metrics
                cache_events = [
                    e for e in recent_events if e.metadata.get("cache_hit", False)
                ]
                self.current_metrics.cache_hit_rate = (
                    len(cache_events) / len(recent_events) if recent_events else 0
                )

                # Update timestamp
                self.current_metrics.timestamp = datetime.now()

        except Exception as e:
            logger.error(f"Error updating real-time metrics: {e}")

    def _get_recent_events(self, hours: int = 24) -> List[ReflectionEvent]:
        """Get events from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.events_history if event.start_time >= cutoff_time
        ]

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        with self.metrics_lock:
            recent_events = self._get_recent_events(hours=24)

            dashboard = {
                "current_metrics": {
                    "reflection_accuracy": self.current_metrics.reflection_accuracy,
                    "knowledge_gap_identification_rate": self.current_metrics.knowledge_gap_identification_rate,
                    "error_rate": self.current_metrics.error_rate,
                    "research_completeness_score": self.current_metrics.research_completeness_score,
                    "cache_hit_rate": self.current_metrics.cache_hit_rate,
                    "last_updated": self.current_metrics.timestamp.isoformat(),
                },
                "target_thresholds": self.target_thresholds,
                "performance_status": self._assess_performance_status(),
                "recent_activity": {
                    "total_events_24h": len(recent_events),
                    "successful_events_24h": len(
                        [e for e in recent_events if e.success]
                    ),
                    "error_events_24h": len(
                        [e for e in recent_events if e.error_message]
                    ),
                    "avg_confidence_24h": self._calculate_average_confidence(
                        recent_events
                    ),
                    "avg_gaps_identified_24h": self._calculate_average_gaps(
                        recent_events
                    ),
                },
                "trends": self._calculate_trends(),
                "alerts": self._check_performance_alerts(),
            }

            return dashboard

    def _assess_performance_status(self) -> str:
        """Assess overall performance status."""
        metrics = self.current_metrics
        thresholds = self.target_thresholds

        # Check critical metrics
        if (
            metrics.reflection_accuracy >= thresholds["reflection_accuracy"]
            and metrics.knowledge_gap_identification_rate
            >= thresholds["knowledge_gap_identification_rate"]
            and metrics.error_rate <= thresholds["error_rate"]
        ):
            return "excellent"
        elif (
            metrics.reflection_accuracy >= thresholds["reflection_accuracy"] * 0.9
            and metrics.error_rate <= thresholds["error_rate"] * 1.5
        ):
            return "good"
        elif metrics.error_rate <= thresholds["error_rate"] * 2:
            return "fair"
        else:
            return "poor"

    def _calculate_average_confidence(self, events: List[ReflectionEvent]) -> float:
        """Calculate average confidence from events."""
        successful_events = [e for e in events if e.success and e.confidence > 0]
        if not successful_events:
            return 0.0
        return sum(e.confidence for e in successful_events) / len(successful_events)

    def _calculate_average_gaps(self, events: List[ReflectionEvent]) -> float:
        """Calculate average gaps identified from events."""
        if not events:
            return 0.0
        return sum(e.gaps_identified for e in events) / len(events)

    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends."""
        try:
            # Get events from different time windows
            events_1h = self._get_recent_events(hours=1)
            events_24h = self._get_recent_events(hours=24)
            events_7d = self._get_recent_events(hours=168)  # 7 days

            trends = {}

            # Accuracy trend
            acc_1h = (
                len([e for e in events_1h if e.success]) / len(events_1h)
                if events_1h
                else 0
            )
            acc_24h = (
                len([e for e in events_24h if e.success]) / len(events_24h)
                if events_24h
                else 0
            )
            trends["accuracy_trend"] = self._determine_trend(acc_1h, acc_24h)

            # Error rate trend
            err_1h = (
                len([e for e in events_1h if e.error_message]) / len(events_1h)
                if events_1h
                else 0
            )
            err_24h = (
                len([e for e in events_24h if e.error_message]) / len(events_24h)
                if events_24h
                else 0
            )
            trends["error_rate_trend"] = self._determine_trend(
                err_24h, err_1h
            )  # Inverted for error rate

            # Volume trend
            vol_1h = len(events_1h)
            vol_24h_avg = len(events_24h) / 24 if events_24h else 0
            trends["volume_trend"] = self._determine_trend(vol_1h, vol_24h_avg)

            return trends

        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {}

    def _determine_trend(self, current: float, baseline: float) -> str:
        """Determine trend direction."""
        if baseline == 0:
            return "stable"

        change_ratio = (current - baseline) / baseline

        if change_ratio > 0.1:
            return "improving"
        elif change_ratio < -0.1:
            return "degrading"
        else:
            return "stable"

    def _check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        metrics = self.current_metrics
        thresholds = self.target_thresholds

        # Critical alerts
        if metrics.error_rate > thresholds["error_rate"] * 2:
            alerts.append(
                {
                    "level": "critical",
                    "metric": "error_rate",
                    "current_value": metrics.error_rate,
                    "threshold": thresholds["error_rate"],
                    "message": f"Error rate ({metrics.error_rate:.2%}) exceeds critical threshold",
                }
            )

        # Warning alerts
        if metrics.reflection_accuracy < thresholds["reflection_accuracy"] * 0.9:
            alerts.append(
                {
                    "level": "warning",
                    "metric": "reflection_accuracy",
                    "current_value": metrics.reflection_accuracy,
                    "threshold": thresholds["reflection_accuracy"],
                    "message": f"Reflection accuracy ({metrics.reflection_accuracy:.2%}) below target",
                }
            )

        if (
            metrics.knowledge_gap_identification_rate
            < thresholds["knowledge_gap_identification_rate"] * 0.9
        ):
            alerts.append(
                {
                    "level": "warning",
                    "metric": "knowledge_gap_identification_rate",
                    "current_value": metrics.knowledge_gap_identification_rate,
                    "threshold": thresholds["knowledge_gap_identification_rate"],
                    "message": f"Gap identification rate ({metrics.knowledge_gap_identification_rate:.2%}) below target",
                }
            )

        return alerts

    def add_alert_callback(self, callback: callable) -> None:
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)

    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file."""
        try:
            dashboard_data = self.get_performance_dashboard()

            with open(filepath, "w") as f:
                json.dump(dashboard_data, f, indent=2, default=str)

            logger.info(f"Metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        dashboard = self.get_performance_dashboard()

        health_report = {
            "overall_health": dashboard["performance_status"],
            "key_metrics": {
                "reflection_accuracy": {
                    "current": dashboard["current_metrics"]["reflection_accuracy"],
                    "target": self.target_thresholds["reflection_accuracy"],
                    "status": (
                        "pass"
                        if dashboard["current_metrics"]["reflection_accuracy"]
                        >= self.target_thresholds["reflection_accuracy"]
                        else "fail"
                    ),
                },
                "error_rate": {
                    "current": dashboard["current_metrics"]["error_rate"],
                    "target": self.target_thresholds["error_rate"],
                    "status": (
                        "pass"
                        if dashboard["current_metrics"]["error_rate"]
                        <= self.target_thresholds["error_rate"]
                        else "fail"
                    ),
                },
                "knowledge_gap_identification": {
                    "current": dashboard["current_metrics"][
                        "knowledge_gap_identification_rate"
                    ],
                    "target": self.target_thresholds[
                        "knowledge_gap_identification_rate"
                    ],
                    "status": (
                        "pass"
                        if dashboard["current_metrics"][
                            "knowledge_gap_identification_rate"
                        ]
                        >= self.target_thresholds["knowledge_gap_identification_rate"]
                        else "fail"
                    ),
                },
            },
            "recommendations": self._generate_recommendations(dashboard),
            "generated_at": datetime.now().isoformat(),
        }

        return health_report

    def _generate_recommendations(self, dashboard: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        metrics = dashboard["current_metrics"]

        if (
            metrics["reflection_accuracy"]
            < self.target_thresholds["reflection_accuracy"]
        ):
            recommendations.append(
                "Consider tuning reflection model parameters or increasing training data quality"
            )

        if metrics["error_rate"] > self.target_thresholds["error_rate"]:
            recommendations.append(
                "Investigate error patterns and implement additional error handling"
            )

        if metrics["cache_hit_rate"] < 0.5:
            recommendations.append(
                "Optimize reflection caching strategy to improve performance"
            )

        if dashboard["recent_activity"]["total_events_24h"] < 10:
            recommendations.append(
                "Low reflection activity detected - verify system integration"
            )

        return recommendations


# Global monitor instance
_monitor_instance: Optional[ReflectionPerformanceMonitor] = None


def get_performance_monitor(
    config: Optional[Dict[str, Any]] = None,
) -> ReflectionPerformanceMonitor:
    """Get or create performance monitor instance."""
    global _monitor_instance

    if _monitor_instance is None:
        if config is None:
            config = {"enable_reflection_monitoring": True}
        _monitor_instance = ReflectionPerformanceMonitor(config)

    return _monitor_instance


def record_reflection_event(event_type: str, node_type: str, **kwargs) -> str:
    """Convenience function to record reflection event."""
    monitor = get_performance_monitor()

    event = ReflectionEvent(
        event_id=f"{event_type}_{node_type}_{int(time.time() * 1000)}",
        event_type=event_type,
        node_type=node_type,
        start_time=datetime.now(),
        **kwargs,
    )

    monitor.record_reflection_event(event)
    return event.event_id
