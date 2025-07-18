# -*- coding: utf-8 -*-
"""
Researcher Isolation Metrics - Phase 4 Optimization

Advanced monitoring with real-time analytics, predictive capabilities,
and intelligent performance trend analysis.
"""

import time
import json
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque, defaultdict
from threading import Lock, Timer
from enum import Enum

# Import Phase 3 components
from .researcher_isolation_metrics import (
    ResearcherIsolationMetrics as BaseMetrics,
)

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""

    metric_name: str
    current_value: float
    trend_direction: str  # "improving", "degrading", "stable"
    trend_strength: float  # 0-1, how strong the trend is
    prediction_7d: float  # Predicted value in 7 days
    confidence: float  # 0-1, confidence in prediction
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SystemAlert:
    """System alert for monitoring"""

    alert_id: str
    level: AlertLevel
    title: str
    description: str
    metric_values: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    acknowledged: bool = False
    auto_resolved: bool = False


@dataclass
class RealTimeMetrics:
    """Real-time metrics snapshot"""

    timestamp: str
    active_sessions: int
    avg_compression_ratio_1h: float
    token_savings_rate_1h: float
    success_rate_1h: float
    performance_overhead_1h: float
    system_health_score: float  # 0-100
    resource_utilization: float  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PredictiveAnalyzer:
    """Predictive analysis for isolation metrics"""

    def __init__(self):
        self.historical_data: deque = deque(maxlen=1000)  # Store historical metrics
        self.trend_window = 50  # Number of data points for trend analysis

    def add_data_point(self, metrics: RealTimeMetrics):
        """Add new data point for analysis"""
        self.historical_data.append({"timestamp": time.time(), "metrics": metrics})

    def analyze_trends(self) -> Dict[str, PerformanceTrend]:
        """Analyze performance trends"""
        if len(self.historical_data) < 10:
            return {}

        trends = {}

        # Analyze key metrics
        metrics_to_analyze = [
            "avg_compression_ratio_1h",
            "token_savings_rate_1h",
            "success_rate_1h",
            "performance_overhead_1h",
            "system_health_score",
        ]

        for metric_name in metrics_to_analyze:
            trend = self._analyze_single_metric_trend(metric_name)
            if trend:
                trends[metric_name] = trend

        return trends

    def _analyze_single_metric_trend(
        self, metric_name: str
    ) -> Optional[PerformanceTrend]:
        """Analyze trend for a single metric"""
        # Extract values for the metric
        values = []
        timestamps = []

        for data_point in list(self.historical_data)[-self.trend_window :]:
            metrics = data_point["metrics"]
            if hasattr(metrics, metric_name):
                values.append(getattr(metrics, metric_name))
                timestamps.append(data_point["timestamp"])

        if len(values) < 5:
            return None

        # Calculate trend
        current_value = values[-1]

        # Simple linear trend analysis
        x = list(range(len(values)))
        y = values

        # Calculate slope (trend direction and strength)
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        if n * sum_x2 - sum_x**2 == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

        # Determine trend direction
        if abs(slope) < 0.001:  # Threshold for "stable"
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = (
                "improving" if metric_name != "performance_overhead_1h" else "degrading"
            )
            trend_strength = min(1.0, abs(slope) * 100)
        else:
            trend_direction = (
                "degrading" if metric_name != "performance_overhead_1h" else "improving"
            )
            trend_strength = min(1.0, abs(slope) * 100)

        # Predict future value (7 days = ~10080 minutes, but we use data points)
        future_steps = min(50, len(values))  # Predict based on recent pattern
        prediction_7d = current_value + slope * future_steps

        # Calculate confidence based on data consistency
        if len(values) >= 10:
            recent_variance = statistics.variance(values[-10:])
            confidence = max(0.1, min(1.0, 1.0 - recent_variance))
        else:
            confidence = 0.5

        return PerformanceTrend(
            metric_name=metric_name,
            current_value=current_value,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            prediction_7d=prediction_7d,
            confidence=confidence,
        )

    def predict_system_load(self, hours_ahead: int = 24) -> Dict[str, float]:
        """Predict system load for the next N hours"""
        if len(self.historical_data) < 20:
            return {"predicted_sessions": 0, "confidence": 0.0}

        # Extract session counts by hour
        hourly_sessions = defaultdict(list)

        for data_point in self.historical_data:
            timestamp = data_point["timestamp"]
            hour = int((timestamp % 86400) // 3600)  # Hour of day
            sessions = data_point["metrics"].active_sessions
            hourly_sessions[hour].append(sessions)

        # Predict based on historical patterns
        current_hour = int((time.time() % 86400) // 3600)
        target_hour = (current_hour + hours_ahead) % 24

        if target_hour in hourly_sessions and hourly_sessions[target_hour]:
            predicted_sessions = statistics.mean(hourly_sessions[target_hour])
            confidence = min(1.0, len(hourly_sessions[target_hour]) / 10.0)
        else:
            # Fallback to overall average
            all_sessions = [
                dp["metrics"].active_sessions for dp in self.historical_data
            ]
            predicted_sessions = statistics.mean(all_sessions) if all_sessions else 0
            confidence = 0.3

        return {
            "predicted_sessions": predicted_sessions,
            "confidence": confidence,
            "target_hour": target_hour,
        }


class AlertManager:
    """Manages system alerts and notifications"""

    def __init__(self):
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: deque = deque(maxlen=500)
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []
        self._lock = Lock()

    def add_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    def check_and_create_alerts(
        self, metrics: RealTimeMetrics, trends: Dict[str, PerformanceTrend]
    ):
        """Check metrics and create alerts if needed"""
        with self._lock:
            # Check for critical conditions
            self._check_success_rate_alert(metrics)
            self._check_performance_degradation_alert(metrics, trends)
            self._check_resource_utilization_alert(metrics)
            self._check_trend_alerts(trends)

            # Auto-resolve alerts if conditions improve
            self._auto_resolve_alerts(metrics)

    def _check_success_rate_alert(self, metrics: RealTimeMetrics):
        """Check for low success rate alerts"""
        alert_id = "low_success_rate"

        if metrics.success_rate_1h < 0.8:
            level = (
                AlertLevel.CRITICAL
                if metrics.success_rate_1h < 0.6
                else AlertLevel.WARNING
            )

            if alert_id not in self.active_alerts:
                alert = SystemAlert(
                    alert_id=alert_id,
                    level=level,
                    title="隔离成功率低",
                    description=f"最近1小时隔离成功率为 {metrics.success_rate_1h:.1%}，低于正常水平",
                    metric_values={"success_rate_1h": metrics.success_rate_1h},
                )
                self._create_alert(alert)

    def _check_performance_degradation_alert(
        self, metrics: RealTimeMetrics, trends: Dict[str, PerformanceTrend]
    ):
        """Check for performance degradation alerts"""
        alert_id = "performance_degradation"

        # Check current overhead
        if metrics.performance_overhead_1h > 0.3:
            if alert_id not in self.active_alerts:
                alert = SystemAlert(
                    alert_id=alert_id,
                    level=AlertLevel.WARNING,
                    title="性能开销过高",
                    description=f"隔离性能开销为 {metrics.performance_overhead_1h:.1%}，超过建议阈值",
                    metric_values={
                        "performance_overhead_1h": metrics.performance_overhead_1h
                    },
                )
                self._create_alert(alert)

        # Check trend
        if "performance_overhead_1h" in trends:
            trend = trends["performance_overhead_1h"]
            if trend.trend_direction == "degrading" and trend.trend_strength > 0.5:
                trend_alert_id = "performance_trend_degrading"
                if trend_alert_id not in self.active_alerts:
                    alert = SystemAlert(
                        alert_id=trend_alert_id,
                        level=AlertLevel.WARNING,
                        title="性能趋势恶化",
                        description=f"性能开销呈恶化趋势，强度: {trend.trend_strength:.1f}",
                        metric_values={"trend_strength": trend.trend_strength},
                    )
                    self._create_alert(alert)

    def _check_resource_utilization_alert(self, metrics: RealTimeMetrics):
        """Check for high resource utilization alerts"""
        alert_id = "high_resource_utilization"

        if metrics.resource_utilization > 0.9:
            level = (
                AlertLevel.CRITICAL
                if metrics.resource_utilization > 0.95
                else AlertLevel.WARNING
            )

            if alert_id not in self.active_alerts:
                alert = SystemAlert(
                    alert_id=alert_id,
                    level=level,
                    title="资源利用率过高",
                    description=f"系统资源利用率为 {metrics.resource_utilization:.1%}，接近上限",
                    metric_values={
                        "resource_utilization": metrics.resource_utilization
                    },
                )
                self._create_alert(alert)

    def _check_trend_alerts(self, trends: Dict[str, PerformanceTrend]):
        """Check for trend-based alerts"""
        for metric_name, trend in trends.items():
            if trend.trend_direction == "degrading" and trend.trend_strength > 0.7:
                alert_id = f"trend_degrading_{metric_name}"

                if alert_id not in self.active_alerts:
                    alert = SystemAlert(
                        alert_id=alert_id,
                        level=AlertLevel.WARNING,
                        title=f"{metric_name} 趋势恶化",
                        description=f"{metric_name} 呈现强烈恶化趋势 (强度: {trend.trend_strength:.1f})",
                        metric_values={
                            "trend_strength": trend.trend_strength,
                            "current_value": trend.current_value,
                        },
                    )
                    self._create_alert(alert)

    def _auto_resolve_alerts(self, metrics: RealTimeMetrics):
        """Auto-resolve alerts when conditions improve"""
        to_resolve = []

        for alert_id, alert in self.active_alerts.items():
            should_resolve = False

            if alert_id == "low_success_rate" and metrics.success_rate_1h >= 0.85:
                should_resolve = True
            elif (
                alert_id == "performance_degradation"
                and metrics.performance_overhead_1h <= 0.25
            ):
                should_resolve = True
            elif (
                alert_id == "high_resource_utilization"
                and metrics.resource_utilization <= 0.85
            ):
                should_resolve = True

            if should_resolve:
                to_resolve.append(alert_id)

        for alert_id in to_resolve:
            self._resolve_alert(alert_id, auto_resolved=True)

    def _create_alert(self, alert: SystemAlert):
        """Create a new alert"""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Alert created: {alert.title} ({alert.level.value})")

    def _resolve_alert(self, alert_id: str, auto_resolved: bool = False):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.auto_resolved = auto_resolved
            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved: {alert.title} (auto: {auto_resolved})")

    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                return True
            return False


class AdvancedResearcherIsolationMetrics(BaseMetrics):
    """Advanced isolation metrics with real-time monitoring and predictive analytics"""

    def __init__(self, metrics_file: Optional[str] = None):
        super().__init__(metrics_file)

        # Advanced components
        self.predictive_analyzer = PredictiveAnalyzer()
        self.alert_manager = AlertManager()

        # Real-time monitoring
        self.real_time_metrics: deque = deque(
            maxlen=1440
        )  # 24 hours of minute-by-minute data
        self.monitoring_enabled = True
        self.monitoring_interval = 60  # seconds

        # Performance optimization
        self.optimization_suggestions: List[str] = []
        self.auto_optimization_enabled = False

        # Start monitoring thread
        self._start_monitoring()

        logger.info(
            "Initialized AdvancedResearcherIsolationMetrics with predictive analytics"
        )

    def _start_monitoring(self):
        """Start real-time monitoring thread"""
        if self.monitoring_enabled:
            timer = Timer(self.monitoring_interval, self._collect_real_time_metrics)
            timer.daemon = True
            timer.start()

    def _collect_real_time_metrics(self):
        """Collect real-time metrics snapshot"""
        try:
            # Calculate current metrics
            current_time = datetime.now().isoformat()

            # Get recent sessions (last hour)
            recent_sessions = [
                s
                for s in self.historical_sessions
                if s.end_time and s.end_time > time.time() - 3600
            ]

            if recent_sessions:
                avg_compression = statistics.mean(
                    [s.compression_ratio for s in recent_sessions]
                )
                token_savings_rate = sum(
                    [s.token_savings for s in recent_sessions]
                ) / len(recent_sessions)
                success_rate = sum([1 for s in recent_sessions if s.success]) / len(
                    recent_sessions
                )
                performance_overhead = statistics.mean(
                    [s.performance_impact for s in recent_sessions]
                )
            else:
                avg_compression = 1.0
                token_savings_rate = 0.0
                success_rate = 1.0
                performance_overhead = 0.0

            # Calculate system health score
            health_score = self._calculate_system_health_score(
                success_rate, performance_overhead, avg_compression
            )

            # Estimate resource utilization
            resource_utilization = min(
                1.0, len(self.sessions) / 10.0 + performance_overhead
            )

            # Create metrics snapshot
            metrics_snapshot = RealTimeMetrics(
                timestamp=current_time,
                active_sessions=len(self.sessions),
                avg_compression_ratio_1h=avg_compression,
                token_savings_rate_1h=token_savings_rate,
                success_rate_1h=success_rate,
                performance_overhead_1h=performance_overhead,
                system_health_score=health_score,
                resource_utilization=resource_utilization,
            )

            # Store metrics
            self.real_time_metrics.append(metrics_snapshot)

            # Add to predictive analyzer
            self.predictive_analyzer.add_data_point(metrics_snapshot)

            # Analyze trends and check alerts
            trends = self.predictive_analyzer.analyze_trends()
            self.alert_manager.check_and_create_alerts(metrics_snapshot, trends)

            # Generate optimization suggestions
            self._update_optimization_suggestions(metrics_snapshot, trends)

            # Schedule next collection
            if self.monitoring_enabled:
                timer = Timer(self.monitoring_interval, self._collect_real_time_metrics)
                timer.daemon = True
                timer.start()

        except Exception as e:
            logger.error(f"Error collecting real-time metrics: {e}")
            # Restart monitoring after error
            if self.monitoring_enabled:
                timer = Timer(
                    self.monitoring_interval * 2, self._collect_real_time_metrics
                )
                timer.daemon = True
                timer.start()

    def _calculate_system_health_score(
        self, success_rate: float, performance_overhead: float, compression_ratio: float
    ) -> float:
        """Calculate overall system health score (0-100)"""
        # Weight different factors
        success_weight = 0.4
        performance_weight = 0.3
        compression_weight = 0.3

        # Normalize scores
        success_score = success_rate * 100
        performance_score = max(
            0, 100 - performance_overhead * 200
        )  # Lower overhead = higher score
        compression_score = min(
            100, (1 - compression_ratio) * 200
        )  # Better compression = higher score

        health_score = (
            success_score * success_weight
            + performance_score * performance_weight
            + compression_score * compression_weight
        )

        return max(0, min(100, health_score))

    def _update_optimization_suggestions(
        self, metrics: RealTimeMetrics, trends: Dict[str, PerformanceTrend]
    ):
        """Update optimization suggestions based on current metrics and trends"""
        suggestions = []

        # Performance-based suggestions
        if metrics.performance_overhead_1h > 0.25:
            suggestions.append("考虑降低隔离级别以减少性能开销")

        if metrics.avg_compression_ratio_1h > 0.8:
            suggestions.append("压缩效果不佳，建议调整隔离策略")

        if metrics.success_rate_1h < 0.9:
            suggestions.append("成功率较低，检查隔离配置和错误日志")

        # Trend-based suggestions
        for metric_name, trend in trends.items():
            if trend.trend_direction == "degrading" and trend.trend_strength > 0.6:
                if metric_name == "success_rate_1h":
                    suggestions.append("成功率呈下降趋势，建议检查系统稳定性")
                elif metric_name == "performance_overhead_1h":
                    suggestions.append("性能开销呈上升趋势，考虑优化隔离算法")

        # Resource utilization suggestions
        if metrics.resource_utilization > 0.8:
            suggestions.append("资源利用率较高，考虑限制并发隔离会话数量")

        # Update suggestions (keep only recent unique suggestions)
        time.time()
        self.optimization_suggestions = [
            f"{datetime.now().strftime('%H:%M')} - {suggestion}"
            for suggestion in suggestions
        ]

    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        if not self.real_time_metrics:
            return {"message": "No real-time data available"}

        latest_metrics = self.real_time_metrics[-1]
        trends = self.predictive_analyzer.analyze_trends()
        active_alerts = self.alert_manager.get_active_alerts()

        # Calculate 24h summary
        if len(self.real_time_metrics) >= 60:  # At least 1 hour of data
            last_24h = list(self.real_time_metrics)[-1440:]  # Last 24 hours
            avg_health_24h = statistics.mean([m.system_health_score for m in last_24h])
            avg_sessions_24h = statistics.mean([m.active_sessions for m in last_24h])
        else:
            avg_health_24h = latest_metrics.system_health_score
            avg_sessions_24h = latest_metrics.active_sessions

        return {
            "current_metrics": latest_metrics.to_dict(),
            "trends": {name: asdict(trend) for name, trend in trends.items()},
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "24h_summary": {
                "avg_health_score": avg_health_24h,
                "avg_active_sessions": avg_sessions_24h,
                "data_points": len(self.real_time_metrics),
            },
            "optimization_suggestions": self.optimization_suggestions,
            "system_status": self._get_overall_system_status(
                latest_metrics, active_alerts
            ),
            "last_updated": latest_metrics.timestamp,
        }

    def _get_overall_system_status(
        self, metrics: RealTimeMetrics, alerts: List[SystemAlert]
    ) -> str:
        """Get overall system status"""
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]

        if critical_alerts:
            return "critical"
        elif warning_alerts:
            return "warning"
        elif metrics.system_health_score >= 80:
            return "healthy"
        elif metrics.system_health_score >= 60:
            return "degraded"
        else:
            return "unhealthy"

    def get_predictive_insights(self) -> Dict[str, Any]:
        """Get predictive insights and forecasts"""
        trends = self.predictive_analyzer.analyze_trends()
        load_prediction = self.predictive_analyzer.predict_system_load(24)

        insights = {
            "trends_analysis": {name: asdict(trend) for name, trend in trends.items()},
            "load_prediction_24h": load_prediction,
            "recommendations": [],
        }

        # Generate recommendations based on predictions
        for metric_name, trend in trends.items():
            if trend.trend_direction == "degrading" and trend.confidence > 0.7:
                if metric_name == "success_rate_1h":
                    insights["recommendations"].append(
                        f"预计成功率将在7天内降至 {trend.prediction_7d:.1%}，建议提前优化"
                    )
                elif metric_name == "performance_overhead_1h":
                    insights["recommendations"].append(
                        f"预计性能开销将在7天内升至 {trend.prediction_7d:.1%}，建议调整配置"
                    )

        # Load-based recommendations
        if load_prediction["confidence"] > 0.6:
            predicted_sessions = load_prediction["predicted_sessions"]
            if predicted_sessions > 10:
                insights["recommendations"].append(
                    f"预计24小时内会话数将达到 {predicted_sessions:.0f}，建议准备扩容"
                )

        return insights

    def export_advanced_report(self, output_file: Optional[str] = None) -> str:
        """Export comprehensive advanced metrics report"""
        output_file = (
            output_file
            or f"advanced_isolation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        dashboard_data = self.get_real_time_dashboard()
        predictive_insights = self.get_predictive_insights()

        report = {
            "report_type": "advanced_isolation_metrics",
            "generated_at": datetime.now().isoformat(),
            "dashboard_data": dashboard_data,
            "predictive_insights": predictive_insights,
            "historical_summary": asdict(self.get_detailed_metrics()),
            "alert_history": [
                asdict(alert) for alert in list(self.alert_manager.alert_history)[-50:]
            ],
            "monitoring_config": {
                "monitoring_enabled": self.monitoring_enabled,
                "monitoring_interval": self.monitoring_interval,
                "auto_optimization_enabled": self.auto_optimization_enabled,
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported advanced metrics report to {output_file}")
        return output_file

    def enable_auto_optimization(self):
        """Enable automatic optimization based on metrics"""
        self.auto_optimization_enabled = True
        logger.info("Auto-optimization enabled")

    def disable_auto_optimization(self):
        """Disable automatic optimization"""
        self.auto_optimization_enabled = False
        logger.info("Auto-optimization disabled")

    def add_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Add callback for alert notifications"""
        self.alert_manager.add_alert_callback(callback)

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_enabled = False
        logger.info("Real-time monitoring stopped")


# Global advanced metrics instance
_global_advanced_metrics: Optional[AdvancedResearcherIsolationMetrics] = None


def get_advanced_isolation_metrics() -> AdvancedResearcherIsolationMetrics:
    """Get global advanced isolation metrics instance"""
    global _global_advanced_metrics
    if _global_advanced_metrics is None:
        _global_advanced_metrics = AdvancedResearcherIsolationMetrics()
    return _global_advanced_metrics


def reset_advanced_isolation_metrics():
    """Reset global advanced isolation metrics (for testing)"""
    global _global_advanced_metrics
    if _global_advanced_metrics:
        _global_advanced_metrics.stop_monitoring()
    _global_advanced_metrics = None
