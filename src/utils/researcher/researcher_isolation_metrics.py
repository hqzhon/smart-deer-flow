# -*- coding: utf-8 -*-
"""
Researcher Isolation Metrics - Phase 3 Implementation

Provides monitoring and metrics collection for researcher context isolation,
including performance tracking, token savings analysis, and user feedback collection.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class IsolationSession:
    """Single isolation session metrics"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    original_context_size: int = 0
    compressed_context_size: int = 0
    token_savings: int = 0
    isolation_level: str = "moderate"
    task_complexity: str = "medium"
    success: bool = True
    error_message: Optional[str] = None
    performance_impact: float = 0.0  # Execution time difference
    # Phase 2 - GFLQ Integration: Reflection metrics
    reflection_enabled: bool = False
    reflection_insights_count: int = 0
    knowledge_gaps_identified: int = 0
    follow_up_queries_generated: int = 0
    reflection_processing_time: float = 0.0
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio"""
        if self.original_context_size == 0:
            return 1.0
        return self.compressed_context_size / self.original_context_size
    
    @property
    def duration(self) -> float:
        """Calculate session duration"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


@dataclass
class IsolationMetricsSummary:
    """Summary of isolation metrics"""
    total_sessions: int = 0
    successful_sessions: int = 0
    total_token_savings: int = 0
    average_compression_ratio: float = 1.0
    average_performance_impact: float = 0.0
    isolation_effectiveness: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_sessions == 0:
            return 1.0
        return self.successful_sessions / self.total_sessions


class ResearcherIsolationMetrics:
    """Researcher isolation monitoring and metrics collection system"""
    
    def __init__(self, metrics_file: Optional[str] = None):
        self.metrics_file = metrics_file or "researcher_isolation_metrics.json"
        self.sessions: Dict[str, IsolationSession] = {}
        self.historical_sessions: List[IsolationSession] = []
        self._lock = Lock()
        
        # Load existing metrics
        self._load_metrics()
        
        # Performance tracking
        self.isolation_sessions = 0
        self.token_savings_estimated = 0
        self.context_compression_ratio = 0.0
        self.performance_overhead = 0.0
        
        # Phase 2 - GFLQ Integration: Reflection metrics tracking
        self.reflection_sessions = 0
        self.total_knowledge_gaps = 0
        self.total_follow_up_queries = 0
        self.average_reflection_time = 0.0
        
        logger.info(f"Initialized ResearcherIsolationMetrics with {len(self.historical_sessions)} historical sessions")
    
    def start_isolation_session(self, session_id: str, task_complexity: str = "medium", 
                              isolation_level: str = "moderate") -> IsolationSession:
        """Start tracking an isolation session"""
        with self._lock:
            session = IsolationSession(
                session_id=session_id,
                start_time=time.time(),
                task_complexity=task_complexity,
                isolation_level=isolation_level
            )
            self.sessions[session_id] = session
            logger.debug(f"Started isolation session: {session_id}")
            return session
    
    def update_session_context(self, session_id: str, original_size: int, compressed_size: int):
        """Update session context size information"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.original_context_size = original_size
                session.compressed_context_size = compressed_size
                session.token_savings = max(0, original_size - compressed_size)
                logger.debug(f"Updated context for session {session_id}: {original_size} -> {compressed_size}")
    
    def end_isolation_session(self, session_id: str, success: bool = True, 
                            error_message: Optional[str] = None, performance_impact: float = 0.0):
        """End an isolation session and record metrics"""
        with self._lock:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return
            
            session = self.sessions[session_id]
            session.end_time = time.time()
            session.success = success
            session.error_message = error_message
            session.performance_impact = performance_impact
            
            # Move to historical sessions
            self.historical_sessions.append(session)
            del self.sessions[session_id]
            
            # Update aggregate metrics
            self._update_aggregate_metrics(session)
            
            # Save metrics less frequently to improve performance
            if len(self.historical_sessions) % 50 == 0:
                self._save_metrics()
            
            logger.info(f"Completed isolation session {session_id}: success={success}, "
                       f"token_savings={session.token_savings}, compression={session.compression_ratio:.2f}")
    
    # Phase 2 - GFLQ Integration: Reflection metrics methods
    def update_reflection_metrics(self, session_id: str, insights_count: int = 0, 
                                knowledge_gaps: int = 0, follow_up_queries: int = 0, 
                                processing_time: float = 0.0):
        """Update reflection-related metrics for a session"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.reflection_enabled = True
                session.reflection_insights_count = insights_count
                session.knowledge_gaps_identified = knowledge_gaps
                session.follow_up_queries_generated = follow_up_queries
                session.reflection_processing_time = processing_time
                
                logger.debug(f"Updated reflection metrics for session {session_id}: "
                           f"insights={insights_count}, gaps={knowledge_gaps}, queries={follow_up_queries}")
    
    def get_reflection_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of reflection-related metrics"""
        with self._lock:
            reflection_sessions = [s for s in self.historical_sessions if s.reflection_enabled]
            
            if not reflection_sessions:
                return {
                    "total_reflection_sessions": 0,
                    "average_insights_per_session": 0.0,
                    "average_knowledge_gaps": 0.0,
                    "average_follow_up_queries": 0.0,
                    "average_processing_time": 0.0,
                    "reflection_effectiveness": 0.0
                }
            
            total_insights = sum(s.reflection_insights_count for s in reflection_sessions)
            total_gaps = sum(s.knowledge_gaps_identified for s in reflection_sessions)
            total_queries = sum(s.follow_up_queries_generated for s in reflection_sessions)
            total_time = sum(s.reflection_processing_time for s in reflection_sessions)
            
            return {
                "total_reflection_sessions": len(reflection_sessions),
                "average_insights_per_session": total_insights / len(reflection_sessions),
                "average_knowledge_gaps": total_gaps / len(reflection_sessions),
                "average_follow_up_queries": total_queries / len(reflection_sessions),
                "average_processing_time": total_time / len(reflection_sessions),
                "reflection_effectiveness": total_queries / max(1, total_gaps),  # Queries generated per gap
                "total_knowledge_gaps_identified": total_gaps,
                "total_follow_up_queries_generated": total_queries
            }
    
    def _update_aggregate_metrics(self, session: IsolationSession):
        """Update aggregate metrics with new session data"""
        self.isolation_sessions += 1
        
        if session.success:
            self.token_savings_estimated += session.token_savings
            
            # Update compression ratio (running average)
            self.context_compression_ratio = (
                self.context_compression_ratio * (self.isolation_sessions - 1) +
                session.compression_ratio
            ) / self.isolation_sessions
            
            # Update performance overhead (running average)
            self.performance_overhead = (
                self.performance_overhead * (self.isolation_sessions - 1) +
                session.performance_impact
            ) / self.isolation_sessions
            
            # Phase 2 - GFLQ Integration: Update reflection aggregate metrics
            if session.reflection_enabled:
                self.reflection_sessions += 1
                self.total_knowledge_gaps += session.knowledge_gaps_identified
                self.total_follow_up_queries += session.follow_up_queries_generated
                
                # Update average reflection time (running average)
                self.average_reflection_time = (
                    self.average_reflection_time * (self.reflection_sessions - 1) +
                    session.reflection_processing_time
                ) / self.reflection_sessions
    
    def get_isolation_health(self) -> Dict[str, Any]:
        """Get current isolation system health status"""
        with self._lock:
            recent_sessions = [s for s in self.historical_sessions 
                             if s.end_time and s.end_time > time.time() - 3600]  # Last hour
            
            return {
                "total_isolation_sessions": self.isolation_sessions,
                "estimated_token_savings": self.token_savings_estimated,
                "average_compression_ratio": self.context_compression_ratio,
                "average_performance_overhead": self.performance_overhead,
                "recent_sessions_count": len(recent_sessions),
                "recent_success_rate": sum(1 for s in recent_sessions if s.success) / len(recent_sessions) if recent_sessions else 1.0,
                "system_status": self._get_system_status(),
                "last_updated": datetime.now().isoformat()
            }
    
    def get_detailed_metrics(self) -> IsolationMetricsSummary:
        """Get detailed metrics summary"""
        with self._lock:
            successful_sessions = [s for s in self.historical_sessions if s.success]
            
            if not self.historical_sessions:
                return IsolationMetricsSummary()
            
            avg_compression = sum(s.compression_ratio for s in successful_sessions) / len(successful_sessions) if successful_sessions else 1.0
            avg_performance = sum(s.performance_impact for s in successful_sessions) / len(successful_sessions) if successful_sessions else 0.0
            
            # Calculate isolation effectiveness (token savings per session)
            effectiveness = self.token_savings_estimated / len(successful_sessions) if successful_sessions else 0.0
            
            return IsolationMetricsSummary(
                total_sessions=len(self.historical_sessions),
                successful_sessions=len(successful_sessions),
                total_token_savings=self.token_savings_estimated,
                average_compression_ratio=avg_compression,
                average_performance_impact=avg_performance,
                isolation_effectiveness=effectiveness
            )
    
    def _get_system_status(self) -> str:
        """Determine system status based on metrics"""
        if self.isolation_sessions == 0:
            return "initializing"
        
        recent_sessions = [s for s in self.historical_sessions 
                         if s.end_time and s.end_time > time.time() - 1800]  # Last 30 minutes
        
        if not recent_sessions:
            return "idle"
        
        success_rate = sum(1 for s in recent_sessions if s.success) / len(recent_sessions)
        
        if success_rate >= 0.95:
            return "healthy"
        elif success_rate >= 0.8:
            return "warning"
        else:
            return "critical"
    
    def should_enable_isolation(self, task_complexity: str, context_size: int, 
                              step_count: int) -> bool:
        """Determine if isolation should be enabled based on current metrics and task characteristics"""
        # Get current configuration
        from src.config.configuration import Configuration
        config = Configuration.get_current()
        
        if not config or not config.researcher_auto_isolation:
            return config.enable_researcher_isolation if config else False
        
        # Check threshold-based enablement
        if step_count >= (config.researcher_isolation_threshold if config else 3):
            return True
        
        # Check context size threshold
        if context_size >= (config.researcher_max_local_context if config else 5000):
            return True
        
        # Check task complexity
        complexity_thresholds = {
            "simple": False,
            "medium": step_count >= 2,
            "complex": True,
            "very_complex": True
        }
        
        if task_complexity in complexity_thresholds:
            return complexity_thresholds[task_complexity]
        
        # Default based on historical performance
        if self.isolation_sessions > 10:
            # Enable if average token savings > 1000 and success rate > 90%
            avg_savings = self.token_savings_estimated / self.isolation_sessions
            recent_success_rate = self._get_recent_success_rate()
            return avg_savings > 1000 and recent_success_rate > 0.9
        
        return False
    
    def _get_recent_success_rate(self) -> float:
        """Get success rate for recent sessions"""
        recent_sessions = [s for s in self.historical_sessions 
                         if s.end_time and s.end_time > time.time() - 3600]  # Last hour
        
        if not recent_sessions:
            return 1.0
        
        return sum(1 for s in recent_sessions if s.success) / len(recent_sessions)
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            # Calculate summary without calling get_detailed_metrics() to avoid performance overhead
            successful_sessions = [s for s in self.historical_sessions if s.success]
            
            if self.historical_sessions:
                avg_compression = sum(s.compression_ratio for s in successful_sessions) / len(successful_sessions) if successful_sessions else 1.0
                avg_performance = sum(s.performance_impact for s in successful_sessions) / len(successful_sessions) if successful_sessions else 0.0
                effectiveness = self.token_savings_estimated / len(successful_sessions) if successful_sessions else 0.0
                
                summary = {
                    "total_sessions": len(self.historical_sessions),
                    "successful_sessions": len(successful_sessions),
                    "total_token_savings": self.token_savings_estimated,
                    "average_compression_ratio": avg_compression,
                    "average_performance_impact": avg_performance,
                    "isolation_effectiveness": effectiveness,
                    "last_updated": datetime.now().isoformat()
                }
            else:
                summary = {
                    "total_sessions": 0,
                    "successful_sessions": 0,
                    "total_token_savings": 0,
                    "average_compression_ratio": 1.0,
                    "average_performance_impact": 0.0,
                    "isolation_effectiveness": 0.0,
                    "last_updated": datetime.now().isoformat()
                }
            
            metrics_data = {
                "summary": summary,
                "historical_sessions": [asdict(s) for s in self.historical_sessions[-100:]],  # Keep last 100
                "aggregate_metrics": {
                    "isolation_sessions": self.isolation_sessions,
                    "token_savings_estimated": self.token_savings_estimated,
                    "context_compression_ratio": self.context_compression_ratio,
                    "performance_overhead": self.performance_overhead
                }
            }
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_metrics(self):
        """Load metrics from file"""
        try:
            if Path(self.metrics_file).exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load historical sessions
                if "historical_sessions" in data:
                    self.historical_sessions = [
                        IsolationSession(**session_data) 
                        for session_data in data["historical_sessions"]
                    ]
                
                # Load aggregate metrics
                if "aggregate_metrics" in data:
                    agg = data["aggregate_metrics"]
                    self.isolation_sessions = agg.get("isolation_sessions", 0)
                    self.token_savings_estimated = agg.get("token_savings_estimated", 0)
                    self.context_compression_ratio = agg.get("context_compression_ratio", 0.0)
                    self.performance_overhead = agg.get("performance_overhead", 0.0)
                
                logger.info(f"Loaded metrics from {self.metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to load metrics: {e}")
    
    def export_metrics_report(self, output_file: Optional[str] = None) -> str:
        """Export comprehensive metrics report"""
        output_file = output_file or f"isolation_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "summary": asdict(self.get_detailed_metrics()),
            "health_status": self.get_isolation_health(),
            "session_analysis": self._analyze_sessions(),
            "recommendations": self._generate_recommendations()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported metrics report to {output_file}")
        return output_file
    
    def _analyze_sessions(self) -> Dict[str, Any]:
        """Analyze session patterns"""
        if not self.historical_sessions:
            return {"message": "No sessions to analyze"}
        
        # Group by isolation level
        by_level = {}
        for session in self.historical_sessions:
            level = session.isolation_level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(session)
        
        # Analyze each level
        level_analysis = {}
        for level, sessions in by_level.items():
            successful = [s for s in sessions if s.success]
            level_analysis[level] = {
                "total_sessions": len(sessions),
                "success_rate": len(successful) / len(sessions),
                "avg_token_savings": sum(s.token_savings for s in successful) / len(successful) if successful else 0,
                "avg_compression_ratio": sum(s.compression_ratio for s in successful) / len(successful) if successful else 1.0,
                "avg_performance_impact": sum(s.performance_impact for s in successful) / len(successful) if successful else 0.0
            }
        
        return {
            "by_isolation_level": level_analysis,
            "total_sessions_analyzed": len(self.historical_sessions),
            "analysis_period": {
                "start": min(s.start_time for s in self.historical_sessions),
                "end": max(s.end_time for s in self.historical_sessions if s.end_time)
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []
        
        if self.isolation_sessions < 10:
            recommendations.append("收集更多数据以获得更准确的性能分析")
            return recommendations
        
        # Analyze performance
        if self.context_compression_ratio > 0.8:
            recommendations.append("考虑使用更激进的隔离级别以提高压缩效果")
        
        if self.performance_overhead > 0.1:
            recommendations.append("性能开销较高，考虑优化隔离算法或降低隔离级别")
        
        if self.token_savings_estimated / self.isolation_sessions < 500:
            recommendations.append("Token节省效果不明显，建议调整隔离阈值")
        
        # Analyze recent success rate
        recent_success_rate = self._get_recent_success_rate()
        if recent_success_rate < 0.9:
            recommendations.append("最近成功率较低，建议检查隔离配置和错误日志")
        
        if not recommendations:
            recommendations.append("隔离系统运行良好，继续监控性能指标")
        
        return recommendations


# Global metrics instance
_global_metrics: Optional[ResearcherIsolationMetrics] = None


def get_isolation_metrics() -> ResearcherIsolationMetrics:
    """Get global isolation metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = ResearcherIsolationMetrics()
    return _global_metrics


def reset_isolation_metrics():
    """Reset global isolation metrics (for testing)"""
    global _global_metrics
    _global_metrics = None