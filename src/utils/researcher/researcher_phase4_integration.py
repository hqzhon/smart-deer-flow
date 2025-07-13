# -*- coding: utf-8 -*-
"""
Researcher Phase 4 Integration Module

Integrates all Phase 4 optimizations into a unified system with
intelligent coordination between components.
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Timer, Lock
from enum import Enum

# Import Phase 4 components
from .researcher_progressive_enablement_phase4 import (
    AdvancedResearcherProgressiveEnabler,
    get_advanced_progressive_enabler
)
from .researcher_isolation_metrics_phase4 import (
    AdvancedResearcherIsolationMetrics,
    get_advanced_isolation_metrics,
    SystemAlert, AlertLevel
)
from .researcher_config_optimizer_phase4 import (
    ConfigurationOptimizer,
    get_config_optimizer,
    ConfigOptimizationLevel
)

# Import Phase 3 components for backward compatibility
from .researcher_context_extension import ResearcherContextExtension
from .researcher_context_isolator import ResearcherContextIsolator

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operation modes"""
    LEARNING = "learning"        # Collecting data, conservative decisions
    OPTIMIZING = "optimizing"    # Active optimization based on data
    STABLE = "stable"           # Stable operation with minimal changes
    EMERGENCY = "emergency"      # Emergency mode with safe fallbacks


@dataclass
class SystemStatus:
    """Overall system status"""
    mode: SystemMode
    health_score: float  # 0-100
    active_sessions: int
    optimization_score: float  # 0-100
    last_optimization: Optional[str]
    alerts_count: int
    uptime_hours: float
    performance_trend: str  # "improving", "stable", "degrading"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enum to string for JSON serialization
        data['mode'] = self.mode.value
        return data


class ResearcherPhase4System:
    """Unified Phase 4 system coordinator"""
    
    def __init__(self, enable_auto_optimization: bool = False):
        # Initialize components
        self.progressive_enabler = get_advanced_progressive_enabler()
        self.metrics_system = get_advanced_isolation_metrics()
        self.config_optimizer = get_config_optimizer()
        
        # System state
        self.system_mode = SystemMode.LEARNING
        self.start_time = time.time()
        self.auto_optimization_enabled = enable_auto_optimization
        
        # Coordination settings
        self.coordination_interval = 300  # 5 minutes
        self.learning_period_hours = 24   # Learn for 24 hours before optimizing
        self.optimization_cooldown = 3600 # 1 hour between optimizations
        self.last_optimization_time = 0
        
        # Callbacks and hooks
        self.status_change_callbacks: List[Callable[[SystemStatus], None]] = []
        self.optimization_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Thread safety
        self._lock = Lock()
        
        # Setup alert handling
        self.metrics_system.add_alert_callback(self._handle_system_alert)
        
        # Start coordination loop
        self._start_coordination_loop()
        
        logger.info("Initialized ResearcherPhase4System")
    
    def _start_coordination_loop(self):
        """Start the system coordination loop"""
        timer = Timer(self.coordination_interval, self._coordination_cycle)
        timer.daemon = True
        timer.start()
    
    def _coordination_cycle(self):
        """Main coordination cycle"""
        try:
            with self._lock:
                # Get current system status
                status = self._calculate_system_status()
                
                # Determine if mode change is needed
                new_mode = self._determine_optimal_mode(status)
                if new_mode != self.system_mode:
                    self._change_system_mode(new_mode, status)
                
                # Perform mode-specific actions
                self._execute_mode_actions(status)
                
                # Notify status change callbacks
                for callback in self.status_change_callbacks:
                    try:
                        callback(status)
                    except Exception as e:
                        logger.error(f"Status callback failed: {e}")
            
            # Schedule next cycle
            timer = Timer(self.coordination_interval, self._coordination_cycle)
            timer.daemon = True
            timer.start()
            
        except Exception as e:
            logger.error(f"Coordination cycle failed: {e}")
            # Restart after error
            timer = Timer(self.coordination_interval * 2, self._coordination_cycle)
            timer.daemon = True
            timer.start()
    
    def _calculate_system_status(self) -> SystemStatus:
        """Calculate current system status"""
        # Get metrics from all components
        dashboard_data = self.metrics_system.get_real_time_dashboard()
        config_report = self.config_optimizer.get_configuration_report()
        enabler_stats = self.progressive_enabler.get_enablement_statistics()
        
        # Calculate health score
        current_metrics = dashboard_data.get("current_metrics", {})
        health_score = current_metrics.get("system_health_score", 50.0)
        
        # Calculate optimization score
        config_health = config_report.get("config_health_score", 50.0)
        enabler_efficiency = enabler_stats.get("overall_efficiency", 0.5) * 100
        optimization_score = (config_health + enabler_efficiency) / 2
        
        # Determine performance trend
        trends = dashboard_data.get("trends", {})
        performance_trend = self._analyze_overall_trend(trends)
        
        # Get alerts
        active_alerts = dashboard_data.get("active_alerts", [])
        
        # Calculate uptime
        uptime_hours = (time.time() - self.start_time) / 3600
        
        # Last optimization time
        optimization_history = config_report.get("optimization_history", [])
        last_optimization = None
        if optimization_history:
            last_opt_time = optimization_history[-1].get("timestamp", 0)
            last_optimization = datetime.fromtimestamp(last_opt_time).isoformat()
        
        return SystemStatus(
            mode=self.system_mode,
            health_score=health_score,
            active_sessions=current_metrics.get("active_sessions", 0),
            optimization_score=optimization_score,
            last_optimization=last_optimization,
            alerts_count=len(active_alerts),
            uptime_hours=uptime_hours,
            performance_trend=performance_trend
        )
    
    def _analyze_overall_trend(self, trends: Dict[str, Any]) -> str:
        """Analyze overall performance trend"""
        if not trends:
            return "stable"
        
        improving_count = 0
        degrading_count = 0
        
        for trend_data in trends.values():
            direction = trend_data.get("trend_direction", "stable")
            strength = trend_data.get("trend_strength", 0.0)
            
            if strength > 0.3:  # Only consider significant trends
                if direction == "improving":
                    improving_count += 1
                elif direction == "degrading":
                    degrading_count += 1
        
        if improving_count > degrading_count:
            return "improving"
        elif degrading_count > improving_count:
            return "degrading"
        else:
            return "stable"
    
    def _determine_optimal_mode(self, status: SystemStatus) -> SystemMode:
        """Determine optimal system mode based on current status"""
        # Emergency mode conditions
        if status.health_score < 30 or status.alerts_count >= 3:
            return SystemMode.EMERGENCY
        
        # Learning mode conditions
        if status.uptime_hours < self.learning_period_hours:
            return SystemMode.LEARNING
        
        # Stable mode conditions
        if (status.health_score >= 80 and 
            status.optimization_score >= 75 and 
            status.performance_trend in ["stable", "improving"]):
            return SystemMode.STABLE
        
        # Default to optimizing mode
        return SystemMode.OPTIMIZING
    
    def _change_system_mode(self, new_mode: SystemMode, status: SystemStatus):
        """Change system mode and adjust settings accordingly"""
        old_mode = self.system_mode
        self.system_mode = new_mode
        
        logger.info(f"System mode changed: {old_mode.value} -> {new_mode.value}")
        
        # Adjust settings based on new mode
        if new_mode == SystemMode.LEARNING:
            # Conservative settings during learning
            self.config_optimizer.disable_auto_tuning()
            self.progressive_enabler.set_learning_mode(True)
            
        elif new_mode == SystemMode.OPTIMIZING:
            # Enable optimizations
            if self.auto_optimization_enabled:
                self.config_optimizer.enable_auto_tuning(ConfigOptimizationLevel.BALANCED)
            self.progressive_enabler.set_learning_mode(False)
            
        elif new_mode == SystemMode.STABLE:
            # Minimal changes in stable mode
            self.config_optimizer.enable_auto_tuning(ConfigOptimizationLevel.CONSERVATIVE)
            
        elif new_mode == SystemMode.EMERGENCY:
            # Safe fallbacks
            self.config_optimizer.disable_auto_tuning()
            self.progressive_enabler.set_emergency_mode(True)
    
    def _execute_mode_actions(self, status: SystemStatus):
        """Execute actions specific to current mode"""
        if self.system_mode == SystemMode.LEARNING:
            # Focus on data collection
            self._ensure_data_collection()
            
        elif self.system_mode == SystemMode.OPTIMIZING:
            # Perform optimizations if conditions are met
            self._perform_optimizations(status)
            
        elif self.system_mode == SystemMode.STABLE:
            # Minimal maintenance
            self._perform_maintenance()
            
        elif self.system_mode == SystemMode.EMERGENCY:
            # Emergency actions
            self._handle_emergency_mode(status)
    
    def _ensure_data_collection(self):
        """Ensure proper data collection during learning mode"""
        # Verify monitoring is active
        if not self.metrics_system.monitoring_enabled:
            logger.warning("Monitoring disabled during learning mode, re-enabling")
            self.metrics_system.monitoring_enabled = True
            self.metrics_system._start_monitoring()
    
    def _perform_optimizations(self, status: SystemStatus):
        """Perform system optimizations"""
        # Check cooldown period
        if time.time() - self.last_optimization_time < self.optimization_cooldown:
            return
        
        # Get current metrics for optimization
        dashboard_data = self.metrics_system.get_real_time_dashboard()
        current_metrics = dashboard_data.get("current_metrics", {})
        
        # Perform configuration optimization
        if self.auto_optimization_enabled:
            optimization_result = self.config_optimizer.auto_tune_configuration(current_metrics)
            
            if optimization_result.get("auto_tuning_applied"):
                self.last_optimization_time = time.time()
                
                # Notify optimization callbacks
                for callback in self.optimization_callbacks:
                    try:
                        callback(optimization_result)
                    except Exception as e:
                        logger.error(f"Optimization callback failed: {e}")
                
                logger.info(f"Auto-optimization applied: {optimization_result.get('applied_count', 0)} changes")
        
        # Optimize progressive enabler thresholds
        self.progressive_enabler.optimize_thresholds()
    
    def _perform_maintenance(self):
        """Perform routine maintenance in stable mode"""
        # Light maintenance tasks
        pass
    
    def _handle_emergency_mode(self, status: SystemStatus):
        """Handle emergency mode actions"""
        logger.warning(f"System in emergency mode: health={status.health_score:.1f}, alerts={status.alerts_count}")
        
        # Disable risky features
        self.config_optimizer.disable_auto_tuning()
        
        # Reset to safe defaults if health is very low
        if status.health_score < 20:
            logger.critical("Resetting to safe defaults due to critical health score")
            self.config_optimizer.reset_to_defaults()
    
    def _handle_system_alert(self, alert: SystemAlert):
        """Handle system alerts"""
        logger.warning(f"System alert: {alert.title} ({alert.level.value})")
        
        # Take action based on alert level
        if alert.level == AlertLevel.CRITICAL:
            # Force emergency mode for critical alerts
            with self._lock:
                if self.system_mode != SystemMode.EMERGENCY:
                    self._change_system_mode(SystemMode.EMERGENCY, self._calculate_system_status())
        
        elif alert.level == AlertLevel.WARNING:
            # Consider switching to optimizing mode
            with self._lock:
                if self.system_mode == SystemMode.STABLE:
                    self._change_system_mode(SystemMode.OPTIMIZING, self._calculate_system_status())
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        with self._lock:
            return self._calculate_system_status()
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive system report"""
        status = self.get_system_status()
        dashboard_data = self.metrics_system.get_real_time_dashboard()
        predictive_insights = self.metrics_system.get_predictive_insights()
        config_report = self.config_optimizer.get_configuration_report()
        enabler_stats = self.progressive_enabler.get_enablement_statistics()
        
        return {
            "system_status": status.to_dict(),
            "dashboard_data": dashboard_data,
            "predictive_insights": predictive_insights,
            "configuration_report": config_report,
            "enablement_statistics": enabler_stats,
            "coordination_settings": {
                "auto_optimization_enabled": self.auto_optimization_enabled,
                "coordination_interval": self.coordination_interval,
                "learning_period_hours": self.learning_period_hours,
                "optimization_cooldown": self.optimization_cooldown
            },
            "report_timestamp": datetime.now().isoformat()
        }
    
    def enable_auto_optimization(self):
        """Enable automatic optimization"""
        self.auto_optimization_enabled = True
        logger.info("Auto-optimization enabled")
    
    def disable_auto_optimization(self):
        """Disable automatic optimization"""
        self.auto_optimization_enabled = False
        self.config_optimizer.disable_auto_tuning()
        logger.info("Auto-optimization disabled")
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization regardless of cooldown"""
        dashboard_data = self.metrics_system.get_real_time_dashboard()
        current_metrics = dashboard_data.get("current_metrics", {})
        
        # Force configuration optimization
        optimization_result = self.config_optimizer.auto_tune_configuration(current_metrics)
        
        # Force progressive enabler optimization
        self.progressive_enabler.optimize_thresholds()
        
        self.last_optimization_time = time.time()
        
        logger.info("Forced optimization completed")
        return optimization_result
    
    def add_status_change_callback(self, callback: Callable[[SystemStatus], None]):
        """Add callback for system status changes"""
        self.status_change_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for optimization events"""
        self.optimization_callbacks.append(callback)
    
    def export_system_backup(self, backup_file: Optional[str] = None) -> str:
        """Export complete system backup"""
        backup_file = backup_file or f"phase4_system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Get comprehensive report
        report = self.get_comprehensive_report()
        
        # Add backup metadata
        report["backup_metadata"] = {
            "backup_type": "phase4_complete_system",
            "backup_timestamp": datetime.now().isoformat(),
            "system_uptime_hours": (time.time() - self.start_time) / 3600
        }
        
        # Export individual component backups
        config_backup = self.config_optimizer.export_config_backup(
            f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        metrics_backup = self.metrics_system.export_advanced_report(
            f"metrics_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        report["component_backups"] = {
            "config_backup_file": config_backup,
            "metrics_backup_file": metrics_backup
        }
        
        # Save main backup
        import json
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Complete system backup exported to {backup_file}")
        return backup_file
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down ResearcherPhase4System")
        
        # Stop monitoring
        self.metrics_system.stop_monitoring()
        
        # Export final backup
        self.export_system_backup(f"shutdown_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        logger.info("ResearcherPhase4System shutdown complete")


# Global Phase 4 system instance
_global_phase4_system: Optional[ResearcherPhase4System] = None


def get_phase4_system(enable_auto_optimization: bool = False) -> ResearcherPhase4System:
    """Get global Phase 4 system instance"""
    global _global_phase4_system
    if _global_phase4_system is None:
        _global_phase4_system = ResearcherPhase4System(enable_auto_optimization)
    return _global_phase4_system


def reset_phase4_system():
    """Reset global Phase 4 system (for testing)"""
    global _global_phase4_system
    if _global_phase4_system:
        _global_phase4_system.shutdown()
    _global_phase4_system = None


# Convenience functions for backward compatibility
def initialize_phase4_system(enable_auto_optimization: bool = False) -> ResearcherPhase4System:
    """Initialize Phase 4 system with specified settings"""
    system = get_phase4_system(enable_auto_optimization)
    logger.info(f"Phase 4 system initialized (auto_optimization: {enable_auto_optimization})")
    return system


def get_phase4_status() -> Dict[str, Any]:
    """Get Phase 4 system status"""
    system = get_phase4_system()
    return system.get_system_status().to_dict()


def get_phase4_report() -> Dict[str, Any]:
    """Get comprehensive Phase 4 report"""
    system = get_phase4_system()
    return system.get_comprehensive_report()