# -*- coding: utf-8 -*-
"""
Health check system
Provides system status monitoring, service availability checking and health metrics collection
"""

import asyncio
import time
import psutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from concurrent.futures import ThreadPoolExecutor
import json

from ..common.structured_logging import get_logger, EventType

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Check type"""

    LIVENESS = "liveness"  # Liveness check
    READINESS = "readiness"  # Readiness check
    STARTUP = "startup"  # Startup check
    DEPENDENCY = "dependency"  # Dependency check
    RESOURCE = "resource"  # Resource check
    CUSTOM = "custom"  # Custom check


@dataclass
class HealthCheckResult:
    """Health check result"""

    name: str
    status: HealthStatus
    check_type: CheckType
    message: str
    duration_ms: float
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["check_type"] = self.check_type.value
        return result


@dataclass
class SystemMetrics:
    """System metrics"""

    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_percent: float
    disk_free_gb: float
    network_connections: int
    process_count: int
    uptime_seconds: float
    load_average: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    """Health report"""

    overall_status: HealthStatus
    timestamp: str
    uptime_seconds: float
    checks: List[HealthCheckResult]
    metrics: SystemMetrics
    summary: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "checks": [check.to_dict() for check in self.checks],
            "metrics": self.metrics.to_dict(),
            "summary": self.summary,
        }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class HealthCheck(ABC):
    """Health check base class"""

    def __init__(self, name: str, check_type: CheckType, timeout_seconds: float = 5.0):
        self.name = name
        self.check_type = check_type
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Execute health check"""
        pass

    async def run_check(self) -> HealthCheckResult:
        """Run health check (with timeout)"""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(self.check(), timeout=self.timeout_seconds)
            result.duration_ms = (time.time() - start_time) * 1000
            result.timestamp = datetime.utcnow().isoformat() + "Z"
            return result

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                message=f"Health check timed out after {self.timeout_seconds}s",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow().isoformat() + "Z",
                error="timeout",
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow().isoformat() + "Z",
                error=str(e),
            )


class DatabaseHealthCheck(HealthCheck):
    """Database health check"""

    def __init__(
        self, name: str, connection_factory: Callable, timeout_seconds: float = 5.0
    ):
        super().__init__(name, CheckType.DEPENDENCY, timeout_seconds)
        self.connection_factory = connection_factory

    async def check(self) -> HealthCheckResult:
        try:
            # Try to get database connection and execute simple query
            conn = await self.connection_factory()

            # Execute simple health check query
            if hasattr(conn, "execute"):
                await conn.execute("SELECT 1")
            elif hasattr(conn, "ping"):
                await conn.ping()

            if hasattr(conn, "close"):
                await conn.close()

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                check_type=self.check_type,
                message="Database connection successful",
                duration_ms=0,  # Will be set in run_check
                timestamp="",  # Will be set in run_check
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                message=f"Database connection failed: {str(e)}",
                duration_ms=0,
                timestamp="",
                error=str(e),
            )


class LLMServiceHealthCheck(HealthCheck):
    """LLM service health check"""

    def __init__(self, name: str, llm_service: Any, timeout_seconds: float = 10.0):
        super().__init__(name, CheckType.DEPENDENCY, timeout_seconds)
        self.llm_service = llm_service

    async def check(self) -> HealthCheckResult:
        try:
            # Send simple test request
            test_messages = [{"role": "user", "content": "ping"}]

            # Use safe LLM call functions to ensure proper context management
            from src.llms.error_handler import safe_llm_call_async, safe_llm_call

            if hasattr(self.llm_service, "ainvoke"):
                response = await safe_llm_call_async(
                    self.llm_service.ainvoke,
                    test_messages,
                    operation_name="Health Check",
                    context="LLM service health check",
                )
            elif hasattr(self.llm_service, "invoke"):
                response = safe_llm_call(
                    self.llm_service.invoke,
                    test_messages,
                    operation_name="Health Check",
                    context="LLM service health check",
                )
            else:
                raise ValueError("LLM service does not have invoke method")

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                check_type=self.check_type,
                message="LLM service is responding",
                duration_ms=0,
                timestamp="",
                details={"response_length": len(str(response)) if response else 0},
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                message=f"LLM service check failed: {str(e)}",
                duration_ms=0,
                timestamp="",
                error=str(e),
            )


class ResourceHealthCheck(HealthCheck):
    """Resource health check"""

    def __init__(
        self,
        name: str = "system_resources",
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 90.0,
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, CheckType.RESOURCE, timeout_seconds)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def check(self) -> HealthCheckResult:
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            issues = []
            status = HealthStatus.HEALTHY

            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = HealthStatus.DEGRADED

            if memory.percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                status = HealthStatus.DEGRADED

            if disk.percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
                status = HealthStatus.UNHEALTHY

            message = (
                "System resources are healthy" if not issues else "; ".join(issues)
            )

            return HealthCheckResult(
                name=self.name,
                status=status,
                check_type=self.check_type,
                message=message,
                duration_ms=0,
                timestamp="",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                message=f"Resource check failed: {str(e)}",
                duration_ms=0,
                timestamp="",
                error=str(e),
            )


class CustomHealthCheck(HealthCheck):
    """Custom health check"""

    def __init__(
        self,
        name: str,
        check_function: Callable[[], Union[bool, HealthCheckResult]],
        check_type: CheckType = CheckType.CUSTOM,
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, check_type, timeout_seconds)
        self.check_function = check_function

    async def check(self) -> HealthCheckResult:
        try:
            if asyncio.iscoroutinefunction(self.check_function):
                result = await self.check_function()
            else:
                result = self.check_function()

            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Custom check passed" if result else "Custom check failed"
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    check_type=self.check_type,
                    message=message,
                    duration_ms=0,
                    timestamp="",
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    check_type=self.check_type,
                    message=f"Invalid check result type: {type(result)}",
                    duration_ms=0,
                    timestamp="",
                    error="invalid_result_type",
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                message=f"Custom check failed: {str(e)}",
                duration_ms=0,
                timestamp="",
                error=str(e),
            )


class HealthCheckManager:
    """Health check manager"""

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.start_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._background_task: Optional[asyncio.Task] = None
        self._check_interval = 30  # seconds
        self._last_report: Optional[HealthReport] = None
        self._running = False

    def register_check(self, check: HealthCheck):
        """Register health check"""
        self.checks[check.name] = check
        logger.info(
            f"Registered health check: {check.name}",
            event_type=EventType.SYSTEM,
            data={"check_type": check.check_type.value},
        )

    def unregister_check(self, name: str):
        """Unregister health check"""
        if name in self.checks:
            del self.checks[name]
            logger.info(
                f"Unregistered health check: {name}", event_type=EventType.SYSTEM
            )

    def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = len(psutil.net_connections())
            process_count = len(psutil.pids())
            uptime = time.time() - self.start_time

            load_avg = None
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have getloadavg
                pass

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024**2),
                disk_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_connections=network,
                process_count=process_count,
                uptime_seconds=uptime,
                load_average=load_avg,
            )

        except Exception as e:
            logger.error(
                f"Failed to get system metrics: {e}",
                error=e,
                event_type=EventType.ERROR,
            )
            # Return default values
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_percent=0.0,
                disk_free_gb=0.0,
                network_connections=0,
                process_count=0,
                uptime_seconds=time.time() - self.start_time,
            )

    async def run_all_checks(self) -> HealthReport:
        """Run all health checks"""
        start_time = time.time()

        # Run all checks in parallel
        tasks = [check.run_check() for check in self.checks.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        check_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_name = list(self.checks.keys())[i]
                check_results.append(
                    HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        check_type=CheckType.CUSTOM,
                        message=f"Check execution failed: {str(result)}",
                        duration_ms=0,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        error=str(result),
                    )
                )
            else:
                check_results.append(result)

        # Calculate overall status
        overall_status = self._calculate_overall_status(check_results)

        # Get system metrics
        metrics = self.get_system_metrics()

        # Generate summary
        summary = {
            "total": len(check_results),
            "healthy": len(
                [r for r in check_results if r.status == HealthStatus.HEALTHY]
            ),
            "degraded": len(
                [r for r in check_results if r.status == HealthStatus.DEGRADED]
            ),
            "unhealthy": len(
                [r for r in check_results if r.status == HealthStatus.UNHEALTHY]
            ),
            "unknown": len(
                [r for r in check_results if r.status == HealthStatus.UNKNOWN]
            ),
        }

        report = HealthReport(
            overall_status=overall_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            uptime_seconds=time.time() - self.start_time,
            checks=check_results,
            metrics=metrics,
            summary=summary,
        )

        self._last_report = report

        # Log health check results
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Health check completed in {duration_ms:.2f}ms",
            event_type=EventType.SYSTEM,
            data={
                "overall_status": overall_status.value,
                "duration_ms": duration_ms,
                "summary": summary,
            },
        )

        return report

    def _calculate_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Calculate overall health status"""
        if not results:
            return HealthStatus.UNKNOWN

        # If any check fails, overall status is unhealthy
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.UNHEALTHY

        # If any check is degraded, overall status is degraded
        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED

        # If any status is unknown, overall status is unknown
        if any(r.status == HealthStatus.UNKNOWN for r in results):
            return HealthStatus.UNKNOWN

        # All checks are healthy
        return HealthStatus.HEALTHY

    async def get_health_report(self) -> HealthReport:
        """Get health report"""
        return await self.run_all_checks()

    def get_last_report(self) -> Optional[HealthReport]:
        """Get last health report"""
        return self._last_report

    def start_background_checks(self, interval_seconds: int = 30):
        """Start background health checks"""
        self._check_interval = interval_seconds
        self._running = True

        async def background_check_loop():
            while self._running:
                try:
                    await self.run_all_checks()
                except Exception as e:
                    logger.error(
                        f"Background health check failed: {e}",
                        error=e,
                        event_type=EventType.ERROR,
                    )

                await asyncio.sleep(self._check_interval)

        self._background_task = asyncio.create_task(background_check_loop())
        logger.info(
            f"Started background health checks with {interval_seconds}s interval",
            event_type=EventType.SYSTEM,
        )

    def stop_background_checks(self):
        """Stop background health checks"""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            self._background_task = None

        logger.info("Stopped background health checks", event_type=EventType.SYSTEM)

    def __del__(self):
        """Destructor"""
        self.stop_background_checks()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


# Global health check manager
global_health_manager = HealthCheckManager()


def register_health_check(check: HealthCheck):
    """Register global health check"""
    global_health_manager.register_check(check)


def get_health_manager() -> HealthCheckManager:
    """Get global health check manager"""
    return global_health_manager


async def get_health_status() -> HealthReport:
    """Get system health status"""
    return await global_health_manager.get_health_report()


def setup_default_health_checks():
    """Setup default health checks"""
    # Register resource health check
    resource_check = ResourceHealthCheck()
    register_health_check(resource_check)

    # Can add more default checks as needed
    logger.info("Default health checks configured", event_type=EventType.SYSTEM)
