# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


class SandboxInfo(BaseModel):
    id: str
    status: str
    created_at: datetime
    image: str
    cpu_limit: float
    memory_limit: str
    network_enabled: bool
    work_dir: str


class SandboxMetrics(BaseModel):
    cpu_usage: float = Field(default=0.0, description="CPU usage percentage")
    memory_usage: float = Field(default=0.0, description="Memory usage percentage")
    memory_used: str = Field(default="0B", description="Memory used")
    memory_total: str = Field(default="0B", description="Total memory")
    disk_usage: float = Field(default=0.0, description="Disk usage percentage")
    disk_used: str = Field(default="0B", description="Disk used")
    disk_total: str = Field(default="0B", description="Total disk")
    network_rx: str = Field(default="0B", description="Network received")
    network_tx: str = Field(default="0B", description="Network transmitted")
    uptime: int = Field(default=0, description="Uptime in seconds")
    process_count: int = Field(default=0, description="Number of processes")


class SandboxStatus(BaseModel):
    info: SandboxInfo
    metrics: Optional[SandboxMetrics] = None
    last_activity: Optional[datetime] = None
    logs: List[str] = Field(default_factory=list, description="Recent logs")


class SandboxCreateRequest(BaseModel):
    image: str = Field(default="python:3.12-slim", description="Docker image")
    memory_limit: str = Field(default="512m", description="Memory limit")
    cpu_limit: float = Field(default=1.0, description="CPU limit")
    network_enabled: bool = Field(default=False, description="Network access")
    timeout: int = Field(default=300, description="Default timeout in seconds")


class CommandRequest(BaseModel):
    command: str = Field(..., description="Command to execute")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")


class CommandResponse(BaseModel):
    output: str
    exit_code: int
    duration: float


class FileInfo(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int
    modified: Optional[str] = None
    permissions: Optional[str] = None


class FileContent(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"


_sandbox_registry: Dict[str, Dict[str, Any]] = {}


async def get_sandbox_manager():
    from src.sandbox.core.manager import get_sandbox_manager

    return get_sandbox_manager()


@router.get("")
async def list_sandboxes():
    """List all sandboxes."""
    manager = await get_sandbox_manager()
    stats = manager.get_stats()

    sandboxes = []
    for sandbox_id, sandbox_data in _sandbox_registry.items():
        sandboxes.append(
            {
                "id": sandbox_id,
                "status": sandbox_data.get("status", "unknown"),
                "created_at": sandbox_data.get("created_at"),
                "image": sandbox_data.get("image", "unknown"),
            }
        )

    return {
        "sandboxes": sandboxes,
        "total": len(sandboxes),
        "manager_stats": stats,
    }


@router.post("", response_model=SandboxInfo)
async def create_sandbox(request: SandboxCreateRequest):
    """Create a new sandbox."""
    from src.sandbox.core.sandbox import SandboxSettings

    manager = await get_sandbox_manager()

    config = SandboxSettings(
        image=request.image,
        memory_limit=request.memory_limit,
        cpu_limit=request.cpu_limit,
        network_enabled=request.network_enabled,
        timeout=request.timeout,
    )

    try:
        sandbox_id = await manager.create_sandbox(config)

        _sandbox_registry[sandbox_id] = {
            "status": "running",
            "created_at": datetime.utcnow(),
            "image": request.image,
            "config": config.model_dump(),
        }

        return SandboxInfo(
            id=sandbox_id,
            status="running",
            created_at=datetime.utcnow(),
            image=request.image,
            cpu_limit=request.cpu_limit,
            memory_limit=request.memory_limit,
            network_enabled=request.network_enabled,
            work_dir="/workspace",
        )
    except Exception as e:
        logger.error(f"Failed to create sandbox: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create sandbox: {e}")


@router.get("/{sandbox_id}", response_model=SandboxStatus)
async def get_sandbox_status(sandbox_id: str):
    """Get sandbox status and metrics."""
    manager = await get_sandbox_manager()

    try:
        sandbox = await manager.get_sandbox(sandbox_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    registry_data = _sandbox_registry.get(sandbox_id, {})

    metrics = await _collect_sandbox_metrics(sandbox)

    info = SandboxInfo(
        id=sandbox_id,
        status=registry_data.get("status", "running"),
        created_at=registry_data.get("created_at", datetime.utcnow()),
        image=registry_data.get("image", "unknown"),
        cpu_limit=registry_data.get("config", {}).get("cpu_limit", 1.0),
        memory_limit=registry_data.get("config", {}).get("memory_limit", "512m"),
        network_enabled=registry_data.get("config", {}).get("network_enabled", False),
        work_dir=sandbox.config.work_dir if sandbox else "/workspace",
    )

    return SandboxStatus(
        info=info,
        metrics=metrics,
        last_activity=registry_data.get("last_activity"),
        logs=registry_data.get("logs", [])[-100:],
    )


@router.delete("/{sandbox_id}")
async def delete_sandbox(sandbox_id: str):
    """Delete a sandbox."""
    manager = await get_sandbox_manager()

    try:
        await manager.delete_sandbox(sandbox_id)
        _sandbox_registry.pop(sandbox_id, None)
        return {"status": "deleted", "sandbox_id": sandbox_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    except Exception as e:
        logger.error(f"Failed to delete sandbox: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete sandbox: {e}")


@router.post("/{sandbox_id}/execute", response_model=CommandResponse)
async def execute_command(sandbox_id: str, request: CommandRequest):
    """Execute a command in the sandbox."""
    manager = await get_sandbox_manager()

    try:
        sandbox = await manager.get_sandbox(sandbox_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    import time

    start_time = time.time()
    try:
        output = await sandbox.run_command(request.command, request.timeout)
        exit_code = 0
    except Exception as e:
        output = str(e)
        exit_code = 1

    duration = time.time() - start_time

    if sandbox_id in _sandbox_registry:
        _sandbox_registry[sandbox_id]["last_activity"] = datetime.utcnow()
        if "logs" not in _sandbox_registry[sandbox_id]:
            _sandbox_registry[sandbox_id]["logs"] = []
        _sandbox_registry[sandbox_id]["logs"].append(
            f"[{datetime.utcnow().isoformat()}] $ {request.command}"
        )
        _sandbox_registry[sandbox_id]["logs"].extend(output.split("\n")[:10])

    return CommandResponse(output=output, exit_code=exit_code, duration=duration)


@router.get("/{sandbox_id}/files", response_model=List[FileInfo])
async def list_files(sandbox_id: str, path: str = "/workspace"):
    """List files in a sandbox directory."""
    manager = await get_sandbox_manager()

    try:
        sandbox = await manager.get_sandbox(sandbox_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    try:
        output = await sandbox.run_command(f"ls -la {path} 2>/dev/null || echo 'ERROR'")
        if "ERROR" in output:
            return []

        files = []
        for line in output.strip().split("\n")[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 9:
                is_dir = parts[0].startswith("d")
                files.append(
                    FileInfo(
                        name=parts[-1],
                        path=(
                            f"{path}/{parts[-1]}"
                            if not is_dir
                            else f"{path}/{parts[-1]}/"
                        ),
                        is_dir=is_dir,
                        size=int(parts[4]) if not is_dir else 0,
                        modified=parts[5] + " " + parts[6] + " " + parts[7],
                        permissions=parts[0],
                    )
                )
        return files
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")


@router.get("/{sandbox_id}/files/{file_path:path}", response_model=FileContent)
async def read_file(sandbox_id: str, file_path: str):
    """Read a file from the sandbox."""
    manager = await get_sandbox_manager()

    try:
        sandbox = await manager.get_sandbox(sandbox_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    try:
        output = await sandbox.run_command(
            f"cat {file_path} 2>/dev/null || echo 'ERROR'"
        )
        if "ERROR" in output:
            raise HTTPException(status_code=404, detail="File not found")

        return FileContent(path=file_path, content=output)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


@router.websocket("/{sandbox_id}/terminal")
async def sandbox_terminal_stream(websocket: WebSocket, sandbox_id: str):
    """WebSocket endpoint for terminal streaming."""
    await websocket.accept()

    manager = await get_sandbox_manager()

    try:
        sandbox = await manager.get_sandbox(sandbox_id)
    except KeyError:
        await websocket.send_json({"type": "error", "message": "Sandbox not found"})
        await websocket.close()
        return

    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
            if data.get("type") == "command":
                command = data.get("command", "")
                try:
                    output = await sandbox.run_command(command, timeout=30)
                    await websocket.send_json(
                        {"type": "output", "output": output, "command": command}
                    )
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "message": str(e), "command": command}
                    )
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for sandbox {sandbox_id}")
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@router.get("/{sandbox_id}/metrics", response_model=SandboxMetrics)
async def get_sandbox_metrics(sandbox_id: str):
    """Get sandbox resource metrics."""
    manager = await get_sandbox_manager()

    try:
        await manager.get_sandbox(sandbox_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Sandbox not found")

    return await _collect_sandbox_metrics_by_id(sandbox_id)


async def _collect_sandbox_metrics(sandbox) -> SandboxMetrics:
    """Collect metrics from a sandbox."""
    try:
        if hasattr(sandbox, "container") and sandbox.container:
            stats = sandbox.container.stats(stream=False)

            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )
            cpu_usage = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0

            memory_usage = stats["memory_stats"].get("usage", 0)
            memory_limit = stats["memory_stats"].get("limit", 1)
            memory_percent = (
                (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            )

            return SandboxMetrics(
                cpu_usage=round(cpu_usage, 2),
                memory_usage=round(memory_percent, 2),
                memory_used=_format_bytes(memory_usage),
                memory_total=_format_bytes(memory_limit),
                uptime=0,
                process_count=0,
            )
    except Exception as e:
        logger.debug(f"Failed to collect metrics: {e}")

    return SandboxMetrics()


async def _collect_sandbox_metrics_by_id(sandbox_id: str) -> SandboxMetrics:
    """Collect metrics from a sandbox by ID."""
    manager = await get_sandbox_manager()
    try:
        sandbox = await manager.get_sandbox(sandbox_id)
        return await _collect_sandbox_metrics(sandbox)
    except Exception:
        return SandboxMetrics()


def _format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"
