import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Set

from src.sandbox.core.sandbox import DockerSandbox, SandboxSettings


class SandboxManager:
    def __init__(
        self,
        max_sandboxes: int = 100,
        idle_timeout: int = 3600,
        cleanup_interval: int = 300,
        auto_cleanup: bool = True,
    ):
        self.max_sandboxes = max_sandboxes
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval
        self.auto_cleanup = auto_cleanup

        self._client: Any = None
        self._sandboxes: dict[str, DockerSandbox] = {}
        self._last_used: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._active_operations: Set[str] = set()
        self._cleanup_task: asyncio.Task | None = None
        self._is_shutting_down = False

        if auto_cleanup:
            self._try_start_cleanup_task()

    def _try_start_cleanup_task(self) -> None:
        try:
            asyncio.get_running_loop()
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            pass

    def _get_client(self):
        if self._client is None:
            import docker

            self._client = docker.from_env()
        return self._client

    async def ensure_image(self, image: str) -> bool:
        try:
            client = self._get_client()
            client.images.get(image)
            return True
        except Exception:
            try:
                client = self._get_client()
                await asyncio.get_event_loop().run_in_executor(
                    None, client.images.pull, image
                )
                return True
            except Exception:
                return False

    @asynccontextmanager
    async def sandbox_operation(self, sandbox_id: str):
        if sandbox_id not in self._locks:
            self._locks[sandbox_id] = asyncio.Lock()

        async with self._locks[sandbox_id]:
            if sandbox_id not in self._sandboxes:
                raise KeyError(f"Sandbox {sandbox_id} not found")

            self._active_operations.add(sandbox_id)
            try:
                self._last_used[sandbox_id] = asyncio.get_event_loop().time()
                yield self._sandboxes[sandbox_id]
            finally:
                self._active_operations.remove(sandbox_id)

    async def create_sandbox(
        self,
        config: SandboxSettings | None = None,
        volume_bindings: dict[str, str] | None = None,
    ) -> str:
        async with self._global_lock:
            if len(self._sandboxes) >= self.max_sandboxes:
                raise RuntimeError(
                    f"Maximum number of sandboxes ({self.max_sandboxes}) reached"
                )

            config = config or SandboxSettings()
            if not await self.ensure_image(config.image):
                raise RuntimeError(f"Failed to ensure Docker image: {config.image}")

            sandbox_id = str(uuid.uuid4())
            try:
                sandbox = DockerSandbox(config, volume_bindings)
                await sandbox.create()

                self._sandboxes[sandbox_id] = sandbox
                self._last_used[sandbox_id] = asyncio.get_event_loop().time()
                self._locks[sandbox_id] = asyncio.Lock()

                return sandbox_id

            except Exception as e:
                if sandbox_id in self._sandboxes:
                    await self.delete_sandbox(sandbox_id)
                raise RuntimeError(f"Failed to create sandbox: {e}")

    async def get_sandbox(self, sandbox_id: str) -> DockerSandbox:
        async with self.sandbox_operation(sandbox_id) as sandbox:
            return sandbox

    async def _cleanup_loop(self):
        while not self._is_shutting_down:
            try:
                await self._cleanup_idle_sandboxes()
            except Exception:
                pass
            await asyncio.sleep(self.cleanup_interval)

    def start_cleanup_task(self) -> None:
        self._try_start_cleanup_task()

    async def _cleanup_idle_sandboxes(self) -> None:
        current_time = asyncio.get_event_loop().time()
        to_cleanup = []

        async with self._global_lock:
            for sandbox_id, last_used in self._last_used.items():
                if (
                    sandbox_id not in self._active_operations
                    and current_time - last_used > self.idle_timeout
                ):
                    to_cleanup.append(sandbox_id)

        for sandbox_id in to_cleanup:
            try:
                await self.delete_sandbox(sandbox_id)
            except Exception:
                pass

    async def cleanup(self) -> None:
        self._is_shutting_down = True

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        async with self._global_lock:
            sandbox_ids = list(self._sandboxes.keys())

        cleanup_tasks = []
        for sandbox_id in sandbox_ids:
            task = asyncio.create_task(self._safe_delete_sandbox(sandbox_id))
            cleanup_tasks.append(task)

        if cleanup_tasks:
            try:
                await asyncio.wait(cleanup_tasks, timeout=30.0)
            except asyncio.TimeoutError:
                pass

        self._sandboxes.clear()
        self._last_used.clear()
        self._locks.clear()
        self._active_operations.clear()

    async def _safe_delete_sandbox(self, sandbox_id: str) -> None:
        try:
            if sandbox_id in self._active_operations:
                for _ in range(10):
                    await asyncio.sleep(0.5)
                    if sandbox_id not in self._active_operations:
                        break

            sandbox = self._sandboxes.get(sandbox_id)
            if sandbox:
                await sandbox.cleanup()

                async with self._global_lock:
                    self._sandboxes.pop(sandbox_id, None)
                    self._last_used.pop(sandbox_id, None)
                    self._locks.pop(sandbox_id, None)
        except Exception:
            pass

    async def delete_sandbox(self, sandbox_id: str) -> None:
        if sandbox_id not in self._sandboxes:
            return

        try:
            await self._safe_delete_sandbox(sandbox_id)
        except Exception:
            pass

    async def __aenter__(self) -> "SandboxManager":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()

    def get_stats(self) -> dict:
        return {
            "total_sandboxes": len(self._sandboxes),
            "active_operations": len(self._active_operations),
            "max_sandboxes": self.max_sandboxes,
            "idle_timeout": self.idle_timeout,
            "cleanup_interval": self.cleanup_interval,
            "is_shutting_down": self._is_shutting_down,
        }


_sandbox_manager: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager()
    return _sandbox_manager
