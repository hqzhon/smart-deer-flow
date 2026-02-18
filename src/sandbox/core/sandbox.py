import asyncio
import io
import os
import tarfile
import tempfile
import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.sandbox.core.exceptions import SandboxTimeoutError
from src.sandbox.core.terminal import AsyncDockerizedTerminal


class SandboxSettings(BaseModel):
    image: str = Field(default="python:3.12-slim", description="Docker image")
    work_dir: str = Field(default="/workspace", description="Working directory")
    memory_limit: str = Field(default="512m", description="Memory limit")
    cpu_limit: float = Field(default=1.0, description="CPU limit")
    timeout: int = Field(default=300, description="Default timeout in seconds")
    network_enabled: bool = Field(default=False, description="Network access")


class DockerSandbox:
    def __init__(
        self,
        config: SandboxSettings | None = None,
        volume_bindings: dict[str, str] | None = None,
    ):
        self.config = config or SandboxSettings()
        self.volume_bindings = volume_bindings or {}
        self.client: Any = None
        self.container: Any = None
        self.terminal: AsyncDockerizedTerminal | None = None

    async def create(self) -> "DockerSandbox":
        import docker

        self.client = docker.from_env()

        try:
            host_config = self.client.api.create_host_config(
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(100000 * self.config.cpu_limit),
                network_mode="none" if not self.config.network_enabled else "bridge",
                binds=self._prepare_volume_bindings(),
            )

            container_name = f"sandbox_{uuid.uuid4().hex[:8]}"

            container = await asyncio.to_thread(
                self.client.api.create_container,
                image=self.config.image,
                command="tail -f /dev/null",
                hostname="sandbox",
                working_dir=self.config.work_dir,
                host_config=host_config,
                name=container_name,
                tty=True,
                detach=True,
            )

            self.container = self.client.containers.get(container["Id"])
            await asyncio.to_thread(self.container.start)

            self.terminal = AsyncDockerizedTerminal(
                container["Id"],
                self.config.work_dir,
                env_vars={"PYTHONUNBUFFERED": "1"},
            )
            await self.terminal.init()

            return self

        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to create sandbox: {e}") from e

    def _prepare_volume_bindings(self) -> dict[str, dict[str, str]]:
        bindings = {}
        work_dir = self._ensure_host_dir(self.config.work_dir)
        bindings[work_dir] = {"bind": self.config.work_dir, "mode": "rw"}

        for host_path, container_path in self.volume_bindings.items():
            bindings[host_path] = {"bind": container_path, "mode": "rw"}

        return bindings

    @staticmethod
    def _ensure_host_dir(path: str) -> str:
        host_path = os.path.join(
            tempfile.gettempdir(),
            f"sandbox_{os.path.basename(path)}_{os.urandom(4).hex()}",
        )
        os.makedirs(host_path, exist_ok=True)
        return host_path

    async def run_command(self, cmd: str, timeout: int | None = None) -> str:
        if not self.terminal:
            raise RuntimeError("Sandbox not initialized")

        try:
            return await self.terminal.run_command(cmd, timeout or self.config.timeout)
        except TimeoutError:
            raise SandboxTimeoutError(
                f"Command execution timed out after {timeout or self.config.timeout} seconds"
            )

    async def read_file(self, path: str) -> str:
        if not self.container:
            raise RuntimeError("Sandbox not initialized")

        try:
            resolved_path = self._safe_resolve_path(path)
            tar_stream, _ = await asyncio.to_thread(
                self.container.get_archive, resolved_path
            )
            content = await self._read_from_tar(tar_stream)
            return content.decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    async def write_file(self, path: str, content: str) -> None:
        if not self.container:
            raise RuntimeError("Sandbox not initialized")

        try:
            resolved_path = self._safe_resolve_path(path)
            parent_dir = os.path.dirname(resolved_path)

            if parent_dir:
                await self.run_command(f"mkdir -p {parent_dir}")

            tar_stream = await self._create_tar_stream(
                os.path.basename(path), content.encode("utf-8")
            )

            await asyncio.to_thread(
                self.container.put_archive, parent_dir or "/", tar_stream
            )

        except Exception as e:
            raise RuntimeError(f"Failed to write file: {e}")

    def _safe_resolve_path(self, path: str) -> str:
        if ".." in path.split("/"):
            raise ValueError("Path contains potentially unsafe patterns")

        resolved = (
            os.path.join(self.config.work_dir, path)
            if not os.path.isabs(path)
            else path
        )
        return resolved

    @staticmethod
    async def _create_tar_stream(name: str, content: bytes) -> io.BytesIO:
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=name)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))
        tar_stream.seek(0)
        return tar_stream

    @staticmethod
    async def _read_from_tar(tar_stream) -> bytes:
        with tempfile.NamedTemporaryFile() as tmp:
            for chunk in tar_stream:
                tmp.write(chunk)
            tmp.seek(0)

            with tarfile.open(fileobj=tmp) as tar:
                member = tar.next()
                if not member:
                    raise RuntimeError("Empty tar archive")

                file_content = tar.extractfile(member)
                if not file_content:
                    raise RuntimeError("Failed to extract file content")

                return file_content.read()

    async def cleanup(self) -> None:
        if self.terminal:
            try:
                await self.terminal.close()
            except Exception:
                pass
            finally:
                self.terminal = None

        if self.container:
            try:
                await asyncio.to_thread(self.container.stop, timeout=5)
            except Exception:
                pass

            try:
                await asyncio.to_thread(self.container.remove, force=True)
            except Exception:
                pass
            finally:
                self.container = None

    async def __aenter__(self) -> "DockerSandbox":
        return await self.create()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()
