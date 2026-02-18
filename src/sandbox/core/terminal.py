import asyncio
import select
import socket
import time
from typing import Any

from src.sandbox.core.exceptions import SandboxTimeoutError


class AsyncDockerizedTerminal:
    def __init__(
        self, container_id: str, work_dir: str, env_vars: dict[str, str] | None = None
    ):
        self.container_id = container_id
        self.work_dir = work_dir
        self.env_vars = env_vars or {}
        self.api: Any = None
        self.exec_id: str | None = None
        self.socket: socket.socket | None = None
        self._sentinel = f"__SENTINEL_{int(time.time() * 1000)}__"

    async def init(self) -> None:
        import docker

        client = docker.from_env()
        self.api = client.api

        env_str = " ".join(f"{k}={v}" for k, v in self.env_vars.items())
        exec_data = self.api.exec_create(
            self.container_id,
            [
                "bash",
                "-c",
                f"cd {self.work_dir} && {env_str} exec bash --norc --noprofile",
            ],
            stdin=True,
            tty=True,
            stdout=True,
            stderr=True,
        )
        self.exec_id = exec_data["Id"]

        sock_data = self.api.exec_start(self.exec_id, socket=True, tty=True, demux=True)
        self.socket = sock_data._sock
        self.socket.setblocking(False)

        await asyncio.sleep(0.1)

    async def run_command(self, cmd: str, timeout: int = 30) -> str:
        if not self.socket:
            raise RuntimeError("Terminal not initialized")

        full_cmd = f"{cmd}\necho '{self._sentinel}'\n"
        self.socket.sendall(full_cmd.encode())

        output = b""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise SandboxTimeoutError(f"Command timed out after {timeout} seconds")

            try:
                ready, _, _ = select.select([self.socket], [], [], 0.1)
                if ready:
                    chunk = self.socket.recv(4096)
                    if chunk:
                        output += chunk
                        if self._sentinel.encode() in output:
                            break
            except (BlockingIOError, OSError):
                await asyncio.sleep(0.05)

        result = output.decode("utf-8", errors="replace")
        sentinel_pos = result.find(self._sentinel)
        if sentinel_pos != -1:
            result = result[:sentinel_pos]

        lines = result.strip().split("\n")
        cleaned_lines = [line for line in lines if not line.startswith(self._sentinel)]
        return "\n".join(cleaned_lines).strip()

    async def close(self) -> None:
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            finally:
                self.socket = None
