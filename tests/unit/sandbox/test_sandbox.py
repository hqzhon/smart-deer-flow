import pytest

from src.sandbox.core.exceptions import (
    SandboxError,
    SandboxTimeoutError,
    SandboxNotFoundError,
    SandboxCreationError,
)
from src.sandbox.core.sandbox import SandboxSettings, DockerSandbox


class TestSandboxSettings:
    def test_default_settings(self):
        settings = SandboxSettings()
        assert settings.image == "python:3.12-slim"
        assert settings.work_dir == "/workspace"
        assert settings.memory_limit == "512m"
        assert settings.cpu_limit == 1.0
        assert settings.timeout == 300
        assert settings.network_enabled is False

    def test_custom_settings(self):
        settings = SandboxSettings(
            image="custom:latest",
            work_dir="/app",
            memory_limit="1g",
            cpu_limit=2.0,
            timeout=600,
            network_enabled=True,
        )
        assert settings.image == "custom:latest"
        assert settings.work_dir == "/app"
        assert settings.memory_limit == "1g"
        assert settings.cpu_limit == 2.0
        assert settings.timeout == 600
        assert settings.network_enabled is True


class TestSandboxExceptions:
    def test_sandbox_error(self):
        error = SandboxError("Test error")
        assert str(error) == "Test error"

    def test_sandbox_timeout_error(self):
        error = SandboxTimeoutError("Timeout")
        assert isinstance(error, SandboxError)
        assert str(error) == "Timeout"

    def test_sandbox_not_found_error(self):
        error = SandboxNotFoundError("Not found")
        assert isinstance(error, SandboxError)

    def test_sandbox_creation_error(self):
        error = SandboxCreationError("Creation failed")
        assert isinstance(error, SandboxError)


class TestDockerSandbox:
    def test_sandbox_creation(self):
        sandbox = DockerSandbox()
        assert sandbox.config is not None
        assert sandbox.container is None
        assert sandbox.terminal is None

    def test_sandbox_with_custom_config(self):
        config = SandboxSettings(image="custom:latest")
        sandbox = DockerSandbox(config=config)
        assert sandbox.config.image == "custom:latest"

    def test_safe_resolve_path(self):
        sandbox = DockerSandbox()
        resolved = sandbox._safe_resolve_path("test.py")
        assert resolved.endswith("test.py")

    def test_safe_resolve_path_absolute(self):
        sandbox = DockerSandbox()
        resolved = sandbox._safe_resolve_path("/absolute/path.py")
        assert resolved == "/absolute/path.py"

    def test_safe_resolve_path_traversal(self):
        sandbox = DockerSandbox()
        with pytest.raises(ValueError):
            sandbox._safe_resolve_path("../../../etc/passwd")

    def test_ensure_host_dir(self):
        host_dir = DockerSandbox._ensure_host_dir("/workspace")
        assert "sandbox" in host_dir

    @pytest.mark.asyncio
    async def test_run_command_without_terminal(self):
        sandbox = DockerSandbox()
        with pytest.raises(RuntimeError, match="not initialized"):
            await sandbox.run_command("echo test")

    @pytest.mark.asyncio
    async def test_read_file_without_container(self):
        sandbox = DockerSandbox()
        with pytest.raises(RuntimeError, match="not initialized"):
            await sandbox.read_file("test.py")

    @pytest.mark.asyncio
    async def test_write_file_without_container(self):
        sandbox = DockerSandbox()
        with pytest.raises(RuntimeError, match="not initialized"):
            await sandbox.write_file("test.py", "content")

    @pytest.mark.asyncio
    async def test_cleanup_without_resources(self):
        sandbox = DockerSandbox()
        await sandbox.cleanup()
        assert sandbox.container is None
        assert sandbox.terminal is None


class TestSandboxManager:
    def test_manager_creation(self):
        from src.sandbox.core.manager import SandboxManager

        manager = SandboxManager(
            max_sandboxes=50, idle_timeout=1800, auto_cleanup=False
        )
        assert manager.max_sandboxes == 50
        assert manager.idle_timeout == 1800
        assert len(manager._sandboxes) == 0

    def test_get_stats(self):
        from src.sandbox.core.manager import SandboxManager

        manager = SandboxManager(auto_cleanup=False)
        stats = manager.get_stats()
        assert "total_sandboxes" in stats
        assert "active_operations" in stats
        assert "max_sandboxes" in stats

    @pytest.mark.asyncio
    async def test_create_sandbox_limit_reached(self):
        from src.sandbox.core.manager import SandboxManager

        manager = SandboxManager(max_sandboxes=0)
        with pytest.raises(RuntimeError, match="Maximum number"):
            await manager.create_sandbox()

    @pytest.mark.asyncio
    async def test_get_sandbox_not_found(self):
        from src.sandbox.core.manager import SandboxManager

        manager = SandboxManager()
        with pytest.raises(KeyError):
            async with manager.sandbox_operation("nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_cleanup(self):
        from src.sandbox.core.manager import SandboxManager

        manager = SandboxManager()
        await manager.cleanup()
        assert len(manager._sandboxes) == 0
        assert manager._is_shutting_down is True
