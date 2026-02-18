import pytest
from unittest.mock import MagicMock

from src.sandbox.daytona.daytona_sandbox import (
    DaytonaSandbox,
    DaytonaSettings,
    DaytonaSandboxInfo,
    DaytonaSandboxState,
)
from src.sandbox.daytona.daytona_tools import (
    DaytonaShellTool,
    DaytonaFileTool,
    DaytonaPythonTool,
)


class TestDaytonaSandboxState:
    def test_state_values(self):
        assert DaytonaSandboxState.CREATING == "Creating"
        assert DaytonaSandboxState.RUNNING == "Running"
        assert DaytonaSandboxState.STOPPED == "Stopped"
        assert DaytonaSandboxState.ARCHIVED == "Archived"
        assert DaytonaSandboxState.ERROR == "Error"


class TestDaytonaSettings:
    def test_default_values(self):
        settings = DaytonaSettings()
        assert settings.api_key is None
        assert settings.server_url is None
        assert settings.target is None
        assert settings.sandbox_image == "whitezxj/sandbox:0.1.0"
        assert settings.vnc_password == "deerflow"
        assert settings.cpu == 2
        assert settings.memory == 4
        assert settings.disk == 5

    def test_custom_values(self):
        settings = DaytonaSettings(
            api_key="test-key",
            server_url="https://api.daytona.io",
            target="us-east-1",
            cpu=4,
            memory=8,
        )
        assert settings.api_key == "test-key"
        assert settings.server_url == "https://api.daytona.io"
        assert settings.target == "us-east-1"
        assert settings.cpu == 4
        assert settings.memory == 8


class TestDaytonaSandboxInfo:
    def test_default_values(self):
        info = DaytonaSandboxInfo(sandbox_id="test-id")
        assert info.sandbox_id == "test-id"
        assert info.state == DaytonaSandboxState.CREATING
        assert info.vnc_url is None
        assert info.website_url is None
        assert info.workspace_path == "/workspace"

    def test_with_values(self):
        info = DaytonaSandboxInfo(
            sandbox_id="test-id",
            state=DaytonaSandboxState.RUNNING,
            vnc_url="https://vnc.example.com",
            website_url="https://web.example.com",
        )
        assert info.state == DaytonaSandboxState.RUNNING
        assert info.vnc_url == "https://vnc.example.com"


class TestDaytonaSandbox:
    def test_initialization(self):
        sandbox = DaytonaSandbox()
        assert sandbox._daytona is None
        assert sandbox._sandbox is None
        assert sandbox._sandbox_info is None
        assert sandbox._initialized is False

    def test_initialization_with_settings(self):
        settings = DaytonaSettings(api_key="test-key")
        sandbox = DaytonaSandbox(settings=settings)
        assert sandbox.settings.api_key == "test-key"

    def test_ensure_daytona_client_no_api_key(self):
        sandbox = DaytonaSandbox()

        with pytest.raises(ValueError, match="API key is required"):
            try:
                sandbox._ensure_daytona_client()
            except ImportError:
                pytest.skip("daytona-sdk not installed")

    def test_get_info(self):
        sandbox = DaytonaSandbox()
        assert sandbox.get_info() is None

        sandbox._sandbox_info = DaytonaSandboxInfo(sandbox_id="test-id")
        info = sandbox.get_info()
        assert info.sandbox_id == "test-id"

    def test_sandbox_id_property(self):
        sandbox = DaytonaSandbox()
        assert sandbox.sandbox_id is None

        sandbox._sandbox_info = DaytonaSandboxInfo(sandbox_id="test-id")
        assert sandbox.sandbox_id == "test-id"

    def test_vnc_url_property(self):
        sandbox = DaytonaSandbox()
        assert sandbox.vnc_url is None

        sandbox._sandbox_info = DaytonaSandboxInfo(
            sandbox_id="test-id", vnc_url="https://vnc.example.com"
        )
        assert sandbox.vnc_url == "https://vnc.example.com"

    def test_is_initialized_property(self):
        sandbox = DaytonaSandbox()
        assert sandbox.is_initialized is False

        sandbox._initialized = True
        assert sandbox.is_initialized is True


class TestDaytonaSandboxMocked:
    @pytest.mark.asyncio
    async def test_execute_command_not_initialized(self):
        sandbox = DaytonaSandbox()

        with pytest.raises(RuntimeError, match="not initialized"):
            await sandbox.execute_command("ls -la")

    @pytest.mark.asyncio
    async def test_read_file_not_initialized(self):
        sandbox = DaytonaSandbox()

        with pytest.raises(RuntimeError, match="not initialized"):
            await sandbox.read_file("/test/path")

    @pytest.mark.asyncio
    async def test_write_file_not_initialized(self):
        sandbox = DaytonaSandbox()

        with pytest.raises(RuntimeError, match="not initialized"):
            await sandbox.write_file("/test/path", "content")

    @pytest.mark.asyncio
    async def test_list_files_not_initialized(self):
        sandbox = DaytonaSandbox()

        with pytest.raises(RuntimeError, match="not initialized"):
            await sandbox.list_files("/workspace")


class TestDaytonaShellTool:
    def test_tool_properties(self):
        tool = DaytonaShellTool()
        assert tool.name == "daytona_shell"
        assert tool.category == "sandbox"
        assert "daytona" in tool.tags
        assert "shell" in tool.tags

    def test_parameters_schema(self):
        tool = DaytonaShellTool()
        params = tool.parameters
        assert "command" in params["properties"]
        assert "timeout" in params["properties"]
        assert "command" in tool.required_parameters

    def test_set_sandbox(self):
        tool = DaytonaShellTool()
        mock_sandbox = MagicMock()
        tool.set_sandbox(mock_sandbox)
        assert tool._sandbox == mock_sandbox

    def test_set_settings(self):
        tool = DaytonaShellTool()
        settings = DaytonaSettings(api_key="test-key")
        tool.set_settings(settings)
        assert tool._settings == settings


class TestDaytonaFileTool:
    def test_tool_properties(self):
        tool = DaytonaFileTool()
        assert tool.name == "daytona_file"
        assert tool.category == "sandbox"

    def test_parameters_schema(self):
        tool = DaytonaFileTool()
        params = tool.parameters
        assert "action" in params["properties"]
        assert "path" in params["properties"]
        assert "content" in params["properties"]

        actions = params["properties"]["action"]["enum"]
        assert "read" in actions
        assert "write" in actions
        assert "list" in actions


class TestDaytonaPythonTool:
    def test_tool_properties(self):
        tool = DaytonaPythonTool()
        assert tool.name == "daytona_python"
        assert tool.category == "sandbox"

    def test_parameters_schema(self):
        tool = DaytonaPythonTool()
        params = tool.parameters
        assert "code" in params["properties"]
        assert "timeout" in params["properties"]
        assert "code" in tool.required_parameters
