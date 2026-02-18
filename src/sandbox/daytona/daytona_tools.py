import asyncio
import logging
from typing import Any, Optional

from pydantic import Field

from src.sandbox.daytona.daytona_sandbox import DaytonaSandbox, DaytonaSettings
from src.tools.base_tool import BaseTool
from src.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)


class DaytonaShellTool(BaseTool):
    """Tool for executing shell commands in a Daytona sandbox."""

    name: str = "daytona_shell"
    description: str = """Execute shell commands in a remote Daytona sandbox.
Use this tool when you need to run commands in an isolated cloud environment.
The sandbox provides a full Linux environment with common tools installed."""

    _sandbox: Optional[DaytonaSandbox] = Field(default=None, exclude=True)
    _settings: Optional[DaytonaSettings] = Field(default=None, exclude=True)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "default": 60,
                    "description": "Command timeout in seconds",
                },
            },
            "required": ["command"],
        }

    @property
    def required_parameters(self) -> list[str]:
        return ["command"]

    @property
    def category(self) -> str:
        return "sandbox"

    @property
    def tags(self) -> list[str]:
        return ["daytona", "shell", "remote", "sandbox"]

    def set_sandbox(self, sandbox: DaytonaSandbox) -> None:
        self._sandbox = sandbox

    def set_settings(self, settings: DaytonaSettings) -> None:
        self._settings = settings

    async def _ensure_sandbox(self) -> DaytonaSandbox:
        if self._sandbox is None:
            self._sandbox = DaytonaSandbox(settings=self._settings)
            await self._sandbox.create()
        return self._sandbox

    def execute(self, **kwargs) -> ToolResult:
        return asyncio.get_event_loop().run_until_complete(self.async_execute(**kwargs))

    async def async_execute(
        self, command: str, timeout: int = 60, **kwargs
    ) -> ToolResult:
        try:
            sandbox = await self._ensure_sandbox()
            result = await sandbox.execute_command(command, timeout)
            return ToolResult(
                message=result, data={"command": command, "output": result}
            )
        except Exception as e:
            logger.error(f"Daytona shell command failed: {e}")
            return ToolResult(
                success=False, message=f"Command execution failed: {str(e)}"
            )


class DaytonaFileTool(BaseTool):
    """Tool for file operations in a Daytona sandbox."""

    name: str = "daytona_file"
    description: str = """Perform file operations in a remote Daytona sandbox.
Supports reading, writing, and listing files in the sandbox environment.
Use this for file management in an isolated cloud environment."""

    _sandbox: Optional[DaytonaSandbox] = Field(default=None, exclude=True)
    _settings: Optional[DaytonaSettings] = Field(default=None, exclude=True)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "File operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write action)",
                },
            },
            "required": ["action", "path"],
        }

    @property
    def required_parameters(self) -> list[str]:
        return ["action", "path"]

    @property
    def category(self) -> str:
        return "sandbox"

    @property
    def tags(self) -> list[str]:
        return ["daytona", "file", "remote", "sandbox"]

    def set_sandbox(self, sandbox: DaytonaSandbox) -> None:
        self._sandbox = sandbox

    def set_settings(self, settings: DaytonaSettings) -> None:
        self._settings = settings

    async def _ensure_sandbox(self) -> DaytonaSandbox:
        if self._sandbox is None:
            self._sandbox = DaytonaSandbox(settings=self._settings)
            await self._sandbox.create()
        return self._sandbox

    def execute(self, **kwargs) -> ToolResult:
        return asyncio.get_event_loop().run_until_complete(self.async_execute(**kwargs))

    async def async_execute(
        self,
        action: str,
        path: str,
        content: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            sandbox = await self._ensure_sandbox()

            if action == "read":
                result = await sandbox.read_file(path)
                return ToolResult(
                    message=result, data={"path": path, "content": result}
                )

            elif action == "write":
                if not content:
                    return ToolResult(
                        success=False, message="Content is required for write action"
                    )
                await sandbox.write_file(path, content)
                return ToolResult(message=f"File written: {path}", data={"path": path})

            elif action == "list":
                files = await sandbox.list_files(path)
                file_list = "\n".join(
                    f"{'[DIR]' if f['is_dir'] else '[FILE]'} {f['name']}" for f in files
                )
                return ToolResult(
                    message=file_list, data={"path": path, "files": files}
                )

            else:
                return ToolResult(success=False, message=f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Daytona file operation failed: {e}")
            return ToolResult(success=False, message=f"File operation failed: {str(e)}")


class DaytonaPythonTool(BaseTool):
    """Tool for executing Python code in a Daytona sandbox."""

    name: str = "daytona_python"
    description: str = """Execute Python code in a remote Daytona sandbox.
Provides an isolated environment for Python code execution with common packages installed.
Use this for safe code execution in a cloud environment."""

    _sandbox: Optional[DaytonaSandbox] = Field(default=None, exclude=True)
    _settings: Optional[DaytonaSettings] = Field(default=None, exclude=True)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "default": 60,
                    "description": "Execution timeout in seconds",
                },
            },
            "required": ["code"],
        }

    @property
    def required_parameters(self) -> list[str]:
        return ["code"]

    @property
    def category(self) -> str:
        return "sandbox"

    @property
    def tags(self) -> list[str]:
        return ["daytona", "python", "remote", "sandbox"]

    def set_sandbox(self, sandbox: DaytonaSandbox) -> None:
        self._sandbox = sandbox

    def set_settings(self, settings: DaytonaSettings) -> None:
        self._settings = settings

    async def _ensure_sandbox(self) -> DaytonaSandbox:
        if self._sandbox is None:
            self._sandbox = DaytonaSandbox(settings=self._settings)
            await self._sandbox.create()
        return self._sandbox

    def execute(self, **kwargs) -> ToolResult:
        return asyncio.get_event_loop().run_until_complete(self.async_execute(**kwargs))

    async def async_execute(self, code: str, timeout: int = 60, **kwargs) -> ToolResult:
        try:
            sandbox = await self._ensure_sandbox()

            escaped_code = code.replace("'", "'\"'\"'")
            command = f"python3 -c '{escaped_code}'"

            result = await sandbox.execute_command(command, timeout)
            return ToolResult(message=result, data={"code": code, "output": result})

        except Exception as e:
            logger.error(f"Daytona Python execution failed: {e}")
            return ToolResult(
                success=False, message=f"Python execution failed: {str(e)}"
            )
