import asyncio
import logging
import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DaytonaSandboxState(str, Enum):
    """Daytona sandbox state enumeration."""

    CREATING = "Creating"
    STARTING = "Starting"
    RUNNING = "Running"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    ARCHIVED = "Archived"
    ERROR = "Error"


class DaytonaSettings(BaseModel):
    """Daytona configuration settings."""

    api_key: Optional[str] = Field(default=None, description="Daytona API key")
    server_url: Optional[str] = Field(default=None, description="Daytona server URL")
    target: Optional[str] = Field(default=None, description="Daytona target")
    sandbox_image: str = Field(
        default="whitezxj/sandbox:0.1.0", description="Sandbox image name"
    )
    vnc_password: str = Field(default="deerflow", description="VNC password")
    cpu: int = Field(default=2, ge=1, le=8, description="CPU cores")
    memory: int = Field(default=4, ge=1, le=16, description="Memory in GB")
    disk: int = Field(default=5, ge=1, le=50, description="Disk in GB")
    auto_stop_interval: int = Field(
        default=15, description="Auto stop interval in minutes"
    )
    auto_archive_interval: int = Field(
        default=1440, description="Auto archive interval in minutes"
    )


class DaytonaSandboxInfo(BaseModel):
    """Daytona sandbox information."""

    sandbox_id: str
    state: DaytonaSandboxState = DaytonaSandboxState.CREATING
    created_at: float = Field(default_factory=time.time)
    vnc_url: Optional[str] = None
    website_url: Optional[str] = None
    workspace_path: str = "/workspace"


class DaytonaSandbox:
    """Daytona cloud sandbox implementation.

    This class provides integration with Daytona cloud sandbox service,
    allowing remote isolated execution environments.
    """

    def __init__(self, settings: Optional[DaytonaSettings] = None):
        self.settings = settings or DaytonaSettings()
        self._daytona: Optional[Any] = None
        self._sandbox: Optional[Any] = None
        self._sandbox_info: Optional[DaytonaSandboxInfo] = None
        self._initialized = False

    def _ensure_daytona_client(self) -> Any:
        """Ensure Daytona client is initialized."""
        if self._daytona is not None:
            return self._daytona

        try:
            from daytona import Daytona, DaytonaConfig

            if not self.settings.api_key:
                raise ValueError("Daytona API key is required")

            config = DaytonaConfig(
                api_key=self.settings.api_key,
                server_url=self.settings.server_url,
                target=self.settings.target,
            )
            self._daytona = Daytona(config)
            logger.info("Daytona client initialized")
            return self._daytona

        except ImportError:
            raise ImportError(
                "daytona-sdk is not installed. Install it with: pip install daytona-sdk"
            )

    async def create(self, project_id: Optional[str] = None) -> DaytonaSandboxInfo:
        """Create a new Daytona sandbox.

        Args:
            project_id: Optional project ID for labeling

        Returns:
            DaytonaSandboxInfo with sandbox details
        """
        daytona = self._ensure_daytona_client()

        try:
            from daytona import CreateSandboxFromImageParams, Resources

            labels = {"id": project_id} if project_id else None

            params = CreateSandboxFromImageParams(
                image=self.settings.sandbox_image,
                public=True,
                labels=labels,
                env_vars={
                    "CHROME_PERSISTENT_SESSION": "true",
                    "RESOLUTION": "1024x768x24",
                    "RESOLUTION_WIDTH": "1024",
                    "RESOLUTION_HEIGHT": "768",
                    "VNC_PASSWORD": self.settings.vnc_password,
                    "ANONYMIZED_TELEMETRY": "false",
                    "CHROME_DEBUGGING_PORT": "9222",
                    "CHROME_DEBUGGING_HOST": "localhost",
                },
                resources=Resources(
                    cpu=self.settings.cpu,
                    memory=self.settings.memory,
                    disk=self.settings.disk,
                ),
                auto_stop_interval=self.settings.auto_stop_interval,
                auto_archive_interval=self.settings.auto_archive_interval,
            )

            logger.info("Creating Daytona sandbox...")
            self._sandbox = daytona.create(params)

            self._sandbox_info = DaytonaSandboxInfo(
                sandbox_id=self._sandbox.id,
                state=DaytonaSandboxState.RUNNING,
            )

            await self._start_supervisord()
            await self._get_preview_urls()

            self._initialized = True
            logger.info(f"Daytona sandbox created with ID: {self._sandbox.id}")

            return self._sandbox_info

        except Exception as e:
            logger.error(f"Failed to create Daytona sandbox: {e}")
            raise

    async def get_or_start(self, sandbox_id: str) -> DaytonaSandboxInfo:
        """Get an existing sandbox or start it if stopped.

        Args:
            sandbox_id: The sandbox ID to retrieve

        Returns:
            DaytonaSandboxInfo with sandbox details
        """
        daytona = self._ensure_daytona_client()

        try:
            from daytona import SandboxState

            self._sandbox = daytona.get(sandbox_id)

            if self._sandbox.state in [SandboxState.ARCHIVED, SandboxState.STOPPED]:
                logger.info(f"Starting sandbox {sandbox_id}...")
                daytona.start(self._sandbox)
                await self._start_supervisord()

            self._sandbox_info = DaytonaSandboxInfo(
                sandbox_id=sandbox_id,
                state=DaytonaSandboxState.RUNNING,
            )

            await self._get_preview_urls()

            self._initialized = True
            return self._sandbox_info

        except Exception as e:
            logger.error(f"Failed to get/start sandbox {sandbox_id}: {e}")
            raise

    async def _start_supervisord(self) -> None:
        """Start supervisord in the sandbox."""
        if not self._sandbox:
            return

        try:
            from daytona import SessionExecuteRequest

            session_id = "supervisord-session"
            logger.info(f"Starting supervisord in session {session_id}")

            self._sandbox.process.create_session(session_id)
            self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(
                    command="exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf",
                    var_async=True,
                ),
            )

            await asyncio.sleep(5)
            logger.info("Supervisord started successfully")

        except Exception as e:
            logger.warning(f"Failed to start supervisord: {e}")

    async def _get_preview_urls(self) -> None:
        """Get VNC and website preview URLs."""
        if not self._sandbox or not self._sandbox_info:
            return

        try:
            vnc_link = self._sandbox.get_preview_link(6080)
            website_link = self._sandbox.get_preview_link(8080)

            self._sandbox_info.vnc_url = (
                vnc_link.url if hasattr(vnc_link, "url") else str(vnc_link)
            )
            self._sandbox_info.website_url = (
                website_link.url if hasattr(website_link, "url") else str(website_link)
            )

            logger.info(f"VNC URL: {self._sandbox_info.vnc_url}")
            logger.info(f"Website URL: {self._sandbox_info.website_url}")

        except Exception as e:
            logger.warning(f"Failed to get preview URLs: {e}")

    async def execute_command(self, command: str, timeout: int = 60) -> str:
        """Execute a command in the sandbox.

        Args:
            command: Command to execute
            timeout: Execution timeout in seconds

        Returns:
            Command output
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not initialized. Call create() first.")

        try:
            from daytona import SessionExecuteRequest

            session_id = f"cmd-{int(time.time())}"
            self._sandbox.process.create_session(session_id)

            result = self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=command, var_async=False),
            )

            return result.result if hasattr(result, "result") else str(result)

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox.

        Args:
            path: File path in the sandbox

        Returns:
            File content
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not initialized. Call create() first.")

        try:
            content = self._sandbox.fs.download_file(path)
            return content.decode("utf-8") if isinstance(content, bytes) else content
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox.

        Args:
            path: File path in the sandbox
            content: File content to write
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not initialized. Call create() first.")

        try:
            self._sandbox.fs.upload_file(path, content.encode("utf-8"))
            logger.info(f"File written: {path}")
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            raise

    async def list_files(self, path: str = "/workspace") -> list[dict[str, Any]]:
        """List files in a directory.

        Args:
            path: Directory path in the sandbox

        Returns:
            List of file information
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not initialized. Call create() first.")

        try:
            files = self._sandbox.fs.list_files(path)
            return [
                {
                    "name": f.name,
                    "path": f.path,
                    "is_dir": f.is_dir,
                    "size": getattr(f, "size", 0),
                }
                for f in files
            ]
        except Exception as e:
            logger.error(f"Failed to list files in {path}: {e}")
            raise

    async def stop(self) -> None:
        """Stop the sandbox."""
        if not self._sandbox or not self._daytona:
            return

        try:
            self._daytona.stop(self._sandbox)
            if self._sandbox_info:
                self._sandbox_info.state = DaytonaSandboxState.STOPPED
            logger.info(f"Sandbox {self._sandbox.id} stopped")
        except Exception as e:
            logger.error(f"Failed to stop sandbox: {e}")
            raise

    async def delete(self) -> None:
        """Delete the sandbox."""
        if not self._sandbox or not self._daytona:
            return

        try:
            self._daytona.delete(self._sandbox)
            logger.info(f"Sandbox {self._sandbox.id} deleted")
            self._sandbox = None
            self._sandbox_info = None
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to delete sandbox: {e}")
            raise

    def get_info(self) -> Optional[DaytonaSandboxInfo]:
        """Get current sandbox information."""
        return self._sandbox_info

    @property
    def sandbox_id(self) -> Optional[str]:
        """Get the sandbox ID."""
        return self._sandbox_info.sandbox_id if self._sandbox_info else None

    @property
    def vnc_url(self) -> Optional[str]:
        """Get the VNC URL."""
        return self._sandbox_info.vnc_url if self._sandbox_info else None

    @property
    def website_url(self) -> Optional[str]:
        """Get the website URL."""
        return self._sandbox_info.website_url if self._sandbox_info else None

    @property
    def is_initialized(self) -> bool:
        """Check if sandbox is initialized."""
        return self._initialized
