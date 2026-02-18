class SandboxError(Exception):
    pass


class SandboxTimeoutError(SandboxError):
    pass


class SandboxNotFoundError(SandboxError):
    pass


class SandboxCreationError(SandboxError):
    pass
