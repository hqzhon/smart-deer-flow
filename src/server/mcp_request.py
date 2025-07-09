# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Dict, List, Optional
import re

from pydantic import BaseModel, Field, validator


class MCPServerMetadataRequest(BaseModel):
    """Request model for MCP server metadata."""

    transport: str = Field(
        ...,
        description="The type of MCP server connection (stdio or sse)",
        pattern=r"^(stdio|sse)$",
    )
    command: Optional[str] = Field(
        None, description="The command to execute (for stdio type)", max_length=500
    )
    args: Optional[List[str]] = Field(
        None, description="Command arguments (for stdio type)"
    )
    url: Optional[str] = Field(
        None, description="The URL of the SSE server (for sse type)", max_length=2048
    )
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    timeout_seconds: Optional[int] = Field(
        None,
        description="Optional custom timeout in seconds for the operation",
        ge=1,
        le=300,
    )

    @validator("command")
    def validate_command(cls, v):
        if v is not None:
            # Check for command injection patterns
            dangerous_patterns = [
                r";",
                r"&&",
                r"\|\|",
                r"`",
                r"\$\(",
                r"\${",
                r">",
                r">>",
                r"<",
                r"&",
                r"\|",
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, v):
                    raise ValueError("Potentially unsafe command detected")
            # Only allow alphanumeric, dash, underscore, dot, and slash
            if not re.match(r"^[a-zA-Z0-9._/-]+$", v):
                raise ValueError("Command contains invalid characters")
        return v

    @validator("args")
    def validate_args(cls, v):
        if v is not None:
            if len(v) > 50:  # Limit number of arguments
                raise ValueError("Too many command arguments")
            for arg in v:
                if len(arg) > 1000:  # Limit argument length
                    raise ValueError("Command argument too long")
                # Check for dangerous patterns in arguments
                dangerous_patterns = [r";", r"&&", r"\|\|", r"`", r"\$\(", r"\${"]
                for pattern in dangerous_patterns:
                    if re.search(pattern, arg):
                        raise ValueError("Potentially unsafe argument detected")
        return v

    @validator("url")
    def validate_url(cls, v):
        if v is not None:
            # Check for valid URL format
            url_pattern = re.compile(
                r"^https?://"  # http:// or https://
                r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"  # domain...
                r"(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # host...
                r"localhost|"  # localhost...
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
                r"(?::\d+)?"  # optional port
                r"(?:/?|[/?]\S+)$",
                re.IGNORECASE,
            )
            if not url_pattern.match(v):
                raise ValueError("Invalid URL format")
        return v

    @validator("env")
    def validate_env(cls, v):
        if v is not None:
            if len(v) > 100:  # Limit number of environment variables
                raise ValueError("Too many environment variables")
            for key, value in v.items():
                if len(key) > 100 or len(value) > 1000:
                    raise ValueError("Environment variable key or value too long")
                # Check for valid environment variable names
                if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
                    raise ValueError("Invalid environment variable name")
        return v


class MCPServerMetadataResponse(BaseModel):
    """Response model for MCP server metadata."""

    transport: str = Field(
        ..., description="The type of MCP server connection (stdio or sse)"
    )
    command: Optional[str] = Field(
        None, description="The command to execute (for stdio type)"
    )
    args: Optional[List[str]] = Field(
        None, description="Command arguments (for stdio type)"
    )
    url: Optional[str] = Field(
        None, description="The URL of the SSE server (for sse type)"
    )
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    tools: List = Field(
        default_factory=list, description="Available tools from the MCP server"
    )
