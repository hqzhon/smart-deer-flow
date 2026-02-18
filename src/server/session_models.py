# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Session(BaseModel):
    id: str = Field(..., description="Unique session identifier")
    thread_id: str = Field(..., description="Thread ID for the session")
    title: Optional[str] = Field(None, description="Session title")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    status: SessionStatus = Field(
        default=SessionStatus.ACTIVE, description="Session status"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional session metadata"
    )
    message_count: int = Field(default=0, description="Number of messages")
    last_message_preview: Optional[str] = Field(
        None, description="Preview of last message", max_length=200
    )
    research_topic: Optional[str] = Field(
        None, description="Research topic", max_length=500
    )
    report_style: Optional[str] = Field(None, description="Report style used")
    tags: List[str] = Field(default_factory=list, description="Session tags")

    class Config:
        use_enum_values = True


class SessionCreate(BaseModel):
    title: Optional[str] = Field(None, description="Session title", max_length=200)
    research_topic: Optional[str] = Field(
        None, description="Initial research topic", max_length=500
    )
    report_style: Optional[str] = Field(None, description="Report style")
    tags: List[str] = Field(default_factory=list, description="Session tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SessionUpdate(BaseModel):
    title: Optional[str] = Field(None, description="Session title", max_length=200)
    status: Optional[SessionStatus] = Field(None, description="Session status")
    tags: Optional[List[str]] = Field(None, description="Session tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SessionListResponse(BaseModel):
    sessions: List[Session] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of sessions")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=20, description="Page size")
    has_more: bool = Field(default=False, description="Whether there are more sessions")


class SessionDetail(Session):
    messages: List[Dict[str, Any]] = Field(
        default_factory=list, description="Session messages"
    )
    plan: Optional[Dict[str, Any]] = Field(None, description="Current plan")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Session metrics")


class SessionSummary(BaseModel):
    id: str
    title: Optional[str]
    research_topic: Optional[str]
    message_count: int
    created_at: datetime
    status: SessionStatus

    class Config:
        use_enum_values = True
