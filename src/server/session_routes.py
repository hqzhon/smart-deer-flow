# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .session_models import (
    Session,
    SessionCreate,
    SessionDetail,
    SessionListResponse,
    SessionStatus,
    SessionUpdate,
)
from .session_store import get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[SessionStatus] = Query(None, description="Filter by status"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search in title and topic"),
    sort_by: str = Query("updated_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
):
    """List all sessions with filtering and pagination."""
    store = get_session_store()

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    sessions, total = await store.list_sessions(
        status=status,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        tags=tag_list,
        search=search,
    )

    return SessionListResponse(
        sessions=sessions,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.post("", response_model=Session)
async def create_session(request: SessionCreate):
    """Create a new session."""
    store = get_session_store()

    session = await store.create_session(
        title=request.title,
        research_topic=request.research_topic,
        report_style=request.report_style,
        tags=request.tags,
        metadata=request.metadata,
    )

    logger.info(f"Created session: {session.id}")
    return session


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    """Get session details by ID."""
    store = get_session_store()

    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from src.context import ResearchMemoryManager

    memory_manager = ResearchMemoryManager.get_instance()
    memory = memory_manager.get_memory(session.thread_id)

    messages = []
    plan = None
    metrics = None

    if memory:
        state = memory.export_state()
        messages = state.get("messages", [])
        plan = state.get("current_plan")
        metrics = {
            "total_tokens": state.get("total_tokens", 0),
            "compressed_tokens": state.get("compressed_tokens", 0),
        }

    return SessionDetail(
        **session.model_dump(),
        messages=messages,
        plan=plan,
        metrics=metrics,
    )


@router.patch("/{session_id}", response_model=Session)
async def update_session(session_id: str, request: SessionUpdate):
    """Update a session."""
    store = get_session_store()

    session = await store.update_session(
        session_id,
        title=request.title,
        status=request.status,
        tags=request.tags,
        metadata=request.metadata,
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Updated session: {session_id}")
    return session


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    hard: bool = Query(False, description="Permanently delete the session"),
):
    """Delete a session (soft delete by default)."""
    store = get_session_store()

    if hard:
        success = await store.hard_delete_session(session_id)
    else:
        success = await store.delete_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    from src.context import ResearchMemoryManager

    memory_manager = ResearchMemoryManager.get_instance()
    memory_manager.delete_session(session_id)

    logger.info(f"{'Hard' if hard else 'Soft'} deleted session: {session_id}")
    return {"status": "deleted", "session_id": session_id}


@router.get("/thread/{thread_id}", response_model=Session)
async def get_session_by_thread(thread_id: str):
    """Get session by thread ID."""
    store = get_session_store()

    session = await store.get_session_by_thread(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.post("/{session_id}/archive")
async def archive_session(session_id: str):
    """Archive a session."""
    store = get_session_store()

    session = await store.update_session(session_id, status=SessionStatus.ARCHIVED)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "archived", "session_id": session_id}


@router.post("/{session_id}/restore")
async def restore_session(session_id: str):
    """Restore an archived session."""
    store = get_session_store()

    session = await store.update_session(session_id, status=SessionStatus.ACTIVE)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "restored", "session_id": session_id}


@router.get("/{session_id}/export")
async def export_session(session_id: str):
    """Export session data."""
    store = get_session_store()

    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from src.context import ResearchMemoryManager

    memory_manager = ResearchMemoryManager.get_instance()
    memory = memory_manager.get_memory(session.thread_id)

    export_data = {
        "session": session.model_dump(),
        "memory": memory.export_state() if memory else None,
    }

    return export_data
