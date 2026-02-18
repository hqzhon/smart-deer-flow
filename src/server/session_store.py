# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .session_models import Session, SessionStatus

logger = logging.getLogger(__name__)


class SessionStore:
    """In-memory session store with optional persistence hooks."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._thread_to_session: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        thread_id: Optional[str] = None,
        title: Optional[str] = None,
        research_topic: Optional[str] = None,
        report_style: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session."""
        async with self._lock:
            session_id = str(uuid4())
            thread_id = thread_id or str(uuid4())

            session = Session(
                id=session_id,
                thread_id=thread_id,
                title=title,
                research_topic=research_topic,
                report_style=report_style,
                tags=tags or [],
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            self._sessions[session_id] = session
            self._thread_to_session[thread_id] = session_id

            logger.info(f"Created session {session_id} with thread {thread_id}")
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def get_session_by_thread(self, thread_id: str) -> Optional[Session]:
        """Get a session by thread ID."""
        session_id = self._thread_to_session.get(thread_id)
        if session_id:
            return self._sessions.get(session_id)
        return None

    async def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> tuple[List[Session], int]:
        """List sessions with filtering and pagination."""
        async with self._lock:
            sessions = list(self._sessions.values())

            if status:
                sessions = [s for s in sessions if s.status == status]

            if tags:
                sessions = [s for s in sessions if any(tag in s.tags for tag in tags)]

            if search:
                search_lower = search.lower()
                sessions = [
                    s
                    for s in sessions
                    if (
                        (s.title and search_lower in s.title.lower())
                        or (
                            s.research_topic
                            and search_lower in s.research_topic.lower()
                        )
                    )
                ]

            reverse = sort_order.lower() == "desc"
            sessions.sort(
                key=lambda s: getattr(s, sort_by, s.updated_at), reverse=reverse
            )

            total = len(sessions)
            start = (page - 1) * page_size
            end = start + page_size

            return sessions[start:end], total

    async def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_count: Optional[int] = None,
        last_message_preview: Optional[str] = None,
        research_topic: Optional[str] = None,
    ) -> Optional[Session]:
        """Update a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            if title is not None:
                session.title = title
            if status is not None:
                session.status = status
            if tags is not None:
                session.tags = tags
            if metadata is not None:
                session.metadata = {**session.metadata, **metadata}
            if message_count is not None:
                session.message_count = message_count
            if last_message_preview is not None:
                session.last_message_preview = last_message_preview[:200]
            if research_topic is not None:
                session.research_topic = research_topic

            session.updated_at = datetime.utcnow()

            return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session (soft delete by default)."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.status = SessionStatus.DELETED
            session.updated_at = datetime.utcnow()

            return True

    async def hard_delete_session(self, session_id: str) -> bool:
        """Permanently delete a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            thread_id = session.thread_id
            del self._sessions[session_id]
            if thread_id in self._thread_to_session:
                del self._thread_to_session[thread_id]

            return True

    async def get_or_create_session(
        self,
        thread_id: str,
        research_topic: Optional[str] = None,
    ) -> Session:
        """Get existing session or create new one."""
        session = await self.get_session_by_thread(thread_id)
        if session:
            return session

        return await self.create_session(
            thread_id=thread_id,
            research_topic=research_topic,
        )

    async def increment_message_count(self, session_id: str) -> None:
        """Increment message count for a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.message_count += 1
                session.updated_at = datetime.utcnow()

    async def set_last_message(
        self, session_id: str, preview: str, research_topic: Optional[str] = None
    ) -> None:
        """Set last message preview for a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_message_preview = preview[:200]
                if research_topic:
                    session.research_topic = research_topic
                session.updated_at = datetime.utcnow()


_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get the global session store instance."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
