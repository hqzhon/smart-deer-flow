# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.server.app import app
from src.server.session_models import (
    Session,
    SessionStatus,
    SessionCreate,
    SessionUpdate,
)
from src.server.session_store import SessionStore


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_session_store():
    store = SessionStore()
    with patch("src.server.session_routes.get_session_store", return_value=store):
        yield store


class TestSessionModels:
    def test_session_creation(self):
        session = Session(
            id="test-id",
            thread_id="thread-123",
            title="Test Session",
            status=SessionStatus.ACTIVE,
        )
        assert session.id == "test-id"
        assert session.thread_id == "thread-123"
        assert session.title == "Test Session"
        assert session.status == "active"

    def test_session_create_model(self):
        create = SessionCreate(
            title="New Session",
            research_topic="AI Research",
            tags=["test", "ai"],
        )
        assert create.title == "New Session"
        assert create.research_topic == "AI Research"
        assert create.tags == ["test", "ai"]

    def test_session_update_model(self):
        update = SessionUpdate(
            title="Updated Title",
            status=SessionStatus.ARCHIVED,
        )
        assert update.title == "Updated Title"
        assert update.status == "archived"


class TestSessionStore:
    @pytest.mark.asyncio
    async def test_create_session(self):
        store = SessionStore()
        session = await store.create_session(
            title="Test Session",
            research_topic="Test Topic",
        )
        assert session.id is not None
        assert session.thread_id is not None
        assert session.title == "Test Session"
        assert session.research_topic == "Test Topic"

    @pytest.mark.asyncio
    async def test_get_session(self):
        store = SessionStore()
        created = await store.create_session(title="Test")
        retrieved = await store.get_session(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        store = SessionStore()
        result = await store.get_session("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        store = SessionStore()
        await store.create_session(title="Session 1")
        await store.create_session(title="Session 2")
        sessions, total = await store.list_sessions()
        assert total == 2
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_with_status_filter(self):
        store = SessionStore()
        await store.create_session(title="Active Session")
        session2 = await store.create_session(title="To Archive")
        await store.update_session(session2.id, status=SessionStatus.ARCHIVED)

        sessions, total = await store.list_sessions(status=SessionStatus.ACTIVE)
        assert total == 1
        assert sessions[0].title == "Active Session"

    @pytest.mark.asyncio
    async def test_update_session(self):
        store = SessionStore()
        session = await store.create_session(title="Original")
        updated = await store.update_session(session.id, title="Updated")
        assert updated is not None
        assert updated.title == "Updated"

    @pytest.mark.asyncio
    async def test_delete_session(self):
        store = SessionStore()
        session = await store.create_session(title="To Delete")
        result = await store.delete_session(session.id)
        assert result is True
        deleted = await store.get_session(session.id)
        assert deleted.status == SessionStatus.DELETED

    @pytest.mark.asyncio
    async def test_get_or_create_session(self):
        store = SessionStore()
        session1 = await store.get_or_create_session("thread-123")
        session2 = await store.get_or_create_session("thread-123")
        assert session1.id == session2.id


class TestSessionRoutes:
    def test_list_sessions_endpoint(self, client, mock_session_store):
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert "page" in data

    def test_create_session_endpoint(self, client):
        response = client.post(
            "/api/sessions",
            json={
                "title": "Test Session",
                "research_topic": "Test Topic",
                "tags": ["test"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Session"
        assert data["research_topic"] == "Test Topic"

    def test_get_session_endpoint(self, client):
        create_response = client.post(
            "/api/sessions",
            json={"title": "Test"},
        )
        session_id = create_response.json()["id"]

        response = client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id

    def test_get_session_not_found(self, client):
        response = client.get("/api/sessions/non-existent-id")
        assert response.status_code == 404

    def test_update_session_endpoint(self, client):
        create_response = client.post(
            "/api/sessions",
            json={"title": "Original"},
        )
        session_id = create_response.json()["id"]

        response = client.patch(
            f"/api/sessions/{session_id}",
            json={"title": "Updated Title"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"

    def test_delete_session_endpoint(self, client):
        create_response = client.post(
            "/api/sessions",
            json={"title": "To Delete"},
        )
        session_id = create_response.json()["id"]

        response = client.delete(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_archive_session_endpoint(self, client):
        create_response = client.post(
            "/api/sessions",
            json={"title": "To Archive"},
        )
        session_id = create_response.json()["id"]

        response = client.post(f"/api/sessions/{session_id}/archive")
        assert response.status_code == 200
        assert response.json()["status"] == "archived"

    def test_restore_session_endpoint(self, client):
        create_response = client.post(
            "/api/sessions",
            json={"title": "Test"},
        )
        session_id = create_response.json()["id"]

        client.post(f"/api/sessions/{session_id}/archive")
        response = client.post(f"/api/sessions/{session_id}/restore")
        assert response.status_code == 200
        assert response.json()["status"] == "restored"

    def test_list_sessions_with_pagination(self, client):
        for i in range(25):
            client.post("/api/sessions", json={"title": f"Session {i}"})

        response = client.get("/api/sessions?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 10
        assert data["has_more"] is True

    def test_list_sessions_with_search(self, client):
        client.post("/api/sessions", json={"title": "AI Research"})
        client.post("/api/sessions", json={"title": "ML Study"})

        response = client.get("/api/sessions?search=AI")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "AI" in data["sessions"][0]["title"]
