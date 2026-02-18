# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.server.app import app
from src.server.browser_routes import (
    BrowserSession,
    BrowserState,
    BrowserAction,
    get_or_create_session,
    _browser_sessions,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_browser_sessions():
    _browser_sessions.clear()
    yield
    _browser_sessions.clear()


class TestBrowserModels:
    def test_browser_action_creation(self):
        action = BrowserAction(
            action="navigate",
            url="https://example.com",
        )
        assert action.action == "navigate"
        assert action.url == "https://example.com"

    def test_browser_state_creation(self):
        state = BrowserState(
            session_id="test-session",
            url="https://example.com",
            title="Example",
        )
        assert state.session_id == "test-session"
        assert state.url == "https://example.com"
        assert state.title == "Example"


class TestBrowserSession:
    @pytest.mark.asyncio
    async def test_browser_session_init(self):
        session = BrowserSession("test-id")
        assert session.session_id == "test-id"
        assert session._initialized is False

    @pytest.mark.asyncio
    async def test_browser_session_state(self):
        session = BrowserSession("test-id")
        assert session.state.session_id == "test-id"
        assert session.state.url == ""


class TestBrowserRoutes:
    def test_list_browser_sessions_empty(self, client):
        response = client.get("/api/browser")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_execute_browser_action(self, client):
        with patch("src.server.browser_routes.BrowserSession") as MockSession:
            mock_session = MagicMock()
            mock_session.execute_action = AsyncMock(
                return_value={
                    "output": "Action executed",
                    "error": None,
                    "base64_image": None,
                }
            )
            MockSession.return_value = mock_session

            response = client.post(
                "/api/browser/test-session/execute",
                json={"action": {"action": "navigate", "url": "https://example.com"}},
            )
            assert response.status_code == 200

    def test_get_browser_state(self, client):
        with patch("src.server.browser_routes.get_or_create_session") as mock_get:
            mock_session = MagicMock()
            mock_session.get_state = AsyncMock(
                return_value=BrowserState(
                    session_id="test-session",
                    url="https://example.com",
                    title="Example",
                )
            )
            mock_get.return_value = mock_session

            response = client.get("/api/browser/test-session/state")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-session"

    def test_get_browser_actions(self, client):
        with patch("src.server.browser_routes.get_or_create_session") as mock_get:
            mock_session = MagicMock()
            mock_session.state.actions = []
            mock_get.return_value = mock_session

            response = client.get("/api/browser/test-session/actions")
            assert response.status_code == 200
            data = response.json()
            assert "actions" in data

    def test_close_browser_session(self, client):
        _browser_sessions["test-session"] = MagicMock()

        response = client.delete("/api/browser/test-session")
        assert response.status_code == 200
        assert "test-session" not in _browser_sessions

    def test_close_browser_session_not_found(self, client):
        response = client.delete("/api/browser/non-existent")
        assert response.status_code == 404

    def test_get_screenshot(self, client):
        with patch("src.server.browser_routes.get_or_create_session") as mock_get:
            mock_session = MagicMock()
            mock_session.get_screenshot = AsyncMock(return_value="base64image")
            mock_get.return_value = mock_session

            response = client.get("/api/browser/test-session/screenshot")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestGetOrCreateSession:
    def test_creates_new_session(self):
        session = get_or_create_session("new-session-id")
        assert session.session_id == "new-session-id"
        assert "new-session-id" in _browser_sessions

    def test_returns_existing_session(self):
        existing = BrowserSession("existing-id")
        _browser_sessions["existing-id"] = existing

        session = get_or_create_session("existing-id")
        assert session is existing
