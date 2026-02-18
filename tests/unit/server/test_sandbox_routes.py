# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.server.app import app
from src.server.sandbox_routes import (
    SandboxInfo,
    SandboxMetrics,
    _sandbox_registry,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sandbox_registry():
    _sandbox_registry.clear()
    yield
    _sandbox_registry.clear()


class TestSandboxModels:
    def test_sandbox_info_creation(self):
        info = SandboxInfo(
            id="test-id",
            status="running",
            created_at="2024-01-01T00:00:00",
            image="python:3.12",
            cpu_limit=1.0,
            memory_limit="512m",
            network_enabled=False,
            work_dir="/workspace",
        )
        assert info.id == "test-id"
        assert info.status == "running"
        assert info.image == "python:3.12"

    def test_sandbox_metrics_creation(self):
        metrics = SandboxMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_used="300MB",
            memory_total="512MB",
        )
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0


class TestSandboxRoutes:
    def test_list_sandboxes_empty(self, client):
        response = client.get("/api/sandbox")
        assert response.status_code == 200
        data = response.json()
        assert data["sandboxes"] == []
        assert data["total"] == 0

    def test_create_sandbox(self, client):
        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_mgr.create_sandbox = AsyncMock(return_value="sandbox-123")
            mock_manager.return_value = mock_mgr

            response = client.post(
                "/api/sandbox",
                json={
                    "image": "python:3.12-slim",
                    "memory_limit": "512m",
                    "cpu_limit": 1.0,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "sandbox-123"
            assert data["status"] == "running"

    def test_get_sandbox_status(self, client):
        _sandbox_registry["test-sandbox"] = {
            "status": "running",
            "created_at": "2024-01-01T00:00:00",
            "image": "python:3.12",
            "config": {
                "cpu_limit": 1.0,
                "memory_limit": "512m",
                "network_enabled": False,
            },
        }

        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_sandbox = MagicMock()
            mock_sandbox.config.work_dir = "/workspace"
            mock_mgr.get_sandbox = AsyncMock(return_value=mock_sandbox)
            mock_manager.return_value = mock_mgr

            response = client.get("/api/sandbox/test-sandbox")
            assert response.status_code == 200
            data = response.json()
            assert data["info"]["id"] == "test-sandbox"

    def test_get_sandbox_status_not_found(self, client):
        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_mgr.get_sandbox = AsyncMock(side_effect=KeyError("not found"))
            mock_manager.return_value = mock_mgr

            response = client.get("/api/sandbox/non-existent")
            assert response.status_code == 404

    def test_delete_sandbox(self, client):
        _sandbox_registry["to-delete"] = {"status": "running"}

        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_mgr.delete_sandbox = AsyncMock()
            mock_manager.return_value = mock_mgr

            response = client.delete("/api/sandbox/to-delete")
            assert response.status_code == 200
            assert "to-delete" not in _sandbox_registry

    def test_delete_sandbox_not_found(self, client):
        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_mgr.delete_sandbox = AsyncMock(side_effect=KeyError("not found"))
            mock_manager.return_value = mock_mgr

            response = client.delete("/api/sandbox/non-existent")
            assert response.status_code == 404

    def test_execute_command(self, client):
        _sandbox_registry["test-sandbox"] = {"status": "running", "logs": []}

        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_sandbox = MagicMock()
            mock_sandbox.run_command = AsyncMock(return_value="output result")
            mock_mgr.get_sandbox = AsyncMock(return_value=mock_sandbox)
            mock_manager.return_value = mock_mgr

            response = client.post(
                "/api/sandbox/test-sandbox/execute",
                json={"command": "echo hello"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["output"] == "output result"
            assert data["exit_code"] == 0

    def test_list_files(self, client):
        _sandbox_registry["test-sandbox"] = {"status": "running"}

        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_sandbox = MagicMock()
            mock_sandbox.run_command = AsyncMock(
                return_value="total 0\ndrwxr-xr-x 2 root root 40 Jan 1 00:00 testdir\n-rw-r--r-- 1 root root 0 Jan 1 00:00 testfile"
            )
            mock_mgr.get_sandbox = AsyncMock(return_value=mock_sandbox)
            mock_manager.return_value = mock_mgr

            response = client.get("/api/sandbox/test-sandbox/files?path=/workspace")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2

    def test_get_sandbox_metrics(self, client):
        _sandbox_registry["test-sandbox"] = {"status": "running"}

        with patch("src.server.sandbox_routes.get_sandbox_manager") as mock_manager:
            mock_mgr = AsyncMock()
            mock_mgr.get_sandbox = AsyncMock(return_value=MagicMock())
            mock_manager.return_value = mock_mgr

            response = client.get("/api/sandbox/test-sandbox/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "cpu_usage" in data
            assert "memory_usage" in data
