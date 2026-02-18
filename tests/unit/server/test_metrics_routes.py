# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from fastapi.testclient import TestClient

from src.server.app import app
from src.server.metrics_routes import (
    SystemMetrics,
    LLMMetrics,
    ContextMetrics,
    PerformanceMetrics,
)


@pytest.fixture
def client():
    return TestClient(app)


class TestMetricsModels:
    def test_system_metrics_creation(self):
        metrics = SystemMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_used="4GB",
            memory_total="8GB",
            disk_usage=40.0,
            disk_used="100GB",
            disk_total="250GB",
        )
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0

    def test_llm_metrics_creation(self):
        metrics = LLMMetrics(
            total_requests=100,
            total_tokens=10000,
            input_tokens=6000,
            output_tokens=4000,
            total_cost=0.5,
            average_latency=150.0,
        )
        assert metrics.total_requests == 100
        assert metrics.total_tokens == 10000

    def test_context_metrics_creation(self):
        metrics = ContextMetrics(
            total_sessions=10,
            active_sessions=5,
            total_messages=100,
            compression_ratio=2.5,
        )
        assert metrics.total_sessions == 10
        assert metrics.active_sessions == 5

    def test_performance_metrics_creation(self):
        metrics = PerformanceMetrics(
            active_connections=10,
            max_connections=100,
            average_response_time=50.0,
            p95_response_time=100.0,
        )
        assert metrics.active_connections == 10
        assert metrics.average_response_time == 50.0


class TestMetricsRoutes:
    def test_get_system_metrics(self, client):
        response = client.get("/api/metrics/system")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_usage" in data
        assert "memory_usage" in data

    def test_get_llm_metrics(self, client):
        response = client.get("/api/metrics/llm")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "total_tokens" in data
        assert "error_rate" in data

    def test_get_context_metrics(self, client):
        response = client.get("/api/metrics/context")
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "active_sessions" in data

    def test_get_performance_metrics(self, client):
        response = client.get("/api/metrics/performance")
        assert response.status_code == 200
        data = response.json()
        assert "active_connections" in data
        assert "average_response_time" in data

    def test_get_metrics_dashboard(self, client):
        response = client.get("/api/metrics/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "llm" in data
        assert "context" in data
        assert "performance" in data
        assert "uptime" in data

    def test_get_metrics_history(self, client):
        response = client.get("/api/metrics/history?duration=3600")
        assert response.status_code == 200
        data = response.json()
        assert "timestamps" in data
        assert "cpu_usage" in data
        assert "memory_usage" in data

    def test_record_metric(self, client):
        response = client.post(
            "/api/metrics/record?metric_type=response_time&value=50.0",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"

    def test_record_llm_usage(self, client):
        response = client.post(
            "/api/metrics/llm/record?model=gpt-4&input_tokens=100&output_tokens=50&latency=150.0&cost=0.01&error=false",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"

    def test_record_llm_usage_with_error(self, client):
        response = client.post(
            "/api/metrics/llm/record?model=gpt-4&input_tokens=100&output_tokens=0&latency=100.0&cost=0.0&error=true",
        )
        assert response.status_code == 200

    def test_record_llm_usage_updates_models(self, client):
        response1 = client.post(
            "/api/metrics/llm/record?model=gpt-4&input_tokens=100&output_tokens=50&latency=150.0",
        )
        assert response1.status_code == 200

        response2 = client.post(
            "/api/metrics/llm/record?model=claude-3&input_tokens=200&output_tokens=100&latency=200.0",
        )
        assert response2.status_code == 200

        response = client.get("/api/metrics/llm")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] >= 2
