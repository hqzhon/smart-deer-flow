# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from fastapi.testclient import TestClient

from src.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAnalysisRoutes:
    def test_create_chart_line(self, client):
        response = client.post(
            "/api/analysis/chart",
            json={
                "chart_type": "line",
                "title": "Test Chart",
                "data": [1, 2, 3, 4, 5],
                "labels": ["A", "B", "C", "D", "E"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chart_type"] == "line"
        assert data["title"] == "Test Chart"
        assert "config" in data
        assert "chart_data" in data

    def test_create_chart_bar(self, client):
        response = client.post(
            "/api/analysis/chart",
            json={
                "chart_type": "bar",
                "title": "Bar Chart",
                "data": [10, 20, 30],
                "labels": ["X", "Y", "Z"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chart_type"] == "bar"

    def test_create_chart_pie(self, client):
        response = client.post(
            "/api/analysis/chart",
            json={
                "chart_type": "pie",
                "title": "Pie Chart",
                "data": [30, 40, 30],
                "labels": ["Red", "Green", "Blue"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chart_type"] == "pie"

    def test_create_chart_invalid_type(self, client):
        response = client.post(
            "/api/analysis/chart",
            json={
                "chart_type": "invalid",
                "data": [1, 2, 3],
            },
        )
        assert response.status_code == 400

    def test_create_chart_with_series(self, client):
        response = client.post(
            "/api/analysis/chart",
            json={
                "chart_type": "line",
                "title": "Multi Series",
                "labels": ["Jan", "Feb", "Mar"],
                "series": [
                    {"name": "Series A", "data": [1, 2, 3]},
                    {"name": "Series B", "data": [4, 5, 6]},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "chart_data" in data

    def test_calculate_statistics(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "include_percentiles": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 10
        assert data["mean"] == 5.5
        assert data["median"] == 5.5
        assert data["min"] == 1
        assert data["max"] == 10
        assert "quartiles" in data
        assert "percentiles" in data

    def test_calculate_statistics_empty_data(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={"data": []},
        )
        assert response.status_code == 400

    def test_calculate_statistics_without_percentiles(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={
                "data": [1, 2, 3, 4, 5],
                "include_percentiles": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["quartiles"] is None
        assert data["percentiles"] is None

    def test_calculate_correlation(self, client):
        response = client.post(
            "/api/analysis/correlation",
            json={
                "x_data": [1, 2, 3, 4, 5],
                "y_data": [2, 4, 6, 8, 10],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pearson"] == 1.0
        assert data["sample_size"] == 5
        assert "interpretation" in data

    def test_calculate_correlation_negative(self, client):
        response = client.post(
            "/api/analysis/correlation",
            json={
                "x_data": [1, 2, 3, 4, 5],
                "y_data": [10, 8, 6, 4, 2],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pearson"] == -1.0

    def test_calculate_correlation_mismatched_length(self, client):
        response = client.post(
            "/api/analysis/correlation",
            json={
                "x_data": [1, 2, 3],
                "y_data": [1, 2],
            },
        )
        assert response.status_code == 400

    def test_calculate_correlation_too_few_points(self, client):
        response = client.post(
            "/api/analysis/correlation",
            json={
                "x_data": [1],
                "y_data": [1],
            },
        )
        assert response.status_code == 400

    def test_analyze_data_summary(self, client):
        response = client.post(
            "/api/analysis/analyze",
            json={
                "data": [
                    {"name": "Alice", "age": 30, "score": 85},
                    {"name": "Bob", "age": 25, "score": 90},
                    {"name": "Charlie", "age": 35, "score": 78},
                ],
                "analysis_type": "summary",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total_records"] == 3
        assert "statistics" in data

    def test_analyze_data_with_columns(self, client):
        response = client.post(
            "/api/analysis/analyze",
            json={
                "data": [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 6},
                ],
                "columns": ["a", "b"],
                "analysis_type": "summary",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "a" in data["statistics"]
        assert "b" in data["statistics"]

    def test_analyze_data_empty(self, client):
        response = client.post(
            "/api/analysis/analyze",
            json={"data": []},
        )
        assert response.status_code == 400

    def test_analyze_data_correlation(self, client):
        response = client.post(
            "/api/analysis/analyze",
            json={
                "data": [
                    {"x": 1, "y": 2},
                    {"x": 2, "y": 4},
                    {"x": 3, "y": 6},
                ],
                "analysis_type": "correlation",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "correlations" in data


class TestStatisticsCalculations:
    def test_mean_calculation(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={"data": [2, 4, 6, 8, 10]},
        )
        data = response.json()
        assert data["mean"] == 6.0

    def test_median_odd_count(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={"data": [1, 3, 5, 7, 9]},
        )
        data = response.json()
        assert data["median"] == 5.0

    def test_median_even_count(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={"data": [1, 2, 3, 4]},
        )
        data = response.json()
        assert data["median"] == 2.5

    def test_std_calculation(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={"data": [2, 4, 4, 4, 5, 5, 7, 9]},
        )
        data = response.json()
        assert abs(data["std"] - 2.0) < 0.01

    def test_range_calculation(self, client):
        response = client.post(
            "/api/analysis/statistics",
            json={"data": [5, 10, 15, 20]},
        )
        data = response.json()
        assert data["range"] == 15.0
        assert data["min"] == 5.0
        assert data["max"] == 20.0
