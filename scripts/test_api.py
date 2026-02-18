#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
API Test Script for DeerFlow Backend

This script tests all the new API endpoints added in Phase 1 and Phase 2:
- Session Management API
- Browser Automation API
- Data Analysis API
- Sandbox Status API
- Metrics Dashboard API

Usage:
    python scripts/test_api.py [--base-url URL] [--verbose]
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


class APITester:
    def __init__(self, base_url: str, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.created_sessions = []
        self.created_sandboxes = []

    def log(self, message: str, color: str = Colors.RESET):
        print(f"{color}{message}{Colors.RESET}")

    def log_verbose(self, message: str):
        if self.verbose:
            print(f"  {message}")

    def test_endpoint(
        self,
        method: str,
        endpoint: str,
        expected_status: int = 200,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        description: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Test a single API endpoint."""
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=json_data, timeout=30)
            elif method.upper() == "PATCH":
                response = requests.patch(url, json=json_data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if response.status_code == expected_status:
                self.passed += 1
                self.log(f"✓ {description or endpoint}", Colors.GREEN)
                self.log_verbose(f"  Status: {response.status_code}")
                try:
                    data = response.json()
                    self.log_verbose(f"  Response: {json.dumps(data, indent=2)[:200]}")
                    return data
                except Exception:
                    return {}
            else:
                self.failed += 1
                self.log(f"✗ {description or endpoint}", Colors.RED)
                self.log(
                    f"  Expected status {expected_status}, got {response.status_code}",
                    Colors.RED,
                )
                return None

        except requests.exceptions.ConnectionError:
            self.failed += 1
            self.log(f"✗ {description or endpoint} - Connection refused", Colors.RED)
            return None
        except requests.exceptions.Timeout:
            self.failed += 1
            self.log(f"✗ {description or endpoint} - Timeout", Colors.RED)
            return None
        except Exception as e:
            self.failed += 1
            self.log(f"✗ {description or endpoint} - Error: {e}", Colors.RED)
            return None

    def test_session_api(self):
        """Test Session Management API."""
        self.log("\n=== Session Management API ===", Colors.BLUE)

        # List sessions (empty)
        data = self.test_endpoint("GET", "/api/sessions", description="List sessions")

        # Create session
        data = self.test_endpoint(
            "POST",
            "/api/sessions",
            json_data={
                "title": "Test Session",
                "research_topic": "API Testing",
                "tags": ["test", "api"],
            },
            description="Create session",
        )
        if data and "id" in data:
            session_id = data["id"]
            self.created_sessions.append(session_id)

            # Get session
            self.test_endpoint(
                "GET", f"/api/sessions/{session_id}", description="Get session by ID"
            )

            # Update session
            self.test_endpoint(
                "PATCH",
                f"/api/sessions/{session_id}",
                json_data={"title": "Updated Session Title"},
                description="Update session",
            )

            # Archive session
            self.test_endpoint(
                "POST",
                f"/api/sessions/{session_id}/archive",
                description="Archive session",
            )

            # Restore session
            self.test_endpoint(
                "POST",
                f"/api/sessions/{session_id}/restore",
                description="Restore session",
            )

        # Test pagination
        self.test_endpoint(
            "GET",
            "/api/sessions",
            params={"page": 1, "page_size": 10},
            description="List sessions with pagination",
        )

        # Test search
        self.test_endpoint(
            "GET",
            "/api/sessions",
            params={"search": "test"},
            description="Search sessions",
        )

    def test_browser_api(self):
        """Test Browser Automation API."""
        self.log("\n=== Browser Automation API ===", Colors.BLUE)

        # List browser sessions
        self.test_endpoint("GET", "/api/browser", description="List browser sessions")

        # Get browser state (will create session)
        session_id = "test-browser-session"
        self.test_endpoint(
            "GET", f"/api/browser/{session_id}/state", description="Get browser state"
        )

        # Get screenshot
        self.test_endpoint(
            "GET",
            f"/api/browser/{session_id}/screenshot",
            expected_status=200,
            description="Get browser screenshot",
        )

        # Get actions
        self.test_endpoint(
            "GET",
            f"/api/browser/{session_id}/actions",
            description="Get browser actions",
        )

        # Close browser session
        self.test_endpoint(
            "DELETE", f"/api/browser/{session_id}", description="Close browser session"
        )

    def test_analysis_api(self):
        """Test Data Analysis API."""
        self.log("\n=== Data Analysis API ===", Colors.BLUE)

        # Create chart
        self.test_endpoint(
            "POST",
            "/api/analysis/chart",
            json_data={
                "chart_type": "bar",
                "title": "Test Chart",
                "data": [10, 20, 30, 40, 50],
                "labels": ["A", "B", "C", "D", "E"],
            },
            description="Create bar chart",
        )

        # Create line chart
        self.test_endpoint(
            "POST",
            "/api/analysis/chart",
            json_data={
                "chart_type": "line",
                "title": "Line Chart",
                "data": [1, 2, 3, 4, 5],
                "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
            },
            description="Create line chart",
        )

        # Calculate statistics
        self.test_endpoint(
            "POST",
            "/api/analysis/statistics",
            json_data={
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "include_percentiles": True,
            },
            description="Calculate statistics",
        )

        # Calculate correlation
        self.test_endpoint(
            "POST",
            "/api/analysis/correlation",
            json_data={
                "x_data": [1, 2, 3, 4, 5],
                "y_data": [2, 4, 6, 8, 10],
            },
            description="Calculate correlation",
        )

        # Analyze data
        self.test_endpoint(
            "POST",
            "/api/analysis/analyze",
            json_data={
                "data": [
                    {"name": "Alice", "age": 30, "score": 85},
                    {"name": "Bob", "age": 25, "score": 90},
                    {"name": "Charlie", "age": 35, "score": 78},
                ],
                "analysis_type": "summary",
            },
            description="Analyze data",
        )

    def test_sandbox_api(self):
        """Test Sandbox Status API."""
        self.log("\n=== Sandbox Status API ===", Colors.BLUE)

        # List sandboxes
        self.test_endpoint("GET", "/api/sandbox", description="List sandboxes")

        # Create sandbox
        data = self.test_endpoint(
            "POST",
            "/api/sandbox",
            json_data={
                "image": "python:3.12-slim",
                "memory_limit": "512m",
                "cpu_limit": 1.0,
            },
            description="Create sandbox",
        )

        if data and "id" in data:
            sandbox_id = data["id"]
            self.created_sandboxes.append(sandbox_id)

            # Get sandbox status
            self.test_endpoint(
                "GET", f"/api/sandbox/{sandbox_id}", description="Get sandbox status"
            )

            # Get sandbox metrics
            self.test_endpoint(
                "GET",
                f"/api/sandbox/{sandbox_id}/metrics",
                description="Get sandbox metrics",
            )

            # List files
            self.test_endpoint(
                "GET",
                f"/api/sandbox/{sandbox_id}/files",
                params={"path": "/workspace"},
                description="List sandbox files",
            )

    def test_metrics_api(self):
        """Test Metrics Dashboard API."""
        self.log("\n=== Metrics Dashboard API ===", Colors.BLUE)

        # Get system metrics
        self.test_endpoint(
            "GET", "/api/metrics/system", description="Get system metrics"
        )

        # Get LLM metrics
        self.test_endpoint("GET", "/api/metrics/llm", description="Get LLM metrics")

        # Get context metrics
        self.test_endpoint(
            "GET", "/api/metrics/context", description="Get context metrics"
        )

        # Get performance metrics
        self.test_endpoint(
            "GET", "/api/metrics/performance", description="Get performance metrics"
        )

        # Get metrics dashboard
        self.test_endpoint(
            "GET", "/api/metrics/dashboard", description="Get metrics dashboard"
        )

        # Get metrics history
        self.test_endpoint(
            "GET",
            "/api/metrics/history",
            params={"duration": 3600},
            description="Get metrics history",
        )

        # Record metric
        self.test_endpoint(
            "POST",
            "/api/metrics/record?metric_type=response_time&value=50.0",
            json_data={},
            description="Record metric",
        )

        # Record LLM usage
        self.test_endpoint(
            "POST",
            "/api/metrics/llm/record",
            json_data={
                "model": "test-model",
                "input_tokens": 100,
                "output_tokens": 50,
                "latency": 150.0,
                "cost": 0.01,
            },
            description="Record LLM usage",
        )

    def cleanup(self):
        """Clean up created resources."""
        self.log("\n=== Cleanup ===", Colors.YELLOW)

        for session_id in self.created_sessions:
            self.test_endpoint(
                "DELETE",
                f"/api/sessions/{session_id}",
                description=f"Delete session {session_id[:8]}",
            )

        for sandbox_id in self.created_sandboxes:
            self.test_endpoint(
                "DELETE",
                f"/api/sandbox/{sandbox_id}",
                description=f"Delete sandbox {sandbox_id[:8]}",
            )

    def run_all_tests(self):
        """Run all API tests."""
        self.log(f"\n{'=' * 50}", Colors.BLUE)
        self.log("DeerFlow API Test Suite", Colors.BLUE)
        self.log(f"Base URL: {self.base_url}", Colors.BLUE)
        self.log(f"{'=' * 50}", Colors.BLUE)

        start_time = time.time()

        self.test_session_api()
        self.test_browser_api()
        self.test_analysis_api()
        self.test_sandbox_api()
        self.test_metrics_api()
        self.cleanup()

        elapsed = time.time() - start_time

        self.log(f"\n{'=' * 50}", Colors.BLUE)
        self.log("Test Results", Colors.BLUE)
        self.log(f"{'=' * 50}", Colors.BLUE)
        self.log(f"Passed: {self.passed}", Colors.GREEN)
        self.log(f"Failed: {self.failed}", Colors.RED if self.failed else Colors.GREEN)
        self.log(f"Total: {self.passed + self.failed}", Colors.BLUE)
        self.log(f"Time: {elapsed:.2f}s", Colors.BLUE)

        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test DeerFlow API endpoints")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    tester = APITester(args.base_url, args.verbose)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
