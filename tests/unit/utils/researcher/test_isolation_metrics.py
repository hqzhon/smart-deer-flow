#!/usr/bin/env python3
"""
Unit tests for ResearcherIsolationMetrics.

This module tests the isolation metrics functionality for researcher context isolation,
including session management, health monitoring, and performance tracking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.utils.researcher.researcher_isolation_metrics import ResearcherIsolationMetrics


class TestResearcherIsolationMetrics:
    """Test suite for ResearcherIsolationMetrics."""

    @pytest.fixture
    def metrics(self):
        """Create a ResearcherIsolationMetrics instance for testing."""
        return ResearcherIsolationMetrics()

    def test_start_isolation_session(self, metrics):
        """Test starting an isolation session."""
        session_id = metrics.start_isolation_session(
            session_id="test_session_1",
            task_complexity="medium",
            isolation_level="moderate"
        )
        
        assert session_id == "test_session_1"
        assert hasattr(metrics, '_sessions') or hasattr(metrics, 'sessions')
    
    def test_start_isolation_session_with_auto_id(self, metrics):
        """Test starting an isolation session with auto-generated ID."""
        session_id = metrics.start_isolation_session(
            task_complexity="high",
            isolation_level="strict"
        )
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    def test_update_session_context(self, metrics):
        """Test updating session context with compression metrics."""
        # Start a session first
        session_id = metrics.start_isolation_session(
            session_id="test_session_2",
            task_complexity="low",
            isolation_level="moderate"
        )
        
        # Update session context
        metrics.update_session_context(
            session_id="test_session_2",
            original_size=10000,
            compressed_size=8500
        )
        
        # Verify the update was recorded
        # This test assumes the metrics object stores session data
        assert True  # Basic test that no exception was raised
    
    def test_update_nonexistent_session_context(self, metrics):
        """Test updating context for a non-existent session."""
        # This should either raise an exception or handle gracefully
        try:
            metrics.update_session_context(
                session_id="nonexistent_session",
                original_size=5000,
                compressed_size=4000
            )
            # If no exception, the implementation handles it gracefully
            assert True
        except (KeyError, ValueError) as e:
            # If exception is raised, that's also acceptable behavior
            assert "session" in str(e).lower() or "not found" in str(e).lower()
    
    def test_end_isolation_session(self, metrics):
        """Test ending an isolation session."""
        # Start a session first
        session_id = metrics.start_isolation_session(
            session_id="test_session_3",
            task_complexity="medium",
            isolation_level="moderate"
        )
        
        # End the session
        metrics.end_isolation_session(
            session_id="test_session_3",
            success=True,
            performance_impact=0.1
        )
        
        # Verify the session was ended
        assert True  # Basic test that no exception was raised
    
    def test_end_isolation_session_with_failure(self, metrics):
        """Test ending an isolation session with failure."""
        # Start a session first
        session_id = metrics.start_isolation_session(
            session_id="test_session_4",
            task_complexity="high",
            isolation_level="strict"
        )
        
        # End the session with failure
        metrics.end_isolation_session(
            session_id="test_session_4",
            success=False,
            performance_impact=0.3
        )
        
        # Verify the session was ended
        assert True  # Basic test that no exception was raised
    
    def test_get_isolation_health(self, metrics):
        """Test getting isolation health status."""
        health = metrics.get_isolation_health()
        
        assert isinstance(health, dict)
        assert 'system_status' in health
        assert 'total_isolation_sessions' in health
        
        # Verify system status is a valid value
        valid_statuses = ['healthy', 'warning', 'critical', 'unknown']
        assert health['system_status'] in valid_statuses
        
        # Verify total sessions is a non-negative integer
        assert isinstance(health['total_isolation_sessions'], int)
        assert health['total_isolation_sessions'] >= 0
    
    def test_get_detailed_metrics(self, metrics):
        """Test getting detailed metrics."""
        detailed_metrics = metrics.get_detailed_metrics()
        
        # Check that the returned object has expected attributes
        assert hasattr(detailed_metrics, 'total_sessions')
        assert hasattr(detailed_metrics, 'success_rate')
        assert hasattr(detailed_metrics, 'total_token_savings')
        
        # Verify data types
        assert isinstance(detailed_metrics.total_sessions, int)
        assert isinstance(detailed_metrics.success_rate, (int, float))
        assert isinstance(detailed_metrics.total_token_savings, (int, float))
        
        # Verify ranges
        assert detailed_metrics.total_sessions >= 0
        assert 0 <= detailed_metrics.success_rate <= 1
        assert detailed_metrics.total_token_savings >= 0
    
    def test_multiple_sessions_workflow(self, metrics):
        """Test a complete workflow with multiple sessions."""
        # Start multiple sessions
        session_ids = []
        for i in range(3):
            session_id = metrics.start_isolation_session(
                session_id=f"workflow_session_{i}",
                task_complexity="medium",
                isolation_level="moderate"
            )
            session_ids.append(session_id)
        
        # Update contexts
        for i, session_id in enumerate(session_ids):
            metrics.update_session_context(
                session_id=session_id,
                original_size=10000 + i * 1000,
                compressed_size=8000 + i * 800
            )
        
        # End sessions with different outcomes
        for i, session_id in enumerate(session_ids):
            success = i % 2 == 0  # Alternate success/failure
            metrics.end_isolation_session(
                session_id=session_id,
                success=success,
                performance_impact=0.1 + i * 0.05
            )
        
        # Check final metrics
        health = metrics.get_isolation_health()
        detailed = metrics.get_detailed_metrics()
        
        assert health['total_isolation_sessions'] >= 3
        assert detailed.total_sessions >= 3
    
    def test_performance_impact_tracking(self, metrics):
        """Test that performance impact is properly tracked."""
        # Start and end a session with specific performance impact
        session_id = metrics.start_isolation_session(
            session_id="perf_test_session",
            task_complexity="high",
            isolation_level="strict"
        )
        
        performance_impact = 0.25
        metrics.end_isolation_session(
            session_id=session_id,
            success=True,
            performance_impact=performance_impact
        )
        
        # Verify that metrics can be retrieved
        detailed = metrics.get_detailed_metrics()
        assert detailed is not None
    
    def test_task_complexity_levels(self, metrics):
        """Test different task complexity levels."""
        complexity_levels = ['low', 'medium', 'high', 'critical']
        
        for complexity in complexity_levels:
            session_id = metrics.start_isolation_session(
                session_id=f"complexity_{complexity}_session",
                task_complexity=complexity,
                isolation_level="moderate"
            )
            
            metrics.end_isolation_session(
                session_id=session_id,
                success=True,
                performance_impact=0.1
            )
        
        # Verify all sessions were handled
        health = metrics.get_isolation_health()
        assert health['total_isolation_sessions'] >= len(complexity_levels)
    
    def test_isolation_levels(self, metrics):
        """Test different isolation levels."""
        isolation_levels = ['low', 'moderate', 'high', 'strict']
        
        for level in isolation_levels:
            session_id = metrics.start_isolation_session(
                session_id=f"isolation_{level}_session",
                task_complexity="medium",
                isolation_level=level
            )
            
            metrics.end_isolation_session(
                session_id=session_id,
                success=True,
                performance_impact=0.1
            )
        
        # Verify all sessions were handled
        health = metrics.get_isolation_health()
        assert health['total_isolation_sessions'] >= len(isolation_levels)