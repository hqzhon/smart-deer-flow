# -*- coding: utf-8 -*-
"""
Shared pytest fixtures and configuration for DeerFlow tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any


def create_test_config() -> Dict[str, Any]:
    """Create a test configuration dictionary."""
    return {
        "test_mode": True,
        "enable_enhanced_reflection": True,
        "max_reflection_loops": 2,
        "reflection_model": "gpt-4",
        "knowledge_gap_threshold": 0.6,
        "sufficiency_threshold": 0.7,
        "llm_provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "timeout": 30,
    }


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return create_test_config()


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=Mock(content="Test response"))
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.test_mode = True
    config.enable_enhanced_reflection = True
    config.max_reflection_loops = 2
    config.reflection_model = "gpt-4"
    config.knowledge_gap_threshold = 0.6
    config.sufficiency_threshold = 0.7
    return config


@pytest.fixture
def mock_metrics():
    """Create a mock metrics system."""
    metrics = Mock()
    metrics.record_reflection_event = Mock()
    metrics.record_research_event = Mock()
    metrics.record_planning_event = Mock()
    metrics.get_reflection_stats = Mock(
        return_value={
            "total_reflections": 5,
            "avg_confidence": 0.8,
            "gap_detection_rate": 0.7,
        }
    )
    return metrics


@pytest.fixture
def sample_research_context():
    """Create sample research context for testing."""
    return {
        "research_topic": "Test research topic",
        "completed_steps": [
            {"step": 1, "action": "Initial search", "result": "Found basic information"}
        ],
        "current_step": {"step": 2, "action": "Deep dive research"},
        "execution_results": ["Result 1", "Result 2"],
        "observations": ["Observation 1", "Observation 2"],
        "resources_found": 5,
        "total_steps": 3,
        "current_step_index": 1,
    }
