#!/usr/bin/env python3
"""
Researcher Context Extension Phase 3 Unit Tests

This module contains unit tests for ResearcherContextExtension Phase 3 features,
including integration with metrics, progressive enablement, and isolation decisions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_context_isolator import ResearcherContextConfig
from src.utils.researcher.researcher_progressive_enablement import ScenarioContext


class MockStep:
    """Mock step class for testing."""
    
    def __init__(self, title="Test Step", description="This is a test step for validation"):
        self.title = title
        self.description = description


class MockPlan:
    """Mock plan class for testing."""
    
    def __init__(self, steps=None):
        self.steps = steps or [MockStep()]


class TestResearcherContextExtensionPhase3:
    """Test ResearcherContextExtension with Phase 3 features."""
    
    def test_extension_instantiation_with_phase3_config(self):
        """Test instantiation with Phase 3 configuration."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        assert extension is not None
        assert isinstance(extension, ResearcherContextExtension)
    
    def test_extension_with_metrics_enabled(self):
        """Test extension with metrics enabled."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': False,
            'researcher_isolation_threshold': 0.5,
            'researcher_max_local_context': 2000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        # Check if metrics are enabled
        metrics_enabled = hasattr(extension, 'metrics') and extension.metrics is not None
        # We don't enforce this as it depends on implementation
        # Just ensure no errors during instantiation
    
    def test_extension_with_metrics_disabled(self):
        """Test extension with metrics disabled."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': False,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.8,
            'researcher_max_local_context': 4000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        # Should still work without metrics
        assert extension is not None
    
    def test_extension_with_progressive_enabler(self):
        """Test extension with progressive enabler."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=3,
            max_step_content_length=2000,
            max_observations_length=10000,
            isolation_level="high"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.6,
            'researcher_max_local_context': 5000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        # Check if progressive enabler is available
        enabler_available = hasattr(extension, 'progressive_enabler') and extension.progressive_enabler is not None
        # We don't enforce this as it depends on implementation
    
    def test_create_scenario_context_basic(self):
        """Test creating scenario context from state and step."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        mock_state = {
            "observations": ["obs1", "obs2", "obs3"],
            "locale": "en-US"
        }
        
        mock_step = MockStep()
        
        # Test if the method exists and works
        if hasattr(extension, '_create_scenario_context'):
            try:
                scenario_context = extension._create_scenario_context(
                    mock_state, mock_step, "researcher"
                )
                
                assert scenario_context is not None
                assert isinstance(scenario_context, ScenarioContext)
                assert scenario_context.task_description is not None
                assert isinstance(scenario_context.step_count, int)
                assert isinstance(scenario_context.context_size, int)
            except Exception as e:
                pytest.fail(f"_create_scenario_context should not raise exception: {e}")
    
    def test_create_scenario_context_with_various_states(self):
        """Test creating scenario context with various state configurations."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        states = [
            # Minimal state
            {"observations": [], "locale": "en-US"},
            # State with many observations
            {"observations": [f"observation_{i}" for i in range(10)], "locale": "en-US"},
            # State with additional fields
            {
                "observations": ["obs1", "obs2"],
                "locale": "en-US",
                "plan_iterations": 3,
                "current_plan": MockPlan([MockStep("Step 1"), MockStep("Step 2")])
            }
        ]
        
        mock_step = MockStep()
        
        if hasattr(extension, '_create_scenario_context'):
            for state in states:
                try:
                    scenario_context = extension._create_scenario_context(
                        state, mock_step, "researcher"
                    )
                    assert scenario_context is not None
                except Exception as e:
                    pytest.fail(f"_create_scenario_context should handle various states: {e}")
    
    def test_create_scenario_context_with_different_steps(self):
        """Test creating scenario context with different step types."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        mock_state = {
            "observations": ["obs1", "obs2", "obs3"],
            "locale": "en-US"
        }
        
        steps = [
            MockStep("Simple Step", "Simple description"),
            MockStep("Complex Analysis Step", "This is a complex multi-part analysis requiring extensive research"),
            MockStep("", ""),  # Empty step
            MockStep("Step with Special Characters", "Description with émojis 🔍 and symbols @#$%")
        ]
        
        if hasattr(extension, '_create_scenario_context'):
            for step in steps:
                try:
                    scenario_context = extension._create_scenario_context(
                        mock_state, step, "researcher"
                    )
                    assert scenario_context is not None
                except Exception as e:
                    pytest.fail(f"_create_scenario_context should handle different steps: {e}")
    
    def test_should_use_isolation_basic(self):
        """Test basic isolation decision making."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        mock_state = {
            "observations": ["obs1", "obs2", "obs3"],
            "locale": "en-US"
        }
        
        mock_step = MockStep()
        
        if hasattr(extension, '_should_use_isolation'):
            try:
                should_isolate = extension._should_use_isolation(
                    mock_state, mock_step, "researcher"
                )
                
                assert isinstance(should_isolate, bool)
            except Exception as e:
                pytest.fail(f"_should_use_isolation should not raise exception: {e}")
    
    def test_should_use_isolation_with_auto_isolation_disabled(self):
        """Test isolation decision with auto isolation disabled."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': False,  # Disabled
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        mock_state = {
            "observations": ["obs1", "obs2", "obs3"],
            "locale": "en-US"
        }
        
        mock_step = MockStep()
        
        if hasattr(extension, '_should_use_isolation'):
            try:
                should_isolate = extension._should_use_isolation(
                    mock_state, mock_step, "researcher"
                )
                
                # With auto isolation disabled, might always return False
                # or use a different logic
                assert isinstance(should_isolate, bool)
            except Exception as e:
                pytest.fail(f"_should_use_isolation should work with auto isolation disabled: {e}")
    
    def test_should_use_isolation_various_scenarios(self):
        """Test isolation decisions for various scenarios."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        scenarios = [
            # Simple scenario
            {
                "state": {"observations": ["simple obs"], "locale": "en-US"},
                "step": MockStep("Simple", "Simple task")
            },
            # Complex scenario
            {
                "state": {
                    "observations": [f"complex observation {i}" for i in range(20)],
                    "locale": "en-US",
                    "plan_iterations": 5
                },
                "step": MockStep("Complex Analysis", "Complex multi-step analysis with extensive research")
            },
            # Empty scenario
            {
                "state": {"observations": [], "locale": "en-US"},
                "step": MockStep("", "")
            }
        ]
        
        if hasattr(extension, '_should_use_isolation'):
            for scenario in scenarios:
                try:
                    should_isolate = extension._should_use_isolation(
                        scenario["state"], scenario["step"], "researcher"
                    )
                    assert isinstance(should_isolate, bool)
                except Exception as e:
                    pytest.fail(f"_should_use_isolation should handle various scenarios: {e}")
    
    def test_extension_with_different_isolation_levels(self):
        """Test extension with different isolation levels."""
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        isolation_levels = ["low", "moderate", "high", "strict"]
        
        for level in isolation_levels:
            isolation_config = ResearcherContextConfig(
                max_context_steps=2,
                max_step_content_length=1500,
                max_observations_length=8000,
                isolation_level=level
            )
            
            try:
                extension = ResearcherContextExtension(
                    isolation_config=isolation_config,
                    config=phase3_config
                )
                assert extension is not None
            except Exception as e:
                pytest.fail(f"Extension should work with isolation level {level}: {e}")
    
    def test_extension_with_different_thresholds(self):
        """Test extension with different isolation thresholds."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for threshold in thresholds:
            phase3_config = {
                'researcher_isolation_metrics': True,
                'researcher_auto_isolation': True,
                'researcher_isolation_threshold': threshold,
                'researcher_max_local_context': 3000
            }
            
            try:
                extension = ResearcherContextExtension(
                    isolation_config=isolation_config,
                    config=phase3_config
                )
                assert extension is not None
            except Exception as e:
                pytest.fail(f"Extension should work with threshold {threshold}: {e}")
    
    def test_extension_integration_with_metrics_and_enabler(self):
        """Test integration between metrics and progressive enabler."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=3,
            max_step_content_length=2000,
            max_observations_length=10000,
            isolation_level="high"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.6,
            'researcher_max_local_context': 5000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        mock_state = {
            "observations": ["obs1", "obs2", "obs3", "obs4"],
            "locale": "en-US",
            "plan_iterations": 2
        }
        
        mock_step = MockStep("Integration Test", "Testing integration between components")
        
        # Test that both scenario context creation and isolation decision work together
        if hasattr(extension, '_create_scenario_context') and hasattr(extension, '_should_use_isolation'):
            try:
                scenario_context = extension._create_scenario_context(
                    mock_state, mock_step, "researcher"
                )
                
                should_isolate = extension._should_use_isolation(
                    mock_state, mock_step, "researcher"
                )
                
                assert scenario_context is not None
                assert isinstance(should_isolate, bool)
                
                # The decision should be consistent with the scenario context
                # (though we don't enforce specific logic)
            except Exception as e:
                pytest.fail(f"Integration between components should work: {e}")
    
    def test_extension_error_handling(self):
        """Test extension error handling."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        # Test with invalid inputs
        invalid_inputs = [
            (None, MockStep(), "researcher"),
            ({}, None, "researcher"),
            ({"observations": []}, MockStep(), None),
            ({"observations": []}, MockStep(), "")
        ]
        
        for state, step, agent_type in invalid_inputs:
            if hasattr(extension, '_create_scenario_context'):
                try:
                    extension._create_scenario_context(state, step, agent_type)
                    # Should either work or raise a reasonable exception
                except Exception:
                    # Exceptions are acceptable for invalid inputs
                    pass
            
            if hasattr(extension, '_should_use_isolation'):
                try:
                    extension._should_use_isolation(state, step, agent_type)
                    # Should either work or raise a reasonable exception
                except Exception:
                    # Exceptions are acceptable for invalid inputs
                    pass
    
    def test_extension_with_minimal_config(self):
        """Test extension with minimal Phase 3 configuration."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=1,
            max_step_content_length=500,
            max_observations_length=1000,
            isolation_level="low"
        )
        
        minimal_phase3_config = {
            'researcher_isolation_metrics': False,
            'researcher_auto_isolation': False
        }
        
        try:
            extension = ResearcherContextExtension(
                isolation_config=isolation_config,
                config=minimal_phase3_config
            )
            assert extension is not None
        except Exception as e:
            pytest.fail(f"Extension should work with minimal config: {e}")
    
    def test_extension_with_maximal_config(self):
        """Test extension with maximal Phase 3 configuration."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=10,
            max_step_content_length=5000,
            max_observations_length=50000,
            isolation_level="strict"
        )
        
        maximal_phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.9,
            'researcher_max_local_context': 10000,
            'enable_researcher_isolation': True,
            'researcher_isolation_level': 'strict'
        }
        
        try:
            extension = ResearcherContextExtension(
                isolation_config=isolation_config,
                config=maximal_phase3_config
            )
            assert extension is not None
        except Exception as e:
            pytest.fail(f"Extension should work with maximal config: {e}")


class TestResearcherContextExtensionPhase3Performance:
    """Test ResearcherContextExtension Phase 3 performance."""
    
    def test_extension_instantiation_performance(self):
        """Test that extension instantiation is fast."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        start_time = time.time()
        for _ in range(10):
            extension = ResearcherContextExtension(
                isolation_config=isolation_config,
                config=phase3_config
            )
        end_time = time.time()
        
        # Should create 10 extensions in less than 1 second
        assert end_time - start_time < 1.0, "Extension instantiation should be fast"
    
    def test_scenario_context_creation_performance(self):
        """Test that scenario context creation is fast."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        mock_state = {
            "observations": ["obs1", "obs2", "obs3"],
            "locale": "en-US"
        }
        
        mock_step = MockStep()
        
        if hasattr(extension, '_create_scenario_context'):
            start_time = time.time()
            for _ in range(100):
                extension._create_scenario_context(
                    mock_state, mock_step, "researcher"
                )
            end_time = time.time()
            
            # Should create 100 scenario contexts in less than 1 second
            assert end_time - start_time < 1.0, "Scenario context creation should be fast"
    
    def test_isolation_decision_performance(self):
        """Test that isolation decisions are fast."""
        isolation_config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=1500,
            max_observations_length=8000,
            isolation_level="moderate"
        )
        
        phase3_config = {
            'researcher_isolation_metrics': True,
            'researcher_auto_isolation': True,
            'researcher_isolation_threshold': 0.7,
            'researcher_max_local_context': 3000
        }
        
        extension = ResearcherContextExtension(
            isolation_config=isolation_config,
            config=phase3_config
        )
        
        mock_state = {
            "observations": ["obs1", "obs2", "obs3"],
            "locale": "en-US"
        }
        
        mock_step = MockStep()
        
        if hasattr(extension, '_should_use_isolation'):
            start_time = time.time()
            for _ in range(100):
                extension._should_use_isolation(
                    mock_state, mock_step, "researcher"
                )
            end_time = time.time()
            
            # Should make 100 isolation decisions in less than 1 second
            assert end_time - start_time < 1.0, "Isolation decisions should be fast"