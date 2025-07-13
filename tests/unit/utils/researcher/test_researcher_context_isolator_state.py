"""Unit tests for ResearcherContextIsolator state processing."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from src.utils.researcher.researcher_context_isolator import ResearcherContextIsolator, ResearcherContextConfig
from src.graph.types import State


class TestResearcherContextIsolatorState:
    """Test cases for ResearcherContextIsolator state processing."""
    
    def setup_method(self):
        """Test setup."""
        self.config = ResearcherContextConfig(
            max_context_steps=2,
            max_step_content_length=500,
            isolation_level="moderate"
        )
        self.isolator = ResearcherContextIsolator(self.config)
    
    def test_state_processing(self):
        """Test State object processing."""
        state = State(
            messages=[],
            observations=["Observation 1", "Observation 2"]
        )
        # Add step_history as dictionary key
        state["step_history"] = [
            {"step": "Step 1", "execution_res": "Result 1" * 100},
            {"step": "Step 2", "execution_res": "Result 2" * 100},
            {"step": "Step 3", "execution_res": "Result 3" * 100},
        ]
        
        processed_state = self.isolator.prepare_isolated_context(state)
        
        # Verify step count is limited
        assert len(processed_state["step_history"]) <= self.config.max_context_steps
        
        # Verify observations are preserved
        assert "observations" in processed_state
        assert len(processed_state["observations"]) > 0
    
    def test_legacy_list_processing(self):
        """Test legacy list processing."""
        completed_steps = [
            {"step": "Step 1", "execution_res": "Result 1"},
            {"step": "Step 2", "execution_res": "Result 2"},
        ]
        current_step = {"step": "Current", "description": "Current task"}
        
        optimized_steps, context_info = self.isolator.prepare_isolated_context(
            completed_steps, current_step
        )
        
        assert isinstance(optimized_steps, list)
        assert isinstance(context_info, str)
        assert len(optimized_steps) <= self.config.max_context_steps
    
    def test_isolation_levels_filtering(self):
        """Test different isolation levels filtering."""
        test_steps = [
            {"step": f"Step {i}", "execution_res": f"Result {i}"}
            for i in range(5)
        ]
        
        # Test minimal isolation
        minimal_config = ResearcherContextConfig(isolation_level="minimal", max_context_steps=2)
        minimal_isolator = ResearcherContextIsolator(minimal_config)
        filtered_minimal = minimal_isolator._apply_isolation_filtering(test_steps)
        assert len(filtered_minimal) == 4  # max_context_steps * 2
        
        # Test aggressive isolation
        aggressive_config = ResearcherContextConfig(isolation_level="aggressive")
        aggressive_isolator = ResearcherContextIsolator(aggressive_config)
        filtered_aggressive = aggressive_isolator._apply_isolation_filtering(test_steps)
        assert len(filtered_aggressive) == 1  # Only keep the last one
        
        # Test moderate isolation
        moderate_config = ResearcherContextConfig(isolation_level="moderate", max_context_steps=2)
        moderate_isolator = ResearcherContextIsolator(moderate_config)
        filtered_moderate = moderate_isolator._apply_isolation_filtering(test_steps)
        assert len(filtered_moderate) == 2  # max_context_steps
    
    def test_state_with_empty_step_history(self):
        """Test state processing with empty step history."""
        state = State(messages=[], observations=["Test observation"])
        state["step_history"] = []
        
        processed_state = self.isolator.prepare_isolated_context(state)
        
        assert "step_history" in processed_state
        assert len(processed_state["step_history"]) == 0
        assert "observations" in processed_state
    
    def test_state_with_no_step_history(self):
        """Test state processing without step_history key."""
        state = State(messages=[], observations=["Test observation"])
        
        processed_state = self.isolator.prepare_isolated_context(state)
        
        # Should handle missing step_history gracefully
        assert processed_state is not None
        assert "observations" in processed_state
    
    def test_state_with_large_content(self):
        """Test state processing with large content."""
        state = State(messages=[], observations=["Test observation"])
        state["step_history"] = [
            {"step": f"Step {i}", "execution_res": "x" * 1000}  # Large content
            for i in range(10)
        ]
        
        processed_state = self.isolator.prepare_isolated_context(state)
        
        # Verify step count is limited
        assert len(processed_state["step_history"]) <= self.config.max_context_steps
        
        # Verify content length is managed
        for step in processed_state["step_history"]:
            # Content may be truncated but should not exceed reasonable limits
            assert len(step["execution_res"]) <= self.config.max_step_content_length * 2
    
    def test_state_immutability(self):
        """Test that original state is not modified."""
        original_observations = ["Observation 1", "Observation 2"]
        original_steps = [
            {"step": "Step 1", "execution_res": "Result 1"},
            {"step": "Step 2", "execution_res": "Result 2"},
        ]
        
        state = State(messages=[], observations=original_observations.copy())
        state["step_history"] = [step.copy() for step in original_steps]
        
        # Store original values for comparison
        original_obs_copy = original_observations.copy()
        original_steps_copy = [step.copy() for step in original_steps]
        
        # Process state
        processed_state = self.isolator.prepare_isolated_context(state)
        
        # Verify original state is unchanged
        assert state.get("observations") == original_obs_copy
        assert state["step_history"] == original_steps_copy
    
    @pytest.mark.parametrize("isolation_level,expected_behavior", [
        ("minimal", "preserves_more_context"),
        ("moderate", "balanced_context"),
        ("aggressive", "minimal_context"),
    ])
    def test_isolation_level_behaviors(self, isolation_level, expected_behavior):
        """Test different isolation level behaviors."""
        config = ResearcherContextConfig(
            isolation_level=isolation_level,
            max_context_steps=3
        )
        isolator = ResearcherContextIsolator(config)
        
        # Create test data with many steps
        test_steps = [
            {"step": f"Step {i}", "execution_res": f"Result {i}"}
            for i in range(10)
        ]
        
        filtered_steps = isolator._apply_isolation_filtering(test_steps)
        
        if expected_behavior == "preserves_more_context":
            # Minimal should preserve more context
            assert len(filtered_steps) >= 3
        elif expected_behavior == "balanced_context":
            # Moderate should be balanced
            assert 1 <= len(filtered_steps) <= 6
        elif expected_behavior == "minimal_context":
            # Aggressive should preserve minimal context
            assert len(filtered_steps) <= 3
    
    def test_state_with_mixed_data_types(self):
        """Test state processing with mixed data types."""
        state = State(messages=[])
        state["step_history"] = [
            {"step": "Step 1", "execution_res": "String result"},
            {"step": "Step 2", "execution_res": 12345},  # Number
            {"step": "Step 3", "execution_res": ["list", "result"]},  # List
            {"step": "Step 4", "execution_res": {"dict": "result"}},  # Dict
        ]
        state["observations"] = ["Text obs", 123, ["list", "obs"]]
        
        # Should handle mixed data types gracefully
        processed_state = self.isolator.prepare_isolated_context(state)
        
        assert "step_history" in processed_state
        assert "observations" in processed_state
        assert len(processed_state["step_history"]) <= self.config.max_context_steps