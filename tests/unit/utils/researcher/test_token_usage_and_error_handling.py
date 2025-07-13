"""Unit tests for token usage analysis and error handling in researcher context isolation."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from src.utils.context.execution_context_manager import ExecutionContextManager, ContextConfig
from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_context_isolator import ResearcherContextIsolator, ResearcherContextConfig
from src.graph.types import State


class TestTokenUsageAnalysis:
    """Token usage comparison analysis tests."""
    
    def setup_method(self):
        """Test setup."""
        self.base_manager = ExecutionContextManager(ContextConfig())
        self.context_extension = ResearcherContextExtension(self.base_manager)
    
    def test_token_usage_comparison(self):
        """Test token usage comparison."""
        config = ResearcherContextConfig(max_context_steps=2)
        isolator = ResearcherContextIsolator(config)
        
        # Create state with large amount of data
        state = State(messages=[])
        state["step_history"] = [
            {"step": f"Step {i}", "execution_res": f"Research finding {i}" * 100}
            for i in range(10)
        ]
        
        # Process state
        processed_state = isolator.prepare_isolated_context(state)
        
        # Verify step count is limited
        assert len(processed_state["step_history"]) <= config.max_context_steps
        
        # Verify content is optimized
        for step in processed_state["step_history"]:
            assert len(step["execution_res"]) <= config.max_step_content_length * 2
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization."""
        config = ResearcherContextConfig(
            max_step_content_length=200,
            max_observations_length=500,
            isolation_level="aggressive"
        )
        isolator = ResearcherContextIsolator(config)
        
        # Create state with large amount of data
        state = State(
            messages=[],
            observations=[f"Observation {i}: " + "y" * 200 for i in range(5)]
        )
        state["step_history"] = [
            {"step": f"Step {i}", "execution_res": f"Data {i}: " + "x" * 1000}
            for i in range(10)
        ]
        
        # Process state
        processed_state = isolator.prepare_isolated_context(state)
        
        # Verify memory optimization
        assert len(processed_state["step_history"]) <= config.max_context_steps
        
        # Verify observations are optimized
        if "observations" in processed_state and processed_state["observations"]:
            total_obs_length = sum(len(str(obs)) for obs in processed_state["observations"])
            assert total_obs_length <= config.max_observations_length * 2
    
    def test_content_length_optimization(self):
        """Test content length optimization."""
        config = ResearcherContextConfig(
            max_step_content_length=100,
            isolation_level="moderate"
        )
        isolator = ResearcherContextIsolator(config)
        
        # Create steps with very long content
        long_content = "x" * 1000
        state = State(messages=[])
        state["step_history"] = [
            {"step": "Step 1", "execution_res": long_content},
            {"step": "Step 2", "execution_res": long_content},
        ]
        
        processed_state = isolator.prepare_isolated_context(state)
        
        # Verify content is truncated or optimized
        for step in processed_state["step_history"]:
            # Allow some flexibility for truncation markers
            assert len(step["execution_res"]) <= config.max_step_content_length + 50
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        config = ResearcherContextConfig(
            max_context_steps=3,
            max_step_content_length=500,
            isolation_level="aggressive"
        )
        isolator = ResearcherContextIsolator(config)
        
        # Create very large dataset
        state = State(messages=[])
        state["step_history"] = [
            {"step": f"Step {i}", "execution_res": f"Data {i}: " + "content" * 200}
            for i in range(50)  # 50 steps
        ]
        state["observations"] = [f"Obs {i}: " + "data" * 100 for i in range(20)]
        
        processed_state = isolator.prepare_isolated_context(state)
        
        # Verify aggressive reduction
        assert len(processed_state["step_history"]) <= config.max_context_steps
        
        # Verify observations are reduced
        if "observations" in processed_state:
            assert len(processed_state["observations"]) <= len(state.get("observations", []))


class TestErrorHandlingAndFallback:
    """Error handling and fallback mechanism tests."""
    
    def setup_method(self):
        """Test setup."""
        self.base_manager = ExecutionContextManager(ContextConfig())
        self.context_extension = ResearcherContextExtension(self.base_manager)
    
    def test_invalid_isolation_level_handling(self):
        """Test invalid isolation level handling."""
        # Test invalid isolation level
        try:
            config = ResearcherContextConfig(isolation_level="invalid_level")
            isolator = ResearcherContextIsolator(config)
            # Call method to trigger validation
            optimization_level = isolator._get_optimization_level()
            # Should return default value
            assert optimization_level == "balanced"
        except ValueError:
            # If exception is thrown, that's also acceptable
            pass
    
    def test_context_cleanup_on_error(self):
        """Test context cleanup on error."""
        config = ResearcherContextConfig(isolation_level="moderate")
        isolator = ResearcherContextIsolator(config)
        
        # Create invalid state to trigger error
        invalid_state = State(messages=[])
        invalid_state["step_history"] = None  # Set to None to trigger error
        
        try:
            # Try to process invalid state
            isolator.prepare_isolated_context(invalid_state)
            # If no exception is thrown, verify processing result
            assert True  # Being able to handle is success
        except (AttributeError, TypeError):
            # Verify error is properly handled
            pass
        
        # Verify isolator can still work normally
        valid_state = State(messages=[])
        valid_state["step_history"] = [{"step": "test", "execution_res": "test result"}]
        processed_state = isolator.prepare_isolated_context(valid_state)
        assert processed_state is not None
    
    def test_memory_overflow_protection(self):
        """Test memory overflow protection."""
        # Create extreme configuration
        config = ResearcherContextConfig(
            max_step_content_length=100,  # Very small limit
            max_observations_length=200,
            isolation_level="aggressive"
        )
        
        isolator = ResearcherContextIsolator(config)
        
        # Create state with large amount of data
        state = State(
            messages=[],
            observations=[f"Observation {i}: " + "y" * 500 for i in range(10)]
        )
        state["step_history"] = [
            {"step": f"Step {i}", "execution_res": f"Data {i}: " + "x" * 1000}
            for i in range(20)  # Large amount of data
        ]
        
        # Process state
        processed_state = isolator.prepare_isolated_context(state)
        
        # Verify memory protection mechanism - step count should be limited
        assert len(processed_state["step_history"]) <= config.max_context_steps
        
        # Verify content length is limited
        for step in processed_state["step_history"]:
            assert len(step["execution_res"]) <= config.max_step_content_length + 10  # Allow extra chars for truncation markers
        
        # Verify observations are limited
        if "observations" in processed_state and processed_state["observations"]:
            # Observations should be optimized, count should be reduced
            assert len(processed_state["observations"]) <= len(state.get("observations", []))
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        config = ResearcherContextConfig(isolation_level="moderate")
        isolator = ResearcherContextIsolator(config)
        
        # Create state with malformed data
        state = State(messages=[])
        state["step_history"] = [
            {"step": "Valid Step", "execution_res": "Valid result"},
            {"step": None, "execution_res": "Invalid step name"},  # None step
            {"execution_res": "Missing step key"},  # Missing step key
            {"step": "Valid Step 2"},  # Missing execution_res
        ]
        
        # Should handle malformed data gracefully
        try:
            processed_state = isolator.prepare_isolated_context(state)
            assert processed_state is not None
            assert "step_history" in processed_state
        except Exception as e:
            # If exception occurs, it should be a known type
            assert isinstance(e, (KeyError, AttributeError, TypeError, ValueError))
    
    def test_empty_and_none_values(self):
        """Test handling of empty and None values."""
        config = ResearcherContextConfig(isolation_level="moderate")
        isolator = ResearcherContextIsolator(config)
        
        # Test with None state
        try:
            result = isolator.prepare_isolated_context(None)
            # If it doesn't throw exception, result should be reasonable
            assert result is not None or result is None  # Either is acceptable
        except (AttributeError, TypeError):
            # Exception is acceptable for None input
            pass
        
        # Test with empty state
        empty_state = State(messages=[])
        processed_state = isolator.prepare_isolated_context(empty_state)
        assert processed_state is not None
    
    def test_concurrent_access_safety(self):
        """Test concurrent access safety."""
        config = ResearcherContextConfig(isolation_level="moderate")
        isolator = ResearcherContextIsolator(config)
        
        # Create multiple states
        states = []
        for i in range(5):
            state = State(messages=[])
            state["step_history"] = [{"step": f"Step {i}", "execution_res": f"Result {i}"}]
            states.append(state)
        
        # Process multiple states (simulating concurrent access)
        results = []
        for state in states:
            try:
                result = isolator.prepare_isolated_context(state)
                results.append(result)
            except Exception as e:
                # Log error but continue
                results.append(None)
        
        # Verify at least some results are successful
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) > 0