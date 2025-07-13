"""Unit tests for ExecutionContextManager integration with researcher context isolation."""

import pytest
import sys
import os
from unittest.mock import Mock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from src.utils.context.execution_context_manager import ExecutionContextManager, ContextConfig
from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_context_isolator import ResearcherContextIsolator, ResearcherContextConfig
from src.graph.types import State


class TestExecutionContextManagerIntegration:
    """ExecutionContextManager integration tests."""
    
    def setup_method(self):
        """Test setup."""
        self.base_config = ContextConfig(
            max_context_steps=3,
            max_step_content_length=1000,
            max_observations_length=5000
        )
        self.base_manager = ExecutionContextManager(self.base_config)
        self.context_extension = ResearcherContextExtension(self.base_manager)
    
    def test_context_extension_initialization(self):
        """Test context extension initialization."""
        assert self.context_extension is not None
        assert self.context_extension.base_manager is not None
        assert self.context_extension.isolator is not None
        assert len(self.context_extension.active_isolators) == 0
    
    def test_isolated_context_creation_and_management(self):
        """Test isolated context creation and management."""
        config = ResearcherContextConfig(isolation_level="moderate")
        
        # Create isolator directly
        isolator = ResearcherContextIsolator(config)
        assert isolator is not None
        assert isinstance(isolator, ResearcherContextIsolator)
        
        # Verify configuration
        assert isolator.config.isolation_level == "moderate"
        
        # Test state processing
        state = State(messages=[])
        state["step_history"] = [{"step": "test", "execution_res": "test result"}]
        
        processed_state = isolator.prepare_isolated_context(state)
        assert processed_state is not None
        assert "step_history" in processed_state
        assert len(processed_state["step_history"]) <= config.max_context_steps
    
    def test_context_processing_with_base_manager(self):
        """Test context processing integration with base manager."""
        # Create test data
        completed_steps = [
            {"step": "Step 1", "execution_res": "Research result 1" * 50},
            {"step": "Step 2", "execution_res": "Research result 2" * 50},
        ]
        current_step = {"step": "Current Step", "description": "Current task"}
        
        # Process context using base manager
        optimized_steps, context_info = self.base_manager.prepare_context_for_execution(
            completed_steps, current_step, "researcher"
        )
        
        # Verify processing results
        assert len(optimized_steps) <= self.base_config.max_context_steps
        assert isinstance(context_info, str)
        assert "已完成" in context_info
    
    def test_observation_management_integration(self):
        """Test observation management integration."""
        observations = [
            "Initial observation",
            "Research finding 1",
            "Research finding 2",
        ]
        new_observation = "Research finding 3"
        
        # Manage observations using base manager
        managed_observations = self.base_manager.manage_observations(
            observations, new_observation
        )
        
        # Verify observations are properly managed
        assert len(managed_observations) >= len(observations)  # May be compressed
        # Verify new observation is added (may be modified)
        assert any("Research finding 3" in obs for obs in managed_observations)
        # Verify all observations have content
        assert all(len(obs) > 0 for obs in managed_observations)
    
    def test_integration_with_different_configs(self):
        """Test integration with different configurations."""
        configs = [
            ContextConfig(max_context_steps=1, max_step_content_length=500),
            ContextConfig(max_context_steps=5, max_step_content_length=2000),
            ContextConfig(max_context_steps=3, max_observations_length=1000),
        ]
        
        for config in configs:
            manager = ExecutionContextManager(config)
            extension = ResearcherContextExtension(manager)
            
            assert extension.base_manager.config.max_context_steps == config.max_context_steps
            assert extension.base_manager.config.max_step_content_length == config.max_step_content_length
    
    def test_state_processing_with_extension(self):
        """Test state processing through extension."""
        state = State(messages=[])
        state["step_history"] = [
            {"step": "Step 1", "execution_res": "Result 1" * 100},
            {"step": "Step 2", "execution_res": "Result 2" * 100},
            {"step": "Step 3", "execution_res": "Result 3" * 100},
        ]
        state["observations"] = ["Observation 1", "Observation 2"]
        
        # Process through extension's isolator
        processed_state = self.context_extension.isolator.prepare_isolated_context(state)
        
        assert "step_history" in processed_state
        assert "observations" in processed_state
        assert len(processed_state["step_history"]) <= self.context_extension.isolator.config.max_context_steps
    
    def test_multiple_extensions_isolation(self):
        """Test that multiple extensions maintain isolation."""
        extension1 = ResearcherContextExtension(self.base_manager)
        extension2 = ResearcherContextExtension(self.base_manager)
        
        # Verify they are separate instances
        assert extension1 is not extension2
        assert extension1.isolator is not extension2.isolator
        
        # Verify they can work independently
        state1 = State(messages=[])
        state1["step_history"] = [{"step": "Test 1", "execution_res": "Result 1"}]
        
        state2 = State(messages=[])
        state2["step_history"] = [{"step": "Test 2", "execution_res": "Result 2"}]
        
        processed1 = extension1.isolator.prepare_isolated_context(state1)
        processed2 = extension2.isolator.prepare_isolated_context(state2)
        
        assert processed1["step_history"][0]["step"] == "Test 1"
        assert processed2["step_history"][0]["step"] == "Test 2"