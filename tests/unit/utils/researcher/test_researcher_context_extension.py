"""Unit tests for ResearcherContextExtension."""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from src.utils.researcher.researcher_context_extension import ResearcherContextExtension
from src.utils.researcher.researcher_context_isolator import ResearcherContextConfig


class TestResearcherContextExtension:
    """Test cases for ResearcherContextExtension."""
    
    def test_initialization_default(self):
        """Test ResearcherContextExtension initialization with default config."""
        extension = ResearcherContextExtension()
        assert extension.isolator is not None
        assert extension.isolator.config.isolation_level == "moderate"  # default
    
    def test_initialization_custom_config(self):
        """Test ResearcherContextExtension initialization with custom config."""
        custom_config = ResearcherContextConfig(isolation_level="minimal")
        extension = ResearcherContextExtension(isolation_config=custom_config)
        assert extension.isolator is not None
        assert extension.isolator.config.isolation_level == "minimal"
    
    def test_isolator_integration(self):
        """Test that the extension properly integrates with the isolator."""
        extension = ResearcherContextExtension()
        
        # Test that we can access isolator methods
        completed_steps = [
            {'step': 'Step 1', 'description': 'First step', 'execution_res': 'Result 1'}
        ]
        current_step = {'step': 'Current Step', 'description': 'Current step description'}
        
        optimized_steps, context_info = extension.isolator.prepare_isolated_context(
            completed_steps, current_step, "researcher"
        )
        
        assert isinstance(optimized_steps, list)
        assert isinstance(context_info, str)
        assert "Research Guidelines" in context_info
    
    @pytest.mark.parametrize("isolation_level", ["minimal", "moderate", "aggressive"])
    def test_different_isolation_levels(self, isolation_level):
        """Test extension with different isolation levels."""
        config = ResearcherContextConfig(isolation_level=isolation_level)
        extension = ResearcherContextExtension(isolation_config=config)
        
        assert extension.isolator.config.isolation_level == isolation_level
    
    @pytest.mark.asyncio
    async def test_integration_with_mock_state(self):
        """Test integration with mock state and plan."""
        # Create mock state and config
        mock_state = {
            "current_plan": Mock(),
            "observations": ["Test observation 1", "Test observation 2"],
            "locale": "en-US",
            "resources": []
        }
        
        # Mock plan with steps
        mock_step = Mock()
        mock_step.title = "Test Research Step"
        mock_step.description = "Test research description"
        mock_step.execution_res = None
        
        completed_step = Mock()
        completed_step.title = "Completed Step"
        completed_step.description = "Completed description"
        completed_step.execution_res = "Completed result"
        
        mock_state["current_plan"].steps = [completed_step, mock_step]
        
        # Test context isolation
        extension = ResearcherContextExtension()
        
        # Test context preparation
        completed_steps = [{
            'step': completed_step.title,
            'description': completed_step.description,
            'execution_res': completed_step.execution_res
        }]
        
        current_step_dict = {
            'step': mock_step.title,
            'description': mock_step.description
        }
        
        optimized_steps, context_info = extension.isolator.prepare_isolated_context(
            completed_steps, current_step_dict, "researcher"
        )
        
        assert len(optimized_steps) >= 0
        assert isinstance(context_info, str)
        assert "Research Guidelines" in context_info
    
    def test_observation_management(self):
        """Test observation management through extension."""
        extension = ResearcherContextExtension()
        
        observations = ["Observation 1", "Observation 2", "Observation 3"]
        new_observation = "New observation"
        
        optimized_observations = extension.isolator.manage_isolated_observations(
            observations, new_observation
        )
        
        assert isinstance(optimized_observations, list)
        assert len(optimized_observations) >= 1
    
    def test_extension_with_different_max_steps(self):
        """Test extension with different max_context_steps configurations."""
        for max_steps in [1, 2, 3, 5]:
            config = ResearcherContextConfig(max_context_steps=max_steps)
            extension = ResearcherContextExtension(isolation_config=config)
            
            assert extension.isolator.config.max_context_steps == max_steps
            
            # Test with more steps than max_steps
            completed_steps = [
                {'step': f'Step {i}', 'description': f'Description {i}', 'execution_res': f'Result {i}'}
                for i in range(1, max_steps + 3)  # More than max_steps
            ]
            current_step = {'step': 'Current', 'description': 'Current description'}
            
            optimized_steps, context_info = extension.isolator.prepare_isolated_context(
                completed_steps, current_step, "researcher"
            )
            
            assert len(optimized_steps) <= max_steps
    
    def test_extension_immutability(self):
        """Test that extension doesn't modify original data."""
        extension = ResearcherContextExtension()
        
        original_steps = [
            {'step': 'Step 1', 'description': 'First step', 'execution_res': 'Result 1'},
            {'step': 'Step 2', 'description': 'Second step', 'execution_res': 'Result 2'}
        ]
        original_observations = ["Observation 1", "Observation 2"]
        
        # Make copies to compare
        steps_copy = [step.copy() for step in original_steps]
        observations_copy = original_observations.copy()
        
        current_step = {'step': 'Current', 'description': 'Current description'}
        
        # Process through extension
        extension.isolator.prepare_isolated_context(original_steps, current_step, "researcher")
        extension.isolator.manage_isolated_observations(original_observations, "New obs")
        
        # Original data should remain unchanged
        assert original_steps == steps_copy
        assert original_observations == observations_copy