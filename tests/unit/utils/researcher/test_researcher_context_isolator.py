"""Unit tests for ResearcherContextIsolator."""

import pytest
import sys
import os
from unittest.mock import Mock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from src.utils.researcher.researcher_context_isolator import ResearcherContextIsolator, ResearcherContextConfig


class TestResearcherContextIsolator:
    """Test cases for ResearcherContextIsolator."""
    
    def test_default_configuration(self):
        """Test ResearcherContextIsolator with default configuration."""
        isolator = ResearcherContextIsolator()
        assert isolator.config.isolation_level == "moderate"
        assert isolator.config.max_context_steps == 2
    
    def test_custom_configuration(self):
        """Test ResearcherContextIsolator with custom configuration."""
        custom_config = ResearcherContextConfig(
            max_context_steps=1,
            isolation_level="aggressive"
        )
        isolator = ResearcherContextIsolator(custom_config)
        assert isolator.config.isolation_level == "aggressive"
        assert isolator.config.max_context_steps == 1
    
    def test_prepare_isolated_context(self):
        """Test context preparation functionality."""
        isolator = ResearcherContextIsolator()
        
        completed_steps = [
            {'step': 'Step 1', 'description': 'First step', 'execution_res': 'Result 1'},
            {'step': 'Step 2', 'description': 'Second step', 'execution_res': 'Result 2'},
            {'step': 'Step 3', 'description': 'Third step', 'execution_res': 'Result 3'},
        ]
        current_step = {'step': 'Current Step', 'description': 'Current step description'}
        
        optimized_steps, context_info = isolator.prepare_isolated_context(
            completed_steps, current_step, "researcher"
        )
        
        # Should limit steps based on max_context_steps (2)
        assert len(optimized_steps) <= 2
        assert "Research Guidelines" in context_info
        assert isinstance(context_info, str)
    
    def test_manage_isolated_observations(self):
        """Test observation management functionality."""
        isolator = ResearcherContextIsolator()
        
        observations = ["Observation 1", "Observation 2", "Observation 3"]
        new_observation = "New observation"
        
        optimized_observations = isolator.manage_isolated_observations(
            observations, new_observation
        )
        
        assert len(optimized_observations) >= 1  # Should have at least one observation
        assert isinstance(optimized_observations, list)
    
    @pytest.mark.parametrize("isolation_level,max_steps", [
        ("minimal", 3),
        ("moderate", 2),
        ("aggressive", 1),
    ])
    def test_isolation_levels(self, isolation_level, max_steps):
        """Test different isolation levels."""
        completed_steps = [
            {'step': f'Step {i}', 'description': f'Description {i}', 'execution_res': f'Result {i}'}
            for i in range(1, 6)  # 5 steps
        ]
        current_step = {'step': 'Current', 'description': 'Current description'}
        
        config = ResearcherContextConfig(isolation_level=isolation_level, max_context_steps=max_steps)
        isolator = ResearcherContextIsolator(config)
        steps, context_info = isolator.prepare_isolated_context(completed_steps, current_step)
        
        assert len(steps) <= max_steps
        assert isinstance(context_info, str)
        assert "Research Guidelines" in context_info
    
    def test_isolation_level_comparison(self):
        """Test that aggressive isolation produces fewer steps than moderate."""
        completed_steps = [
            {'step': f'Step {i}', 'description': f'Description {i}', 'execution_res': f'Result {i}'}
            for i in range(1, 6)  # 5 steps
        ]
        current_step = {'step': 'Current', 'description': 'Current description'}
        
        # Test moderate isolation
        config_moderate = ResearcherContextConfig(isolation_level="moderate", max_context_steps=2)
        isolator_moderate = ResearcherContextIsolator(config_moderate)
        steps_moderate, _ = isolator_moderate.prepare_isolated_context(completed_steps, current_step)
        
        # Test aggressive isolation
        config_aggressive = ResearcherContextConfig(isolation_level="aggressive", max_context_steps=2)
        isolator_aggressive = ResearcherContextIsolator(config_aggressive)
        steps_aggressive, _ = isolator_aggressive.prepare_isolated_context(completed_steps, current_step)
        
        # Aggressive should have fewer or equal steps than moderate
        assert len(steps_aggressive) <= len(steps_moderate)
    
    def test_empty_completed_steps(self):
        """Test behavior with empty completed steps."""
        isolator = ResearcherContextIsolator()
        
        completed_steps = []
        current_step = {'step': 'Current Step', 'description': 'Current step description'}
        
        optimized_steps, context_info = isolator.prepare_isolated_context(
            completed_steps, current_step, "researcher"
        )
        
        assert len(optimized_steps) == 0
        assert "Research Guidelines" in context_info
    
    def test_single_completed_step(self):
        """Test behavior with single completed step."""
        isolator = ResearcherContextIsolator()
        
        completed_steps = [
            {'step': 'Step 1', 'description': 'First step', 'execution_res': 'Result 1'}
        ]
        current_step = {'step': 'Current Step', 'description': 'Current step description'}
        
        optimized_steps, context_info = isolator.prepare_isolated_context(
            completed_steps, current_step, "researcher"
        )
        
        assert len(optimized_steps) <= isolator.config.max_context_steps
        assert "Research Guidelines" in context_info