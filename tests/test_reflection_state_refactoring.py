# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from src.graph.types import State, ReflectionState


class TestReflectionStateRefactoring:
    """Test suite for the refactored reflection state management."""
    
    def test_reflection_state_creation(self):
        """Test that ReflectionState can be created with default values."""
        reflection_state = ReflectionState()
        
        assert reflection_state.enabled is False
        assert reflection_state.count == 0
        assert reflection_state.integration_active is False
        assert reflection_state.current_session is None
        assert reflection_state.triggered is False
        assert reflection_state.last_step == 0
        assert reflection_state.results == []
        assert reflection_state.knowledge_gaps == []
        assert reflection_state.follow_up_queries == []
        assert reflection_state.sufficiency_score == 0.0
    
    def test_state_with_modular_reflection(self):
        """Test that State class properly initializes with ReflectionState."""
        state = State()
        
        # Test that reflection is properly initialized
        assert isinstance(state.reflection, ReflectionState)
        assert state.reflection.enabled is False
        assert state.reflection.count == 0
    
    def test_backward_compatibility_properties(self):
        """Test that all backward compatibility properties work correctly."""
        state = State()
        
        # Test reflection_enabled
        state.reflection_enabled = True
        assert state.reflection_enabled is True
        assert state.reflection.enabled is True
        
        # Test reflection_count
        state.reflection_count = 5
        assert state.reflection_count == 5
        assert state.reflection.count == 5
        
        # Test knowledge_gaps
        gaps = ["gap1", "gap2"]
        state.knowledge_gaps = gaps
        assert state.knowledge_gaps == gaps
        assert state.reflection.knowledge_gaps == gaps
        
        # Test follow_up_queries
        queries = ["query1", "query2"]
        state.follow_up_queries = queries
        assert state.follow_up_queries == queries
        assert state.reflection.follow_up_queries == queries
        
        # Test research_sufficiency_score
        state.research_sufficiency_score = 0.85
        assert state.research_sufficiency_score == 0.85
        assert state.reflection.sufficiency_score == 0.85
        
        # Test current_reflection_session
        session_id = "session_123"
        state.current_reflection_session = session_id
        assert state.current_reflection_session == session_id
        assert state.reflection.current_session == session_id
        
        # Test last_reflection_step
        state.last_reflection_step = 3
        assert state.last_reflection_step == 3
        assert state.reflection.last_step == 3
        
        # Test reflection_triggered
        state.reflection_triggered = True
        assert state.reflection_triggered is True
        assert state.reflection.triggered is True
        
        # Test reflection_integration_active
        state.reflection_integration_active = True
        assert state.reflection_integration_active is True
        assert state.reflection.integration_active is True
    
    def test_reflection_results_compatibility(self):
        """Test that reflection_results property works correctly."""
        state = State()
        
        results = [
            {"step": 1, "gaps": ["gap1"]},
            {"step": 2, "gaps": ["gap2"]}
        ]
        
        state.reflection_results = results
        assert state.reflection_results == results
        assert state.reflection.results == results
    
    def test_direct_reflection_access(self):
        """Test that direct access to reflection object works."""
        state = State()
        
        # Direct modification through reflection object
        state.reflection.enabled = True
        state.reflection.count = 10
        state.reflection.knowledge_gaps.append("direct_gap")
        
        # Verify through backward compatibility properties
        assert state.reflection_enabled is True
        assert state.reflection_count == 10
        assert "direct_gap" in state.knowledge_gaps
    
    def test_reflection_state_serialization(self):
        """Test that ReflectionState can be properly serialized."""
        reflection_state = ReflectionState(
            enabled=True,
            count=3,
            knowledge_gaps=["gap1", "gap2"],
            sufficiency_score=0.75
        )
        
        # Test dict conversion
        state_dict = reflection_state.dict()
        assert state_dict["enabled"] is True
        assert state_dict["count"] == 3
        assert state_dict["knowledge_gaps"] == ["gap1", "gap2"]
        assert state_dict["sufficiency_score"] == 0.75
        
        # Test JSON serialization
        json_str = reflection_state.json()
        assert isinstance(json_str, str)
        assert "true" in json_str.lower()  # enabled: true
    
    def test_state_maintains_other_fields(self):
        """Test that State class maintains all other non-reflection fields."""
        state = State()
        
        # Test that all original fields are still present
        assert hasattr(state, "locale")
        assert hasattr(state, "research_topic")
        assert hasattr(state, "observations")
        assert hasattr(state, "resources")
        assert hasattr(state, "plan_iterations")
        assert hasattr(state, "current_plan")
        assert hasattr(state, "final_report")
        assert hasattr(state, "auto_accepted_plan")
        assert hasattr(state, "enable_background_investigation")
        assert hasattr(state, "background_investigation_results")
        assert hasattr(state, "enable_collaboration")
        assert hasattr(state, "collaboration_systems")
        
        # Test default values
        assert state.locale == "en-US"
        assert state.research_topic == ""
        assert state.observations == []
        assert state.resources == []
        assert state.plan_iterations == 0
        assert state.current_plan is None
        assert state.final_report == ""
        assert state.auto_accepted_plan is False
        assert state.enable_background_investigation is True
        assert state.background_investigation_results is None
        assert state.enable_collaboration is True
        assert state.collaboration_systems is None