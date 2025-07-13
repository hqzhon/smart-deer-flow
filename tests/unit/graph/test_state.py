"""Unit tests for State class functionality."""

import pytest
import sys
import os
from typing import Annotated

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Import MessagesState directly from langgraph
from langgraph.graph import MessagesState


# Create stub versions of Plan/Step/StepType to avoid dependencies
class StepType:
    """Step type enumeration."""
    RESEARCH = "research"
    PROCESSING = "processing"


class Step:
    """Step class for testing."""
    
    def __init__(self, need_search, title, description, step_type):
        self.need_search = need_search
        self.title = title
        self.description = description
        self.step_type = step_type


class Plan:
    """Plan class for testing."""
    
    def __init__(self, locale, has_enough_context, thought, title, steps):
        self.locale = locale
        self.has_enough_context = has_enough_context
        self.thought = thought
        self.title = title
        self.steps = steps


# Import the actual State class by loading the module directly
def load_state_class():
    """Load State class from types.py to avoid import loops."""
    # Get the absolute path to the types.py file
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    types_path = os.path.join(src_dir, "graph", "types.py")
    
    # Create a namespace for the module
    import types
    
    module_name = "src.graph.types_direct"
    spec = types.ModuleType(module_name)
    
    # Add the module to sys.modules to avoid import loops
    sys.modules[module_name] = spec
    
    # Set up the namespace with required imports
    spec.__dict__["operator"] = __import__("operator")
    spec.__dict__["Annotated"] = Annotated
    spec.__dict__["MessagesState"] = MessagesState
    spec.__dict__["Plan"] = Plan
    
    # Execute the module code
    with open(types_path, "r", encoding='utf-8') as f:
        module_code = f.read()
    
    exec(module_code, spec.__dict__)
    
    # Return the State class
    return spec.State


# Load the actual State class
State = load_state_class()


class TestStateInitialization:
    """Test cases for State class initialization."""
    
    def test_state_class_attributes(self):
        """Test that State class has correct default attribute definitions."""
        # Test that the class has the expected attribute definitions
        assert hasattr(State, 'locale')
        assert hasattr(State, 'observations')
        assert hasattr(State, 'plan_iterations')
        assert hasattr(State, 'current_plan')
        assert hasattr(State, 'final_report')
        assert hasattr(State, 'auto_accepted_plan')
        assert hasattr(State, 'enable_background_investigation')
        assert hasattr(State, 'background_investigation_results')
        
        # Test default values
        assert State.locale == "en-US"
        assert State.observations == []
        assert State.plan_iterations == 0
        assert State.current_plan is None
        assert State.final_report == ""
        assert State.auto_accepted_plan is False
        assert State.enable_background_investigation is True
        assert State.background_investigation_results is None
    
    def test_state_basic_initialization(self):
        """Test basic State initialization."""
        # Verify state initialization with required messages field
        state = State(messages=[])
        assert "messages" in state
        assert state["messages"] == []
        
        # Without explicitly passing attributes, they're not in the state
        assert "locale" not in state
        assert "observations" not in state
        assert "plan_iterations" not in state
        assert "current_plan" not in state
        assert "final_report" not in state
        assert "auto_accepted_plan" not in state
        assert "enable_background_investigation" not in state
        assert "background_investigation_results" not in state
    
    def test_state_initialization_with_messages(self):
        """Test State initialization with messages."""
        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        state = State(messages=test_messages)
        assert "messages" in state
        assert state["messages"] == test_messages
        assert len(state["messages"]) == 2
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][1]["role"] == "assistant"
    
    def test_state_inheritance_from_messages_state(self):
        """Test that State properly inherits from MessagesState."""
        state = State(messages=[])
        
        # Should be an instance of MessagesState
        assert isinstance(state, MessagesState)
        
        # Should have MessagesState functionality
        assert hasattr(state, 'messages')
        assert callable(getattr(state, 'get', None))
        assert callable(getattr(state, 'keys', None))
        assert callable(getattr(state, 'values', None))
        assert callable(getattr(state, 'items', None))


class TestStateCustomValues:
    """Test cases for State with custom values."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_step = Step(
            need_search=True,
            title="Test Step",
            description="Step description",
            step_type=StepType.RESEARCH,
        )
        
        self.test_plan = Plan(
            locale="en-US",
            has_enough_context=False,
            thought="Test thought",
            title="Test Plan",
            steps=[self.test_step],
        )
    
    def test_state_with_custom_locale(self):
        """Test State with custom locale."""
        state = State(
            messages=[],
            locale="fr-FR"
        )
        
        assert "locale" in state
        assert state["locale"] == "fr-FR"
    
    def test_state_with_custom_observations(self):
        """Test State with custom observations."""
        observations = ["Observation 1", "Observation 2"]
        state = State(
            messages=[],
            observations=observations
        )
        
        assert "observations" in state
        assert state["observations"] == observations
        assert len(state["observations"]) == 2
        assert state["observations"][0] == "Observation 1"
    
    def test_state_with_plan_iterations(self):
        """Test State with custom plan iterations."""
        state = State(
            messages=[],
            plan_iterations=5
        )
        
        assert "plan_iterations" in state
        assert state["plan_iterations"] == 5
        assert isinstance(state["plan_iterations"], int)
    
    def test_state_with_current_plan(self):
        """Test State with current plan."""
        state = State(
            messages=[],
            current_plan=self.test_plan
        )
        
        assert "current_plan" in state
        assert state["current_plan"] is not None
        assert state["current_plan"].title == "Test Plan"
        assert state["current_plan"].thought == "Test thought"
        assert state["current_plan"].locale == "en-US"
        assert state["current_plan"].has_enough_context is False
        assert len(state["current_plan"].steps) == 1
        assert state["current_plan"].steps[0].title == "Test Step"
        assert state["current_plan"].steps[0].need_search is True
        assert state["current_plan"].steps[0].step_type == StepType.RESEARCH
    
    def test_state_with_final_report(self):
        """Test State with final report."""
        report = "This is a test final report with detailed findings."
        state = State(
            messages=[],
            final_report=report
        )
        
        assert "final_report" in state
        assert state["final_report"] == report
        assert len(state["final_report"]) > 0
    
    def test_state_with_auto_accepted_plan(self):
        """Test State with auto accepted plan flag."""
        state = State(
            messages=[],
            auto_accepted_plan=True
        )
        
        assert "auto_accepted_plan" in state
        assert state["auto_accepted_plan"] is True
        assert isinstance(state["auto_accepted_plan"], bool)
    
    def test_state_with_background_investigation_disabled(self):
        """Test State with background investigation disabled."""
        state = State(
            messages=[],
            enable_background_investigation=False
        )
        
        assert "enable_background_investigation" in state
        assert state["enable_background_investigation"] is False
        assert isinstance(state["enable_background_investigation"], bool)
    
    def test_state_with_background_investigation_results(self):
        """Test State with background investigation results."""
        results = "Background investigation completed successfully."
        state = State(
            messages=[],
            background_investigation_results=results
        )
        
        assert "background_investigation_results" in state
        assert state["background_investigation_results"] == results
    
    def test_state_with_all_custom_values(self):
        """Test State with all custom values."""
        test_messages = [{"role": "user", "content": "Test message"}]
        
        state = State(
            messages=test_messages,
            locale="fr-FR",
            observations=["Observation 1", "Observation 2"],
            plan_iterations=3,
            current_plan=self.test_plan,
            final_report="Comprehensive test report",
            auto_accepted_plan=True,
            enable_background_investigation=False,
            background_investigation_results="Investigation complete",
        )
        
        # Verify all values are set correctly
        assert state["messages"] == test_messages
        assert state["locale"] == "fr-FR"
        assert state["observations"] == ["Observation 1", "Observation 2"]
        assert state["plan_iterations"] == 3
        assert state["current_plan"].title == "Test Plan"
        assert state["final_report"] == "Comprehensive test report"
        assert state["auto_accepted_plan"] is True
        assert state["enable_background_investigation"] is False
        assert state["background_investigation_results"] == "Investigation complete"


class TestStateOperations:
    """Test cases for State operations."""
    
    def test_state_key_access(self):
        """Test accessing state keys."""
        state = State(
            messages=[],
            locale="en-US",
            plan_iterations=1
        )
        
        # Test key access
        assert "messages" in state
        assert "locale" in state
        assert "plan_iterations" in state
        assert "nonexistent_key" not in state
        
        # Test value access
        assert state["locale"] == "en-US"
        assert state["plan_iterations"] == 1
    
    def test_state_key_modification(self):
        """Test modifying state values."""
        state = State(
            messages=[],
            locale="en-US"
        )
        
        # Modify existing value
        state["locale"] = "es-ES"
        assert state["locale"] == "es-ES"
        
        # Add new value
        state["new_field"] = "new_value"
        assert "new_field" in state
        assert state["new_field"] == "new_value"
    
    def test_state_iteration(self):
        """Test iterating over state."""
        state = State(
            messages=[],
            locale="en-US",
            plan_iterations=2
        )
        
        # Test keys iteration
        keys = list(state.keys())
        assert "messages" in keys
        assert "locale" in keys
        assert "plan_iterations" in keys
        
        # Test items iteration
        items = dict(state.items())
        assert items["locale"] == "en-US"
        assert items["plan_iterations"] == 2
    
    def test_state_get_method(self):
        """Test state get method with defaults."""
        state = State(
            messages=[],
            locale="en-US"
        )
        
        # Test getting existing key
        assert state.get("locale") == "en-US"
        
        # Test getting non-existing key with default
        assert state.get("nonexistent", "default") == "default"
        
        # Test getting non-existing key without default
        assert state.get("nonexistent") is None
    
    def test_state_update(self):
        """Test updating state with dictionary."""
        state = State(messages=[])
        
        update_data = {
            "locale": "de-DE",
            "plan_iterations": 5,
            "final_report": "Updated report"
        }
        
        state.update(update_data)
        
        assert state["locale"] == "de-DE"
        assert state["plan_iterations"] == 5
        assert state["final_report"] == "Updated report"


class TestStateEdgeCases:
    """Test cases for State edge cases."""
    
    def test_state_with_none_values(self):
        """Test State with None values."""
        state = State(
            messages=[],
            current_plan=None,
            background_investigation_results=None
        )
        
        assert "current_plan" in state
        assert state["current_plan"] is None
        assert "background_investigation_results" in state
        assert state["background_investigation_results"] is None
    
    def test_state_with_empty_collections(self):
        """Test State with empty collections."""
        state = State(
            messages=[],
            observations=[],
            final_report=""
        )
        
        assert state["messages"] == []
        assert state["observations"] == []
        assert state["final_report"] == ""
        assert len(state["messages"]) == 0
        assert len(state["observations"]) == 0
        assert len(state["final_report"]) == 0
    
    def test_state_with_complex_plan(self):
        """Test State with complex plan structure."""
        # Create multiple steps
        step1 = Step(
            need_search=True,
            title="Research Step",
            description="Conduct research",
            step_type=StepType.RESEARCH
        )
        
        step2 = Step(
            need_search=False,
            title="Processing Step",
            description="Process data",
            step_type=StepType.PROCESSING
        )
        
        complex_plan = Plan(
            locale="zh-CN",
            has_enough_context=True,
            thought="Complex analysis required",
            title="Multi-step Analysis Plan",
            steps=[step1, step2]
        )
        
        state = State(
            messages=[],
            current_plan=complex_plan
        )
        
        plan = state["current_plan"]
        assert plan.locale == "zh-CN"
        assert plan.has_enough_context is True
        assert len(plan.steps) == 2
        assert plan.steps[0].step_type == StepType.RESEARCH
        assert plan.steps[1].step_type == StepType.PROCESSING
        assert plan.steps[0].need_search is True
        assert plan.steps[1].need_search is False
    
    def test_state_type_consistency(self):
        """Test that state maintains type consistency."""
        state = State(
            messages=[],
            plan_iterations=0,
            auto_accepted_plan=False,
            enable_background_investigation=True
        )
        
        # Verify types
        assert isinstance(state["plan_iterations"], int)
        assert isinstance(state["auto_accepted_plan"], bool)
        assert isinstance(state["enable_background_investigation"], bool)
        
        # Test type changes
        state["plan_iterations"] = 10
        assert isinstance(state["plan_iterations"], int)
        assert state["plan_iterations"] == 10
    
    def test_state_large_data(self):
        """Test State with large data structures."""
        # Create large observations list
        large_observations = [f"Observation {i}" for i in range(1000)]
        
        # Create large messages list
        large_messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(100)
        ]
        
        state = State(
            messages=large_messages,
            observations=large_observations,
            final_report="A" * 10000  # Large report
        )
        
        assert len(state["messages"]) == 100
        assert len(state["observations"]) == 1000
        assert len(state["final_report"]) == 10000
        
        # Verify data integrity
        assert state["observations"][0] == "Observation 0"
        assert state["observations"][999] == "Observation 999"
        assert state["messages"][0]["content"] == "Message 0"
        assert state["messages"][99]["content"] == "Message 99"


class TestStatePerformance:
    """Performance tests for State operations."""
    
    def test_state_creation_performance(self):
        """Test State creation performance."""
        import time
        
        start_time = time.time()
        
        for _ in range(1000):
            state = State(
                messages=[],
                locale="en-US",
                plan_iterations=1
            )
            assert state is not None
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        
        # State creation should be fast
        assert avg_time < 0.001  # Less than 1ms per creation
    
    def test_state_access_performance(self):
        """Test State access performance."""
        state = State(
            messages=[],
            locale="en-US",
            observations=["obs1", "obs2", "obs3"]
        )
        
        import time
        start_time = time.time()
        
        for _ in range(10000):
            _ = state["locale"]
            _ = state["observations"]
            _ = state.get("plan_iterations", 0)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10000
        
        # State access should be very fast
        assert avg_time < 0.0001  # Less than 0.1ms per access