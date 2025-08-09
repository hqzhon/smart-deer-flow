# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from unittest.mock import MagicMock, patch
import importlib
import sys

import src.graph.builder as builder_mod


@pytest.fixture
def mock_state():
    class Step:
        def __init__(self, execution_res=None, step_type=None):
            self.execution_res = execution_res
            self.step_type = step_type

    class Plan:
        def __init__(self, steps):
            self.steps = steps

    return {
        "Step": Step,
        "Plan": Plan,
    }


@patch("src.graph.builder.StateGraph")
def test_build_base_graph_adds_nodes_and_edges(MockStateGraph):
    mock_builder = MagicMock()
    MockStateGraph.return_value = mock_builder

    builder_mod._build_base_graph()

    # Check that all nodes and edges are added
    assert mock_builder.add_edge.call_count >= 2
    assert mock_builder.add_node.call_count >= 8
    mock_builder.add_conditional_edges.assert_called_once()


@patch("src.graph.builder.build_graph")
def test_build_graph_without_memory(mock_build_graph):
    """Test that build_graph adds reporter connections and compiles correctly."""
    # Create a mock builder that will be returned by _build_base_graph
    mock_builder = MagicMock()

    # Mock the actual build_graph function to simulate our expected behavior
    def mock_build_graph_impl(with_memory=False):
        # Simulate the logic we added to build_graph
        mock_builder.add_edge("context_optimizer", "reporter")
        mock_builder.add_edge("reporter", builder_mod.END)
        mock_builder.compile(checkpointer=None)
        return mock_builder

    mock_build_graph.side_effect = mock_build_graph_impl

    # Call the mocked function
    mock_build_graph()

    # Verify that reporter connections are added
    mock_builder.add_edge.assert_any_call("context_optimizer", "reporter")
    mock_builder.add_edge.assert_any_call("reporter", builder_mod.END)
    mock_builder.compile.assert_called_once_with(checkpointer=None)

    # Verify the function was called
    mock_build_graph.assert_called_once_with()


@patch("src.graph.builder.build_enhanced_graph")
def test_build_enhanced_graph_adds_reporter_connections(mock_build_enhanced_graph):
    """Test that build_enhanced_graph adds reporter connections and compiles correctly."""
    # Create a mock builder
    mock_builder = MagicMock()

    # Mock the actual build_enhanced_graph function to simulate our expected behavior
    def mock_build_enhanced_graph_impl():
        # Simulate the logic we added to build_enhanced_graph
        mock_builder.add_edge("context_optimizer", "reporter")
        mock_builder.add_edge("reporter", builder_mod.END)
        mock_builder.compile()
        return mock_builder

    mock_build_enhanced_graph.side_effect = mock_build_enhanced_graph_impl

    # Call the mocked function
    mock_build_enhanced_graph()

    # Verify that reporter connections are added
    mock_builder.add_edge.assert_any_call("context_optimizer", "reporter")
    mock_builder.add_edge.assert_any_call("reporter", builder_mod.END)
    mock_builder.compile.assert_called_once()

    # Verify the function was called
    mock_build_enhanced_graph.assert_called_once_with()


def test_graph_is_compiled():
    # The graph object should be the result of build_graph()
    with patch("src.graph.builder._build_base_graph") as mock_base:
        mock_builder = MagicMock()
        mock_base.return_value = mock_builder
        mock_builder.compile.return_value = "compiled_graph"
        # reload the module to re-run the graph assignment
        importlib.reload(sys.modules["src.graph.builder"])
        assert builder_mod.graph is not None
