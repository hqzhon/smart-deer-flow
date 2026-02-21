# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langgraph.graph import END, START, StateGraph

NodeDefinition = Union[
    Tuple[str, Callable],
    Dict[str, Any],
]


def build_simple_graph(
    state_class: Any,
    nodes: List[NodeDefinition],
    edges: Optional[List[Tuple[str, str]]] = None,
    entry_point: Optional[str] = None,
    finish_point: Optional[str] = None,
    conditional_edges: Optional[List[Tuple[str, Callable, Dict[str, str]]]] = None,
) -> Any:
    """
    Build a simple state graph from node and edge definitions.

    Args:
        state_class: The state class for the graph
        nodes: List of node definitions, each can be:
               - Tuple of (node_name, node_function)
               - Dictionary with "name" and "func" keys
        edges: List of (from_node, to_node) edge tuples
        entry_point: Optional entry point node name
        finish_point: Optional finish point node name
        conditional_edges: List of conditional edges:
                            (from_node, condition_func, {condition_result: target_node})

    Returns:
        Compiled StateGraph
    """
    builder = StateGraph(state_class)

    # Add nodes
    for node_def in nodes:
        if isinstance(node_def, tuple) and len(node_def) == 2:
            node_name, node_func = node_def
        elif isinstance(node_def, dict):
            node_name = node_def["name"]
            node_func = node_def["func"]
        else:
            raise ValueError(f"Invalid node definition: {node_def}")
        builder.add_node(node_name, node_func)

    # Add edges
    if edges:
        for from_node, to_node in edges:
            builder.add_edge(from_node, to_node)

    # Add conditional edges
    if conditional_edges:
        for from_node, condition_func, condition_map in conditional_edges:
            if from_node is None or from_node == START:
                builder.add_conditional_edges(START, condition_func, condition_map)
            else:
                builder.add_conditional_edges(from_node, condition_func, condition_map)

    # Set entry and finish points if specified
    if entry_point:
        if hasattr(builder, "set_entry_point"):
            builder.set_entry_point(entry_point)
        else:
            builder.add_edge(START, entry_point)

    if finish_point:
        if hasattr(builder, "set_finish_point"):
            builder.set_finish_point(finish_point)
        else:
            builder.add_edge(finish_point, END)

    # Default to adding START -> first node and last node -> END
    # if no entry/finish points specified and no edges from START/to END
    if not entry_point and edges:
        # Find nodes with no incoming edges (potential starting points)
        target_nodes = {edge[1] for edge in edges}
        source_nodes = {edge[0] for edge in edges}
        starting_nodes = source_nodes - target_nodes
        if starting_nodes and START not in source_nodes:
            for node in starting_nodes:
                builder.add_edge(START, node)

    if not finish_point and edges:
        # Find nodes with no outgoing edges (potential finish points)
        source_nodes = {edge[0] for edge in edges}
        target_nodes = {edge[1] for edge in edges}
        finishing_nodes = target_nodes - source_nodes
        if finishing_nodes and END not in target_nodes:
            for node in finishing_nodes:
                builder.add_edge(node, END)

    return builder.compile()
