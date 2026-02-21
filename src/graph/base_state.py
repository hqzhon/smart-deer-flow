# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Base state definitions for graph workflows.

This module provides common base classes and utilities for defining
workflow states across different graph modules.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, TypedDict
from dataclasses import dataclass
import types


@dataclass
class InputOutputState:
    """
    Base state with input and output fields.

    Common pattern used across podcast, ppt, prose, and other workflows.
    """

    input: str = ""
    output: Optional[Any] = None


class BaseWorkflowState(TypedDict, total=False):
    """
    Base state class for workflow graphs.

    Provides common workflow patterns:
    - Input/output fields
    - Error tracking
    - Status tracking
    - Messages (for LLM interactions)
    """

    messages: List[Any]
    input: str
    output: Optional[Any]
    error: Optional[str]
    error_history: List[Dict[str, Any]]
    status: str
    progress: float


class ProcessingState(TypedDict, total=False):
    """
    State for processing workflows.

    Common pattern for content processing workflows like prose, prompt_enhancer.
    """

    messages: List[Any]
    content: str
    option: str
    locale: str
    result: Optional[str]
    metadata: Dict[str, Any]


class GenerationState(TypedDict, total=False):
    """
    State for generation workflows.

    Common pattern for content generation workflows like podcast, ppt.
    """

    messages: List[Any]
    input: str
    generated_content: Optional[Any]
    output_path: Optional[str]
    assets: Dict[str, Any]
    current_step: str
    steps_completed: List[str]


T = TypeVar("T")


class StateMixin(Generic[T]):
    """
    Mixin for adding common state operations to TypedDict states.

    Note: This is a helper class for working with state dictionaries.
    """

    @staticmethod
    def get_data(state: Dict[str, Any]) -> Optional[T]:
        """Get the data field from state."""
        return state.get("data")

    @staticmethod
    def set_data(state: Dict[str, Any], data: T) -> None:
        """Set the data field in state."""
        state["data"] = data

    @staticmethod
    def has_data(state: Dict[str, Any]) -> bool:
        """Check if data exists in state."""
        return state.get("data") is not None


def add_error_to_state(
    state: Dict[str, Any], error: str, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add an error to a state's error history.

    Args:
        state: The state dictionary.
        error: Error message.
        context: Optional error context.
    """
    state["error"] = error
    if "error_history" not in state:
        state["error_history"] = []
    state["error_history"].append(
        {
            "error": error,
            "context": context or {},
        }
    )


def set_state_status(
    state: Dict[str, Any], status: str, progress: Optional[float] = None
) -> None:
    """
    Set the workflow status in a state.

    Args:
        state: The state dictionary.
        status: Status string.
        progress: Optional progress value.
    """
    state["status"] = status
    if progress is not None:
        state["progress"] = progress


def create_simple_state_class(
    name: str,
    fields: Dict[str, Any],
) -> type:
    """
    Factory function to create a simple TypedDict state class.

    Args:
        name: Name of the state class.
        fields: Dictionary of field names and default values.

    Returns:
        A new TypedDict state class.

    Usage:
        MyState = create_simple_state_class(
            "MyState",
            {"input": "", "output": None, "count": 0}
        )
        state: MyState = {"input": "test", "output": None, "count": 0}
    """
    annotations = {}
    for field_name, default_value in fields.items():
        field_type = type(default_value)
        if default_value is None:
            field_type = Optional[Any]
        elif isinstance(default_value, list):
            field_type = List[Any]
        elif isinstance(default_value, dict):
            field_type = Dict[str, Any]
        annotations[field_name] = field_type

    def exec_body(ns: dict) -> None:
        ns["__annotations__"] = annotations
        ns["__total__"] = False

    return types.new_class(name, (TypedDict,), {}, exec_body)
