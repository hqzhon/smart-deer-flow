# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any, List, Optional, Set

from langchain_core.messages import AIMessage, ToolMessage


def get_last_assistant_message(messages: List[Any]) -> Optional[AIMessage]:
    """Get the last assistant message from the message list.

    Args:
        messages: List of messages

    Returns:
        Last AIMessage or None if not found
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def has_specific_tool_call(message: AIMessage, tool_name: str) -> bool:
    """Check if the assistant message contains a specific tool call.

    Args:
        message: Assistant message to check
        tool_name: Name of the tool to check for

    Returns:
        True if message contains the specified tool call
    """
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return False

    return any(tool_call.get("name") == tool_name for tool_call in message.tool_calls)


def are_all_tools_completed(messages: List[Any], assistant_msg: AIMessage) -> bool:
    """Check if all tool calls in the assistant message have been completed.

    Args:
        messages: List of all messages
        assistant_msg: The assistant message containing tool calls

    Returns:
        True if all tool calls have corresponding ToolMessages
    """
    if not hasattr(assistant_msg, "tool_calls") or not assistant_msg.tool_calls:
        return False

    tool_call_ids = {
        tool_call.get("id")
        for tool_call in assistant_msg.tool_calls
        if tool_call.get("id")
    }

    try:
        assistant_idx = messages.index(assistant_msg)
    except ValueError:
        return False

    completed_tool_ids = set()
    for msg in messages[assistant_idx + 1 :]:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            completed_tool_ids.add(msg.tool_call_id)

    return tool_call_ids.issubset(completed_tool_ids)


def get_completed_tool_ids(messages: List[Any], assistant_idx: int) -> Set[str]:
    """Get all completed tool call IDs after a specific assistant message index.

    Args:
        messages: List of all messages
        assistant_idx: Index of the assistant message in the list

    Returns:
        Set of completed tool call IDs
    """
    completed_tool_ids = set()
    for msg in messages[assistant_idx + 1 :]:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            completed_tool_ids.add(msg.tool_call_id)
    return completed_tool_ids


def filter_messages_for_memory(messages: List[Any]) -> List[Any]:
    """Filter messages to keep only user inputs and final assistant responses.

    This filters out:
    - Tool messages (intermediate tool call results)
    - AI messages with tool_calls (intermediate steps, not final responses)

    Only keeps:
    - Human messages (user input)
    - AI messages without tool_calls (final assistant responses)

    Args:
        messages: List of all conversation messages.

    Returns:
        Filtered list containing only user inputs and final assistant responses.
    """
    filtered = []
    for msg in messages:
        msg_type = getattr(msg, "type", None)

        if msg_type == "human":
            filtered.append(msg)
        elif msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                filtered.append(msg)

    return filtered


def is_chinese_text(text: str) -> bool:
    """Check if text contains Chinese characters.

    Args:
        text: Text to check

    Returns:
        True if text contains Chinese characters
    """
    return any("\u4e00" <= char <= "\u9fff" for char in text)
