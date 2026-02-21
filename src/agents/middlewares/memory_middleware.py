"""Middleware for memory mechanism."""

from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from src.agents.memory.queue import get_memory_queue
from src.config.memory_config import get_memory_config
from src.utils.common.middleware_utils import filter_messages_for_memory


class MemoryMiddleware(AgentMiddleware[AgentState]):
    """Middleware that queues conversation for memory update after agent execution.

    This middleware:
    1. After each agent execution, queues the conversation for memory update
    2. Only includes user inputs and final assistant responses (ignores tool calls)
    3. The queue uses debouncing to batch multiple updates together
    4. Memory is updated asynchronously via LLM summarization
    """

    state_schema = AgentState

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Queue conversation for memory update after agent completes.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            None (no state changes needed from this middleware).
        """
        config = get_memory_config()
        if not config.enabled:
            return None

        thread_id = runtime.context.get("thread_id")
        if not thread_id:
            print("MemoryMiddleware: No thread_id in context, skipping memory update")
            return None

        messages = state.get("messages", [])
        if not messages:
            print("MemoryMiddleware: No messages in state, skipping memory update")
            return None

        filtered_messages = filter_messages_for_memory(messages)

        user_messages = [
            m for m in filtered_messages if getattr(m, "type", None) == "human"
        ]
        assistant_messages = [
            m for m in filtered_messages if getattr(m, "type", None) == "ai"
        ]

        if not user_messages or not assistant_messages:
            return None

        queue = get_memory_queue()
        queue.add(thread_id=thread_id, messages=filtered_messages)

        return None
