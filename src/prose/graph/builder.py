# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import logging
from langgraph.graph import END, START, StateGraph

from src.prose.graph.prose_processor import _process_prose_operation, OPERATIONS
from src.prose.graph.state import ProseState


def route_by_option(state: ProseState) -> str:
    """Route to the appropriate prose operation based on the option."""
    option = state.get("option")
    return "prose_processor" if option in OPERATIONS else END


def build_graph():
    """Build and return the simplified prose workflow graph."""
    graph = StateGraph(ProseState)

    # Single processor node for all operations
    graph.add_node("prose_processor", _process_prose_operation)

    # Add conditional routing from START
    graph.add_conditional_edges(
        START,
        route_by_option,
        {
            "prose_processor": "prose_processor",
            END: END,
        },
    )

    # Add edge from processor to END
    graph.add_edge("prose_processor", END)

    return graph.compile()


async def _test_workflow():
    workflow = build_graph()
    events = workflow.astream(
        {
            "content": "The weather in Beijing is sunny",
            "option": "continue",
        },
        stream_mode="messages",
        subgraphs=True,
    )
    async for node, event in events:
        e = event[0]
        print({"id": e.id, "object": "chat.completion.chunk", "content": e.content})


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_test_workflow())
