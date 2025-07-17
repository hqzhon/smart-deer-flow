# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain_core.messages import HumanMessage
from src.config.config_loader import get_settings
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call
from src.prose.graph.state import ProseState
from src.utils.template import get_prompt_template

logger = logging.getLogger(__name__)


def prose_longer_node(state: ProseState):
    logger.info("Generating prose longer content...")
    settings = get_settings()
    llm_type = getattr(settings.agent_llm_map, "prose_writer", "basic")
    model = get_llm_by_type(llm_type)
    prompt_template = get_prompt_template("prose_longer")
    messages = [HumanMessage(content=prompt_template.format(content=state["content"]))]

    response = safe_llm_call(
        model.invoke,
        messages,
        operation_name="Prose Longer",
        context="Making prose longer",
    )
    response_content = (
        response.content if hasattr(response, "content") else str(response)
    )
    logger.info(f"prose_longer_node response: {response_content}")
    return {"prose_content": response_content}
