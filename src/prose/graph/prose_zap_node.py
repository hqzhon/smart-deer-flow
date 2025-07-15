# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain.schema import HumanMessage, SystemMessage

from src.config.config_loader import get_settings
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call
from src.utils.template import get_prompt_template
from src.prose.graph.state import ProseState

logger = logging.getLogger(__name__)


def prose_zap_node(state: ProseState):
    logger.info("Generating prose zap content...")
    settings = get_settings()
    llm_type = settings.agent_llm_map.get("prose_writer", "gpt-4o-mini")
    model = get_llm_by_type(llm_type)
    prose_content = safe_llm_call(
        model.invoke,
        [
            SystemMessage(content=get_prompt_template("prose/prose_zap")),
            HumanMessage(
                content=f"For this text: {state['content']}.\nYou have to respect the command: {state['command']}"
            ),
        ],
        operation_name="Prose Zap",
        context="Processing text with custom command",
    )
    logger.info(f"prose_content: {prose_content}")
    return {"output": prose_content.content}
