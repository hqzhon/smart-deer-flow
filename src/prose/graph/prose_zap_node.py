# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain_core.messages import HumanMessage

from src.config.config_loader import get_settings
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call
from src.utils.template import apply_prompt_template
from src.prose.graph.state import ProseState

logger = logging.getLogger(__name__)


def prose_zap_node(state: ProseState):
    logger.info("Generating prose zap content...")
    settings = get_settings()
    llm_type = getattr(settings.agent_llm_map, "prose_writer", "basic")
    model = get_llm_by_type(llm_type)
    # Create a temporary state-like object with locale, content and command for template rendering
    template_vars = {
        "locale": state.get("locale", "en-US"),
        "content": state["content"],
        "command": state["command"]
    }
    prompt_content = apply_prompt_template("prose/prose_zap", state, template_vars)
    messages = [HumanMessage(content=prompt_content)]
    
    prose_content = safe_llm_call(
        model.invoke,
        messages,
        operation_name="Prose Zap",
        context="Processing text with custom command",
    )
    logger.info(f"prose_content: {prose_content}")
    return {"output": prose_content.content}
