# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call
from src.prompts.template import get_prompt_template
from src.prose.graph.state import ProseState

logger = logging.getLogger(__name__)


def prose_fix_node(state: ProseState):
    logger.info("Generating prose fix content...")
    model = get_llm_by_type(AGENT_LLM_MAP["prose_writer"])
    messages = [
        SystemMessage(content=get_prompt_template("prose/prose_fix")),
        HumanMessage(content=f"The existing text is: {state['content']}"),
    ]
    
    prose_content = safe_llm_call(
        model.invoke,
        messages,
        operation_name="Prose Fixer",
        context="Fix prose content",
        enable_context_evaluation=True
    )
    response_content = prose_content.content if hasattr(prose_content, 'content') else str(prose_content)
    logger.info(f"prose_content: {response_content}")
    return {"prose_content": response_content}
