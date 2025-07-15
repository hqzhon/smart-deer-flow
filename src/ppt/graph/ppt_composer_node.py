# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
import uuid

from langchain.schema import HumanMessage, SystemMessage

from src.config.config_loader import get_settings
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call
from src.utils.template import get_prompt_template

from .state import PPTState

logger = logging.getLogger(__name__)


def ppt_composer_node(state: PPTState):
    logger.info("Generating ppt content...")
    settings = get_settings()
    llm_type = settings.agent_llm_map.get("ppt_composer", "gpt-4o-mini")
    model = get_llm_by_type(llm_type)
    ppt_content = safe_llm_call(
        model.invoke,
        [
            SystemMessage(content=get_prompt_template("ppt/ppt_composer")),
            HumanMessage(content=state["input"]),
        ],
        operation_name="PPT Composer",
        context="Generating PPT content",
    )
    logger.info(f"ppt_content: {ppt_content}")
    # save the ppt content in a temp file
    temp_ppt_file_path = os.path.join(os.getcwd(), f"ppt_content_{uuid.uuid4()}.md")
    with open(temp_ppt_file_path, "w") as f:
        f.write(ppt_content.content)
    return {"ppt_content": ppt_content, "ppt_file_path": temp_ppt_file_path}
