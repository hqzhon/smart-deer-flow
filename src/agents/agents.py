# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.config.config_loader import get_settings


# Create agents using configured LLM types
def create_agent(agent_name: str, agent_type: str, tools: list, prompt_template: str):
    """Factory function to create agents with consistent configuration."""
    # Use delayed import to avoid circular import
    from src.llms.llm import get_llm_by_type

    settings = get_settings()
    agent_llm_config = settings.agent_llm_map

    # Get LLM type for the agent from configuration
    llm_type = getattr(agent_llm_config, agent_type, "basic")

    return create_react_agent(
        name=agent_name,
        model=get_llm_by_type(llm_type),
        tools=tools,
        prompt=lambda state: apply_prompt_template(prompt_template, state),
    )
