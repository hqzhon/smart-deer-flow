# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.prompts.prompt_manager import get_prompt_manager
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


def create_agent_with_managed_prompt(
    agent_name: str, agent_type: str, tools: list, settings=None
):
    """Factory function to create agents using PromptManager for prompt retrieval.

    This function reads the prompt_name from configuration and uses PromptManager
    to load the actual prompt content, enabling separation of prompts from code.

    Args:
        agent_name: Name of the agent
        agent_type: Type of the agent (used for LLM selection)
        tools: List of tools for the agent
        settings: Optional settings object (will use get_settings() if not provided)

    Returns:
        Configured agent with managed prompt
    """
    # Use delayed import to avoid circular import
    from src.llms.llm import get_llm_by_type

    if settings is None:
        settings = get_settings()

    agent_llm_config = settings.agent_llm_map

    # Get LLM type for the agent from configuration
    llm_type = getattr(agent_llm_config, agent_type, "basic")

    # Get prompt name from configuration (default to agent_name if not configured)
    prompt_name = getattr(settings.agents, f"{agent_name}_prompt", agent_name)

    # Get prompt manager and load prompt content
    prompt_manager = get_prompt_manager()

    def get_prompt_content(state):
        """Get prompt content using PromptManager and apply template."""
        try:
            # Get prompt content from PromptManager
            prompt_content = prompt_manager.get_prompt(prompt_name)
            # Apply template using existing template system
            return apply_prompt_template_content(prompt_content, state)
        except KeyError:
            # Fallback to original behavior if prompt not found
            return apply_prompt_template(prompt_name, state)

    return create_react_agent(
        name=agent_name,
        model=get_llm_by_type(llm_type),
        tools=tools,
        prompt=get_prompt_content,
    )


def apply_prompt_template_content(prompt_content: str, state) -> list:
    """Apply template variables to prompt content and return formatted messages.

    This is a simplified version of apply_prompt_template that works with
    prompt content directly instead of loading from file.

    Args:
        prompt_content: The prompt content string
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    from src.utils.context.context_evaluator import get_global_context_evaluator
    from datetime import datetime
    import logging

    logger = logging.getLogger(__name__)

    # Apply context evaluation (simplified version)
    try:
        config = get_settings()
        get_global_context_evaluator(config)

        messages = state.get("messages", [])

        # Limit messages to prevent token overflow
        if len(messages) > 10:
            messages = messages[-10:]
            logger.info("Limited messages to last 10 for template rendering")
        elif len(messages) > 5:
            messages = messages[-5:]
            logger.info("Limited messages to last 5 for template rendering")

        # Convert messages to proper format
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "human")
                content = msg.get("content", "")
                if role == "system":
                    formatted_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    formatted_messages.append(AIMessage(content=content))
                else:
                    formatted_messages.append(HumanMessage(content=content))
            else:
                formatted_messages.append(msg)

        # Convert back to dict format
        evaluated_messages = [
            {
                "role": (
                    "system"
                    if isinstance(msg, SystemMessage)
                    else "assistant" if isinstance(msg, AIMessage) else "user"
                ),
                "content": msg.content,
            }
            for msg in formatted_messages
        ]

    except Exception as e:
        logger.warning(f"Context evaluation failed, using original messages: {e}")
        evaluated_messages = state.get("messages", [])

    # Prepare state variables for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **{k: v for k, v in state.items() if k != "messages"},
    }

    try:
        # Use Jinja2 for template rendering
        from jinja2 import Environment

        env = Environment(
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.from_string(prompt_content)
        system_prompt = template.render(**state_vars)
        return [{"role": "system", "content": system_prompt}] + evaluated_messages
    except Exception as e:
        logger.warning(f"Template formatting failed: {e}, using prompt as-is")
        return [{"role": "system", "content": prompt_content}] + evaluated_messages
