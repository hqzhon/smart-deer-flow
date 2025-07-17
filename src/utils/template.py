# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import dataclasses
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState
# Remove TYPE_CHECKING import for Configuration as it's no longer needed

# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), '..', 'prompts')),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        prompt_name: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        template = env.get_template(f"{prompt_name}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def apply_prompt_template(
    prompt_name: str, state: AgentState, configurable = None
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.
    CRITICAL: This function now includes context evaluation to prevent token overflow.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    from src.utils.context.context_evaluator import (
        get_global_context_evaluator,
    )
    from src.config import get_settings
    import logging

    logger = logging.getLogger(__name__)

    # CRITICAL: Apply context evaluation before template rendering
    try:
        config = get_settings()
        evaluator = get_global_context_evaluator(config)

        # Get messages from state, with aggressive limitation
        messages = state.get("messages", [])

        # AGGRESSIVE: Limit messages to prevent token explosion in template rendering
        if len(messages) > 10:  # Only keep last 10 messages
            messages = messages[-10:]
            logger.info(
                "Limited messages to last 10 for template rendering to prevent token overflow"
            )
        elif len(messages) > 5:  # Only keep last 5 messages if more than 5
            messages = messages[-5:]
            logger.info("Limited messages to last 5 for template rendering")

        # Apply context evaluation to messages
        model_name = getattr(config, "model", "deepseek-chat")

        # Convert messages to proper format if needed
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

        # Evaluate context and optimize if needed
        metrics = evaluator.evaluate_context_before_llm_call(
            formatted_messages, model_name, "template_rendering"
        )

        if metrics.compression_needed:
            optimized_messages, optimization_info = (
                evaluator.optimize_context_for_llm_call(
                    formatted_messages,
                    model_name,
                    metrics.recommended_strategy,
                    "template_rendering",
                )
            )
            logger.info(f"Context optimization applied: {optimization_info}")
            evaluated_messages = optimized_messages
        else:
            evaluated_messages = formatted_messages

        # Convert back to dict format for consistency
        evaluated_messages = [
            {
                "role": (
                    "system"
                    if isinstance(msg, SystemMessage)
                    else "assistant" if isinstance(msg, AIMessage) else "user"
                ),
                "content": msg.content,
            }
            for msg in evaluated_messages
        ]

        logger.info(
            f"Context evaluation applied in template rendering: {len(messages)} -> {len(evaluated_messages)} messages"
        )

    except Exception as e:
        logger.warning(
            f"Context evaluation failed in template rendering, using original messages: {e}"
        )
        evaluated_messages = state.get("messages", [])

    # Convert state to dict for template rendering, but exclude messages to avoid duplication
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **{
            k: v for k, v in state.items() if k != "messages"
        },  # CRITICAL: Exclude messages from state_vars
    }

    # Add configurable variables
    if configurable:
        # Handle different types of configurable objects
        if hasattr(configurable, 'model_dump'):  # Pydantic model
            state_vars.update(configurable.model_dump())
        elif hasattr(configurable, '__dict__'):  # Regular class instance
            state_vars.update(configurable.__dict__)
        elif dataclasses.is_dataclass(configurable):  # Dataclass instance
            state_vars.update(dataclasses.asdict(configurable))

    try:
        template = env.get_template(f"{prompt_name}.md")
        system_prompt = template.render(**state_vars)
        return [{"role": "system", "content": system_prompt}] + evaluated_messages
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
