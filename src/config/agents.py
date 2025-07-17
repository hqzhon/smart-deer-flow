# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Agent configuration constants and utilities.
Provides backward compatibility for agent-related constants.
"""

from .models import LLMType, AgentLLMSettings
from .config_loader import get_settings


def get_agent_llm_map() -> AgentLLMSettings:
    """Get the agent-LLM mapping configuration.
    
    Returns:
        The agent-LLM mapping settings.
    """
    try:
        settings = get_settings()
        return settings.agent_llm_map
    except Exception:
        # Return default mapping if configuration fails
        return AgentLLMSettings()


# Backward compatibility exports
__all__ = [
    'LLMType',
    'AgentLLMSettings',
    'get_agent_llm_map'
]