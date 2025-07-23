# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from typing import Dict, Any
from src.prose.graph.state import ProseState


# Operation definitions
OPERATIONS = {
    "continue": "continues existing text based on context",
    "improve": "improves existing text quality",
    "fix": "fixes grammar and spelling errors",
    "longer": "lengthens existing text",
    "shorter": "shortens existing text",
    "zap": "generates text based on a command",
}


def _process_prose_operation(state: ProseState) -> Dict[str, Any]:
    """Prose operation processor using template files."""
    option = state.get("option", "")
    content = state.get("content", "")
    command = state.get("command", "")
    locale = state.get("locale", "en-US")

    # Validate operation
    if option not in OPERATIONS:
        return {"prose_content": f"Unknown operation: {option}"}

    try:
        # Lazy imports to avoid circular dependencies
        from src.config.config_loader import get_settings
        from src.llms.llm import get_llm_by_type
        from src.llms.error_handler import safe_llm_call
        from src.utils.template import apply_prompt_template

        # Get configuration and LLM
        settings = get_settings()
        llm = get_llm_by_type(settings.llm.type)

        # Build prompt using template
        template_context = {
            "operation": option,
            "operation_description": OPERATIONS[option],
            "content": content,
            "command": command,
            "locale": locale,
        }

        prompt = apply_prompt_template("prose/prose_operations.md", **template_context)

        # Call LLM
        response = safe_llm_call(
            llm=llm,
            prompt=prompt,
            operation_name=f"prose_{option}",
            context=f"Prose {option} operation",
        )

        # Return appropriate output
        output_key = "output" if option == "zap" else "prose_content"
        return {output_key: response}

    except Exception as e:
        logging.error(f"Error in prose {option} operation: {e}")
        return {"prose_content": f"Error processing {option}: {str(e)}"}


# Backward compatibility - these functions are no longer needed
# but kept for any external references
def prose_continue_node(state: ProseState) -> Dict[str, Any]:
    """Deprecated: Use _process_prose_operation directly."""
    return _process_prose_operation(state)


def prose_improve_node(state: ProseState) -> Dict[str, Any]:
    """Deprecated: Use _process_prose_operation directly."""
    return _process_prose_operation(state)


def prose_fix_node(state: ProseState) -> Dict[str, Any]:
    """Deprecated: Use _process_prose_operation directly."""
    return _process_prose_operation(state)


def prose_longer_node(state: ProseState) -> Dict[str, Any]:
    """Deprecated: Use _process_prose_operation directly."""
    return _process_prose_operation(state)


def prose_shorter_node(state: ProseState) -> Dict[str, Any]:
    """Deprecated: Use _process_prose_operation directly."""
    return _process_prose_operation(state)


def prose_zap_node(state: ProseState) -> Dict[str, Any]:
    """Deprecated: Use _process_prose_operation directly."""
    return _process_prose_operation(state)
