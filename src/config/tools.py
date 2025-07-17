# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Tool configuration constants and utilities.
Provides backward compatibility for tool-related constants.
"""

import os
from typing import Optional

from .models import RAGProvider, SearchEngine
from .config_loader import get_settings


def _get_selected_rag_provider() -> Optional[str]:
    """Get the selected RAG provider from configuration.

    Returns:
        The selected RAG provider value or None if not configured.
    """
    try:
        settings = get_settings()
        rag_provider = settings.tools.rag_provider
        return rag_provider.value if rag_provider else None
    except Exception:
        # Fallback to environment variable
        return os.getenv("SELECTED_RAG_PROVIDER")


def _get_selected_search_engine() -> str:
    """Get the selected search engine from configuration.

    Returns:
        The selected search engine value.
    """
    try:
        settings = get_settings()
        return settings.tools.search_engine.value
    except Exception:
        # Fallback to environment variable or default
        return os.getenv("SELECTED_SEARCH_ENGINE", SearchEngine.TAVILY.value)


# Backward compatibility constants
SELECTED_RAG_PROVIDER = _get_selected_rag_provider()
SELECTED_SEARCH_ENGINE = _get_selected_search_engine()

# Export the enum for convenience
__all__ = [
    "RAGProvider",
    "SearchEngine",
    "SELECTED_RAG_PROVIDER",
    "SELECTED_SEARCH_ENGINE",
]
