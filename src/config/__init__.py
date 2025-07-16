# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Configuration package with unified configuration system.
Provides type-safe configuration management with Pydantic models.
"""

# Unified configuration system
from .models import (
    AppSettings,
    SearchEngine,
    RAGProvider,
    LLMType,
    ReportStyle,
    SummaryType,
    IsolationLevel
)
from .config_loader import get_settings, get_config_loader

# Backward compatibility
from src.constants.questions import BUILT_IN_QUESTIONS, BUILT_IN_QUESTIONS_ZH_CN

__all__ = [
    "AppSettings",
    "get_settings",
    "get_config_loader",
    "SearchEngine",
    "RAGProvider",
    "LLMType",
    "ReportStyle",
    "SummaryType",
    "IsolationLevel",
    "BUILT_IN_QUESTIONS",
    "BUILT_IN_QUESTIONS_ZH_CN"
]
