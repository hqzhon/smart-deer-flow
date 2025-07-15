# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Configuration package with new optimized system.
Provides unified configuration management with Pydantic models.
"""

# New configuration system
from .models import AppSettings
from .config_loader import get_settings

# Backward compatibility
from .report_style import ReportStyle
from .questions import BUILT_IN_QUESTIONS, BUILT_IN_QUESTIONS_ZH_CN
from .tools import SearchEngine, SELECTED_SEARCH_ENGINE

__all__ = [
    "AppSettings",
    "get_settings",
    "ReportStyle",
    "BUILT_IN_QUESTIONS",
    "BUILT_IN_QUESTIONS_ZH_CN",
    "SearchEngine",
    "SELECTED_SEARCH_ENGINE"
]