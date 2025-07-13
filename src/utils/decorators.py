# -*- coding: utf-8 -*-
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Decorators module - Re-exports decorators from common module for backward compatibility.

This module provides a convenient import path for decorators that are actually
implemented in src.utils.common.decorators.
"""

# Import all decorators from the common module
from src.utils.common.decorators import (
    safe_background_task,
    retry_on_failure,
    log_execution_time,
)

# Re-export for backward compatibility
__all__ = [
    "safe_background_task",
    "retry_on_failure", 
    "log_execution_time",
]