# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

# Import server module for test compatibility
import importlib.util

if importlib.util.find_spec("src.server") is not None:
    from . import server  # noqa: F401
