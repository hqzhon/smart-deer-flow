# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Optional

from src.graph.types import MessagesState


class ProseState(MessagesState):
    """Minimal state for prose generation workflow."""

    # Input fields
    content: str = ""  # Text content to process
    option: str = ""  # Operation: continue, improve, fix, longer, shorter, zap
    command: Optional[str] = None  # Command for zap operation
    locale: str = "en-US"  # Language locale

    # Output fields
    prose_content: Optional[str] = None  # Main output
    output: Optional[str] = None  # Alternative output for zap
