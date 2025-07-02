# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from src.rag.retriever import Resource
from src.config.report_style import ReportStyle
from src.utils.content_processor import ModelTokenLimits


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields."""

    resources: list[Resource] = field(
        default_factory=list
    )  # Resources to be used for the research
    max_plan_iterations: int = 1  # Maximum number of plan iterations
    max_step_num: int = 3  # Maximum number of steps in a plan
    max_search_results: int = 3  # Maximum number of search results
    mcp_settings: dict = None  # MCP settings, including dynamic loaded tools
    report_style: str = ReportStyle.ACADEMIC.value  # Report style
    enable_deep_thinking: bool = False  # Whether to enable deep thinking
    enable_parallel_execution: bool = True  # Whether to enable parallel step execution
    max_parallel_tasks: int = 3  # Maximum number of parallel tasks
    max_context_steps_parallel: int = 5  # Maximum number of context steps in parallel execution
    
    enable_smart_chunking: bool = True  # Whether to enable smart content chunking
    enable_content_summarization: bool = True  # Whether to enable content summarization
    chunk_strategy: str = "auto"  # Chunking strategy: "auto", "sentences", "paragraphs"
    summary_type: str = "comprehensive"  # Summary type: "comprehensive", "key_points", "abstract"
    model_token_limits: dict[str, ModelTokenLimits] = field(default_factory=dict)  # Model token limits
    basic_model: Optional[Any] = None  # Basic model instance
    reasoning_model: Optional[Any] = None  # Reasoning model instance

    _current_instance: Optional["Configuration"] = None
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        instance = cls(**{k: v for k, v in values.items() if v})
        cls._current_instance = instance  # Store current instance
        return instance
    
    @classmethod
    def get_current(cls) -> Optional["Configuration"]:
        """Get the current Configuration instance."""
        return cls._current_instance
