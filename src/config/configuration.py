# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from src.rag.retriever import Resource
from src.config.report_style import ReportStyle
from src.utils.tokens.content_processor import ModelTokenLimits


@dataclass
class AdvancedContextConfig:
    """Configuration for advanced context management."""

    max_context_ratio: float = 0.6
    sliding_window_size: int = 5
    overlap_ratio: float = 0.2
    compression_threshold: float = 0.8
    default_strategy: str = "adaptive"
    priority_weights: dict = field(
        default_factory=lambda: {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }
    )
    enable_caching: bool = True
    enable_analytics: bool = True
    debug_mode: bool = False


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
    max_context_steps_parallel: int = (
        1  # Maximum number of context steps in parallel execution (reduced for token optimization)
    )
    disable_context_parallel: bool = (
        False  # Whether to completely disable context sharing in parallel execution
    )

    # enable_smart_chunking removed - smart chunking is now always enabled
    enable_content_summarization: bool = True  # Whether to enable content summarization
    enable_smart_filtering: bool = (
        True  # Whether to enable LLM-based search result filtering
    )
    # chunk_strategy removed - smart chunking now uses 'auto' strategy by default
    summary_type: str = (
        "comprehensive"  # Summary type: "comprehensive", "key_points", "abstract"
    )
    model_token_limits: dict[str, ModelTokenLimits] = field(
        default_factory=dict
    )  # Model token limits
    basic_model: Optional[Any] = None  # Basic model instance
    reasoning_model: Optional[Any] = None  # Reasoning model instance

    # Advanced context management configuration
    advanced_context_config: AdvancedContextConfig = field(
        default_factory=AdvancedContextConfig
    )  # Advanced context management settings

    # Researcher context isolation configuration (Phase 3)
    enable_researcher_isolation: bool = False  # Whether to enable researcher context isolation
    researcher_isolation_level: str = "moderate"  # Isolation level: minimal, moderate, aggressive
    researcher_max_local_context: int = 5000  # Maximum local context size for researcher
    researcher_isolation_threshold: int = 3  # Enable isolation when steps exceed this value
    researcher_auto_isolation: bool = True  # Enable automatic isolation based on complexity
    researcher_isolation_metrics: bool = True  # Enable isolation metrics collection
    max_context_steps_researcher: int = 2  # Maximum context steps for researcher isolation

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
