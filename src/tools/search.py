# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import Any, Dict, List

from langchain_community.tools import BraveSearch, DuckDuckGoSearchResults
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper, BraveSearchWrapper
from langchain_core.tools import BaseTool
from pydantic import Field

from src.config import SearchEngine, SELECTED_SEARCH_ENGINE
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)
from src.tools.decorators import create_logged_tool

logger = logging.getLogger(__name__)

# Create logged versions of the search tools
LoggedTavilySearch = create_logged_tool(TavilySearchResultsWithImages)
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
LoggedBraveSearch = create_logged_tool(BraveSearch)
LoggedArxivSearch = create_logged_tool(ArxivQueryRun)


class SmartSearchTool(BaseTool):
    """Smart search tool wrapper that applies SearchResultFilter to search results."""

    name: str = "web_search"
    description: str = "Search the web for information with intelligent filtering"
    base_tool: BaseTool = Field(..., description="The underlying search tool")
    enable_smart_filtering: bool = Field(
        default=True, description="Whether to enable smart filtering"
    )
    max_search_results: int = Field(
        default=10, description="Maximum number of search results"
    )

    def _run(self, query: str, **kwargs) -> str:
        """Execute search with smart filtering."""
        try:
            # Get raw search results from the base tool
            raw_results = self.base_tool._run(query, **kwargs)

            # If smart filtering is disabled, return raw results
            if not self.enable_smart_filtering:
                return raw_results

            # Try to parse results for filtering
            try:
                if isinstance(raw_results, str):
                    search_results = json.loads(raw_results)
                else:
                    search_results = raw_results

                # Only apply filtering if results are in list format (structured results)
                if isinstance(search_results, list) and len(search_results) > 0:
                    return self._apply_smart_filtering(query, search_results)
                else:
                    # Return raw results if not in expected format
                    return raw_results

            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse search results for filtering: {e}")
                return raw_results

        except Exception as e:
            logger.error(f"Smart search tool execution failed: {e}")
            # Fallback to base tool execution
            return self.base_tool._run(query, **kwargs)

    def _apply_smart_filtering(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> str:
        """Apply SearchResultFilter to search results."""
        try:
            from src.utils.content_processor import ContentProcessor
            from src.utils.search_result_filter import SearchResultFilter

            # Initialize content processor and filter
            processor = ContentProcessor()
            filter_instance = SearchResultFilter(processor)

            # Use default model name for content processing
            model_name = "deepseek-chat"  # Default model name for content processing

            # Check if smart filtering should be enabled
            should_filter = filter_instance.should_enable_smart_filtering(
                search_results, model_name
            )

            if should_filter:
                logger.info(
                    f"Applying smart filtering to {len(search_results)} search results"
                )

                # Apply smart filtering
                filtered_data = filter_instance.filter_search_results(
                    query=query,
                    search_results=search_results,
                    model_name=model_name,
                    max_results=self.max_search_results,
                )

                # Return filtered results as JSON string
                return json.dumps(filtered_data, ensure_ascii=False, indent=2)
            else:
                logger.info(
                    "Smart filtering threshold not met, returning original results"
                )
                return json.dumps(search_results, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Smart filtering failed: {e}")
            logger.error(f"Full traceback: {__import__('traceback').format_exc()}")
            # Return original results if filtering fails
            return json.dumps(search_results, ensure_ascii=False, indent=2)

    async def _arun(self, query: str, **kwargs) -> str:
        """Async version of _run."""
        # For now, use sync version
        return self._run(query, **kwargs)


# Get the selected search tool with smart filtering
def get_web_search_tool(max_search_results: int, enable_smart_filtering: bool = True):
    """Get web search tool with optional smart filtering.

    Args:
        max_search_results: Maximum number of search results
        enable_smart_filtering: Whether to enable SearchResultFilter (default: True)

    Returns:
        SmartSearchTool wrapper with the selected search engine
    """
    # Get the base search tool based on selected engine
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        base_tool = LoggedTavilySearch(
            name="web_search_base",
            max_results=max_search_results,
            include_raw_content=False,
            include_images=True,
            include_image_descriptions=True,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        base_tool = LoggedDuckDuckGoSearch(
            name="web_search_base",
            num_results=max_search_results,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        base_tool = LoggedBraveSearch(
            name="web_search_base",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": max_search_results},
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        base_tool = LoggedArxivSearch(
            name="web_search_base",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,
                load_max_docs=max_search_results,
                load_all_available_meta=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")

    # Wrap with SmartSearchTool for intelligent filtering
    return SmartSearchTool(
        base_tool=base_tool,
        enable_smart_filtering=enable_smart_filtering,
        max_search_results=max_search_results,
    )


def get_raw_web_search_tool(max_search_results: int):
    """Get raw web search tool without smart filtering (for backward compatibility).

    Args:
        max_search_results: Maximum number of search results

    Returns:
        Raw search tool without SmartSearchTool wrapper
    """
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        return LoggedTavilySearch(
            name="web_search",
            max_results=max_search_results,
            include_raw_content=False,
            include_images=True,
            include_image_descriptions=True,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        return LoggedDuckDuckGoSearch(
            name="web_search",
            num_results=max_search_results,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        return LoggedBraveSearch(
            name="web_search",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": max_search_results},
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        return LoggedArxivSearch(
            name="web_search",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,
                load_max_docs=max_search_results,
                load_all_available_meta=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")
