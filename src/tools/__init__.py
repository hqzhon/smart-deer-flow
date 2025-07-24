# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Tools package with new BaseTool interface.
All tools in this package are automatically discovered by the ToolRegistry.
"""

# Import new tool implementations
from .web_search_tool import WebSearchTool, ArxivSearchTool
from .crawl_tool import CrawlTool, ArticleExtractorTool
from .python_repl_tool import PythonREPLTool, DataAnalysisTool

# Make tools available for import
__all__ = [
    "WebSearchTool",
    "ArxivSearchTool",
    "CrawlTool",
    "ArticleExtractorTool",
    "PythonREPLTool",
    "DataAnalysisTool",
    # Legacy compatibility exports
    "crawl_tool",
    "python_repl_tool",
    "get_retriever_tool",
    "get_web_search_tool",
    "VolcengineTTS",
]

# Legacy compatibility imports are available through their respective modules
from .crawl import crawl_tool
from .python_repl import python_repl_tool
from .retriever import get_retriever_tool
from .search import get_web_search_tool
from .tts import VolcengineTTS
