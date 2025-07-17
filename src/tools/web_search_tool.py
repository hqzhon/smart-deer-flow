"""
Web search tool implementation using the new BaseTool interface.
Provides unified web search functionality with multiple search engines.
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional

from src.tools.base_tool import BaseTool, ToolInput, ToolOutput
from src.config import get_settings

logger = logging.getLogger(__name__)


class WebSearchInput(ToolInput):
    """Input model for web search tool."""
    query: str
    max_results: int = 5
    include_raw_content: bool = False
    include_images: bool = False


class WebSearchTool(BaseTool):
    """Web search tool with multiple search engine support."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information using various search engines"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query to execute"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5,
                "minimum": 1,
                "maximum": 20
            },
            "include_raw_content": {
                "type": "boolean",
                "description": "Include raw content from search results",
                "default": False
            },
            "include_images": {
                "type": "boolean",
                "description": "Include images in search results",
                "default": False
            }
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["query"]
    
    @property
    def category(self) -> str:
        return "search"
    
    @property
    def tags(self) -> List[str]:
        return ["web", "search", "information"]
    
    def execute(self, query: str, max_results: int = 5, 
                include_raw_content: bool = False, include_images: bool = False) -> ToolOutput:
        """Execute web search.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            include_raw_content: Include raw content
            include_images: Include images
            
        Returns:
            ToolOutput with search results
        """
        try:
            settings = get_settings()
            search_engine = settings.tools.search_engine
            
            if search_engine == "tavily":
                return self._search_tavily(query, max_results, include_raw_content, include_images)
            elif search_engine == "duckduckgo":
                return self._search_duckduckgo(query, max_results)
            elif search_engine == "brave":
                return self._search_brave(query, max_results)
            else:
                return ToolOutput(
                    success=False,
                    message=f"Unsupported search engine: {search_engine}",
                    data=[]
                )
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolOutput(
                success=False,
                message=f"Search failed: {str(e)}",
                data=[]
            )
    
    def _search_tavily(self, query: str, max_results: int, 
                      include_raw_content: bool, include_images: bool) -> ToolOutput:
        """Search using Tavily API."""
        try:
            from src.tools.tavily_search.tavily_search_results_with_images import TavilySearchResultsWithImages
            
            search_tool = TavilySearchResultsWithImages(
                max_results=max_results,
                include_raw_content=include_raw_content,
                include_images=include_images
            )
            
            results = search_tool.invoke({"query": query})
            
            return ToolOutput(
                success=True,
                message=f"Found {len(results)} results",
                data=results
            )
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return ToolOutput(
                success=False,
                message=f"Tavily search failed: {str(e)}",
                data=[]
            )
    
    def _search_duckduckgo(self, query: str, max_results: int) -> ToolOutput:
        """Search using DuckDuckGo."""
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            
            search_tool = DuckDuckGoSearchResults(num_results=max_results)
            results = search_tool.invoke(query)
            
            # Parse results
            if isinstance(results, str):
                try:
                    parsed_results = json.loads(results)
                except json.JSONDecodeError:
                    parsed_results = [{"content": results, "title": query}]
            else:
                parsed_results = results
            
            return ToolOutput(
                success=True,
                message=f"Found {len(parsed_results)} results",
                data=parsed_results
            )
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return ToolOutput(
                success=False,
                message=f"DuckDuckGo search failed: {str(e)}",
                data=[]
            )
    
    def _search_brave(self, query: str, max_results: int) -> ToolOutput:
        """Search using Brave Search API."""
        try:
            from langchain_community.tools import BraveSearch
            from langchain_community.utilities import BraveSearchWrapper
            
            api_key = os.getenv("BRAVE_SEARCH_API_KEY")
            if not api_key:
                return ToolOutput(
                    success=False,
                    message="Brave Search API key not found",
                    data=[]
                )
            
            search_tool = BraveSearch(
                search_wrapper=BraveSearchWrapper(
                    api_key=api_key,
                    search_kwargs={"count": max_results}
                )
            )
            
            results = search_tool.invoke(query)
            return ToolOutput(
                success=True,
                message=f"Found results via Brave Search",
                data=[{"content": results, "title": query}]
            )
            
        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return ToolOutput(
                success=False,
                message=f"Brave search failed: {str(e)}",
                data=[]
            )


class ArxivSearchTool(BaseTool):
    """ArXiv search tool for academic papers."""
    
    @property
    def name(self) -> str:
        return "arxiv_search"
    
    @property
    def description(self) -> str:
        return "Search academic papers on ArXiv"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Search query for academic papers"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of papers to return",
                "default": 5,
                "minimum": 1,
                "maximum": 20
            }
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["query"]
    
    @property
    def category(self) -> str:
        return "academic"
    
    @property
    def tags(self) -> List[str]:
        return ["arxiv", "academic", "papers", "research"]
    
    def execute(self, query: str, max_results: int = 5) -> ToolOutput:
        """Execute ArXiv search.
        
        Args:
            query: Search query
            max_results: Maximum papers to return
            
        Returns:
            ToolOutput with academic papers
        """
        try:
            from langchain_community.tools.arxiv import ArxivQueryRun
            from langchain_community.utilities import ArxivAPIWrapper
            
            search_tool = ArxivQueryRun(
                api_wrapper=ArxivAPIWrapper(
                    top_k_results=max_results,
                    load_max_docs=max_results,
                    load_all_available_meta=True
                )
            )
            
            results = search_tool.invoke(query)
            
            return ToolOutput(
                success=True,
                message=f"Found {max_results} academic papers",
                data=[{"content": results, "title": query}]
            )
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return ToolOutput(
                success=False,
                message=f"ArXiv search failed: {str(e)}",
                data=[]
            )