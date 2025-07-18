"""
Web crawling tool implementation using the new BaseTool interface.
Provides web content extraction and processing capabilities.
"""

import logging
from typing import Any, Dict, List
from urllib.parse import urlparse

from src.tools.base_tool import BaseTool, ToolInput, ToolOutput

logger = logging.getLogger(__name__)


class CrawlInput(ToolInput):
    """Input model for crawl tool."""

    url: str
    max_length: int = 5000
    include_images: bool = False
    extract_main_content: bool = True


class CrawlTool(BaseTool):
    """Web crawling tool for content extraction."""

    @property
    def name(self) -> str:
        return "crawl"

    @property
    def description(self) -> str:
        return "Crawl and extract content from web pages"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "description": "URL to crawl and extract content from",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum content length to extract",
                "default": 5000,
                "minimum": 100,
                "maximum": 50000,
            },
            "include_images": {
                "type": "boolean",
                "description": "Include image URLs in extracted content",
                "default": False,
            },
            "extract_main_content": {
                "type": "boolean",
                "description": "Extract main content using readability",
                "default": True,
            },
        }

    @property
    def required_parameters(self) -> List[str]:
        return ["url"]

    @property
    def category(self) -> str:
        return "content"

    @property
    def tags(self) -> List[str]:
        return ["crawl", "extract", "web", "content"]

    def execute(
        self,
        url: str,
        max_length: int = 5000,
        include_images: bool = False,
        extract_main_content: bool = True,
    ) -> ToolOutput:
        """Execute web crawling.

        Args:
            url: URL to crawl
            max_length: Maximum content length
            include_images: Include images
            extract_main_content: Use readability extraction

        Returns:
            ToolOutput with extracted content
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return ToolOutput(success=False, message="Invalid URL format", data={})

            # Use the crawler module
            from src.crawler.crawler import Crawler

            crawler = Crawler()

            # Configure crawler options
            options = {
                "max_length": max_length,
                "include_images": include_images,
                "extract_main_content": extract_main_content,
            }

            # Crawl the URL
            result = crawler.crawl(url, **options)

            if result.get("success"):
                return ToolOutput(
                    success=True, message=f"Successfully crawled {url}", data=result
                )
            else:
                return ToolOutput(
                    success=False,
                    message=f"Failed to crawl {url}: {result.get('error', 'Unknown error')}",
                    data={},
                )

        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            return ToolOutput(success=False, message=f"Crawl failed: {str(e)}", data={})


class ArticleExtractorTool(BaseTool):
    """Article extraction tool for news and blog content."""

    @property
    def name(self) -> str:
        return "extract_article"

    @property
    def description(self) -> str:
        return "Extract clean article content from web pages using readability"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {"type": "string", "description": "URL of the article to extract"},
            "include_metadata": {
                "type": "boolean",
                "description": "Include article metadata (title, author, date)",
                "default": True,
            },
        }

    @property
    def required_parameters(self) -> List[str]:
        return ["url"]

    @property
    def category(self) -> str:
        return "content"

    @property
    def tags(self) -> List[str]:
        return ["article", "extract", "readability", "news"]

    def execute(self, url: str, include_metadata: bool = True) -> ToolOutput:
        """Extract article content.

        Args:
            url: Article URL
            include_metadata: Include metadata

        Returns:
            ToolOutput with extracted article
        """
        try:
            from src.crawler.article import ArticleExtractor

            extractor = ArticleExtractor()
            article = extractor.extract(url, include_metadata=include_metadata)

            if article.get("success"):
                return ToolOutput(
                    success=True,
                    message=f"Successfully extracted article from {url}",
                    data=article,
                )
            else:
                return ToolOutput(
                    success=False,
                    message=f"Failed to extract article: {article.get('error', 'Unknown error')}",
                    data={},
                )

        except Exception as e:
            logger.error(f"Article extraction failed: {e}")
            return ToolOutput(
                success=False, message=f"Article extraction failed: {str(e)}", data={}
            )
