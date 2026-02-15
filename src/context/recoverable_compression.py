"""Recoverable compression for context management.

This module implements recoverable compression strategies from Manus AI,
ensuring that compressed content can be restored when needed.

Key principles:
1. Any irreversible compression carries risk
2. Compression strategies should always be recoverable
3. Keep URL/path to allow content recovery
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import re


@dataclass
class CompressedContent:
    """Compressed content with recovery info."""

    compressed: str
    compression_type: str
    recovery_info: Dict[str, Any] = field(default_factory=dict)
    original_size: int = 0
    compressed_size: int = 0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_size == 0:
            return 0.0
        return 1 - (self.compressed_size / self.original_size)


class RecoverableCompressor:
    """Recoverable compressor for context optimization.

    Key principle: Any irreversible compression carries risk.
    Always design compression to be recoverable.

    Recovery strategies:
    - Web content: Keep URL, content can be re-fetched
    - Documents: Keep file path, content can be re-read
    - Observations: Keep source reference
    """

    def __init__(self, max_key_points: int = 5, summary_max_length: int = 200):
        self.max_key_points = max_key_points
        self.summary_max_length = summary_max_length

    def compress_web_content(
        self, content: str, url: str, title: str = "Web Content"
    ) -> CompressedContent:
        """Compress web content with recovery capability.

        Recovery strategy: Keep URL, content can be re-fetched.

        Args:
            content: Web content to compress
            url: Source URL
            title: Content title

        Returns:
            CompressedContent instance
        """
        key_points = self._extract_key_points(content)

        compressed_lines = [
            f"**来源**: [{title}]({url})",
            "**关键点**:",
        ]
        for point in key_points[: self.max_key_points]:
            compressed_lines.append(f"- {point}")

        compressed = "\n".join(compressed_lines)

        return CompressedContent(
            compressed=compressed,
            compression_type="web_content",
            recovery_info={
                "url": url,
                "title": title,
                "original_hash": hash(content),
            },
            original_size=len(content),
            compressed_size=len(compressed),
        )

    def compress_document_content(
        self, content: str, file_path: str, doc_type: str = "document"
    ) -> CompressedContent:
        """Compress document content with recovery capability.

        Recovery strategy: Keep file path, content can be re-read.

        Args:
            content: Document content
            file_path: Path to the document
            doc_type: Type of document

        Returns:
            CompressedContent instance
        """
        summary = self._summarize_content(content)
        key_points = self._extract_key_points(content)

        compressed_lines = [
            f"**文档**: {file_path}",
            f"**类型**: {doc_type}",
            f"**摘要**: {summary}",
        ]

        if key_points:
            compressed_lines.append("**要点**:")
            for point in key_points[:3]:
                compressed_lines.append(f"- {point}")

        compressed = "\n".join(compressed_lines)

        return CompressedContent(
            compressed=compressed,
            compression_type="document",
            recovery_info={
                "file_path": file_path,
                "doc_type": doc_type,
                "original_hash": hash(content),
            },
            original_size=len(content),
            compressed_size=len(compressed),
        )

    def compress_observation_history(
        self, observations: List[str], keep_recent: int = 3
    ) -> Tuple[str, List[CompressedContent]]:
        """Compress observation history.

        Recovery strategy: Keep recent observations, compress older ones.

        Args:
            observations: List of observations
            keep_recent: Number of recent observations to keep

        Returns:
            Tuple of (compressed string, list of CompressedContent)
        """
        if not observations:
            return "", []

        compressed_items = []

        recent = (
            observations[-keep_recent:]
            if len(observations) > keep_recent
            else observations
        )
        old_observations = (
            observations[:-keep_recent] if len(observations) > keep_recent else []
        )

        compressed_old = []
        for obs in old_observations:
            source = self._extract_source(obs)
            if source:
                compressed = CompressedContent(
                    compressed=f"[已压缩: {source}]",
                    compression_type="observation",
                    recovery_info={"source": source},
                    original_size=len(obs),
                    compressed_size=len(source) + 20,
                )
                compressed_items.append(compressed)
                compressed_old.append(compressed.compressed)

        result_parts = list(recent)
        if compressed_old:
            result_parts.append("")
            result_parts.append("---")
            result_parts.append("**已压缩的旧观察**:")
            result_parts.extend(compressed_old)

        return "\n".join(result_parts), compressed_items

    def compress_tool_result(
        self, result: str, tool_name: str, args: Dict[str, Any]
    ) -> CompressedContent:
        """Compress tool execution result.

        Args:
            result: Tool result
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            CompressedContent instance
        """
        if tool_name in ("web_search", "tavily_search", "duckduckgo_search"):
            url = args.get("url", "") or self._extract_url(result)
            return self.compress_web_content(
                result, url, f"Search: {args.get('query', '')}"
            )

        elif tool_name == "crawl_tool":
            url = args.get("url", "")
            return self.compress_web_content(result, url, "Crawled Content")

        elif tool_name in ("python_repl", "shell_execute"):
            return self.compress_document_content(
                result, f"execution:{tool_name}", "execution_result"
            )

        else:
            summary = self._summarize_content(result)
            return CompressedContent(
                compressed=f"[{tool_name}] {summary}",
                compression_type="tool_result",
                recovery_info={"tool_name": tool_name, "args": args},
                original_size=len(result),
                compressed_size=len(summary) + len(tool_name) + 10,
            )

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        sentences = re.split(r"[。.!?\n]", content)

        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                key_points.append(sentence)
                if len(key_points) >= self.max_key_points * 2:
                    break

        return key_points

    def _summarize_content(self, content: str) -> str:
        """Summarize content."""
        if len(content) <= self.summary_max_length:
            return content

        summary = content[: self.summary_max_length]

        last_period = max(summary.rfind("。"), summary.rfind("."), summary.rfind("!"))

        if last_period > self.summary_max_length // 2:
            summary = summary[: last_period + 1]
        else:
            summary = summary + "..."

        return summary

    def _extract_source(self, observation: str) -> Optional[str]:
        """Extract source from observation."""
        url_match = re.search(r'https?://[^\s<>"]+', observation)
        if url_match:
            return url_match.group(0)

        file_match = re.search(
            r"(?:file|document|doc):\s*([^\s]+)", observation, re.IGNORECASE
        )
        if file_match:
            return file_match.group(1)

        return None

    def _extract_url(self, content: str) -> str:
        """Extract URL from content."""
        match = re.search(r'https?://[^\s<>"]+', content)
        return match.group(0) if match else ""


class ContextManagerWithCompression:
    """Context manager with compression support.

    Manages context size by applying recoverable compression
    when token limits are approached.
    """

    def __init__(
        self,
        max_tokens: int,
        compressor: Optional[RecoverableCompressor] = None,
        compression_threshold: float = 0.9,
    ):
        self.max_tokens = max_tokens
        self.compressor = compressor or RecoverableCompressor()
        self.compression_threshold = compression_threshold
        self.compressed_items: List[CompressedContent] = []

    def optimize_context(
        self, messages: List[Dict[str, Any]], current_tokens: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Optimize context by applying compression.

        Args:
            messages: Message list
            current_tokens: Current token count

        Returns:
            Tuple of (optimized messages, new token count)
        """
        if current_tokens <= self.max_tokens * self.compression_threshold:
            return messages, current_tokens

        messages = [m.copy() for m in messages]
        overflow = current_tokens - int(self.max_tokens * self.compression_threshold)

        for i, msg in enumerate(messages):
            if overflow <= 0:
                break

            if msg.get("role") == "tool":
                content = msg.get("content", "")

                if len(content) > 500:
                    tool_name = msg.get("name", "unknown")
                    compressed = self.compressor.compress_tool_result(
                        content, tool_name, {}
                    )

                    self.compressed_items.append(compressed)
                    msg["content"] = compressed.compressed

                    char_reduction = (
                        compressed.original_size - compressed.compressed_size
                    )
                    token_reduction = char_reduction // 4
                    overflow -= token_reduction

        new_tokens = current_tokens - overflow
        return messages, max(new_tokens, 0)

    def get_compressed_items(self) -> List[CompressedContent]:
        """Get list of compressed items."""
        return self.compressed_items.copy()

    def clear_compressed_items(self) -> None:
        """Clear compressed items list."""
        self.compressed_items.clear()

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self.compressed_items:
            return {
                "total_items": 0,
                "total_original_size": 0,
                "total_compressed_size": 0,
                "average_ratio": 0.0,
            }

        total_original = sum(c.original_size for c in self.compressed_items)
        total_compressed = sum(c.compressed_size for c in self.compressed_items)

        return {
            "total_items": len(self.compressed_items),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "average_ratio": (
                1 - (total_compressed / total_original) if total_original > 0 else 0
            ),
        }


def compress_content(
    content: str,
    content_type: str = "general",
    metadata: Optional[Dict[str, Any]] = None,
) -> CompressedContent:
    """Convenience function to compress content.

    Args:
        content: Content to compress
        content_type: Type of content
        metadata: Additional metadata

    Returns:
        CompressedContent instance
    """
    compressor = RecoverableCompressor()
    metadata = metadata or {}

    if content_type == "web":
        return compressor.compress_web_content(
            content, metadata.get("url", ""), metadata.get("title", "Web Content")
        )
    elif content_type == "document":
        return compressor.compress_document_content(
            content,
            metadata.get("file_path", "unknown"),
            metadata.get("doc_type", "document"),
        )
    else:
        summary = compressor._summarize_content(content)
        return CompressedContent(
            compressed=summary,
            compression_type="general",
            recovery_info=metadata,
            original_size=len(content),
            compressed_size=len(summary),
        )
