# -*- coding: utf-8 -*-
"""
Smart Context Compressor - Semantic importance-based compression.

Implements intelligent compression strategies:
1. Preserve critical information (errors, decisions, conclusions)
2. Compress intermediate processes
3. Use LLM-assisted summarization when needed
4. Preserve Artifact handles for recoverability
"""

import logging
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompressionLevel(str, Enum):
    """Compression level determining how aggressively to compress."""

    LIGHT = "light"  # Keep 70%+ of content
    MEDIUM = "medium"  # Keep 40-70% of content
    HEAVY = "heavy"  # Keep < 40% of content


class ContentType(str, Enum):
    """Type of content being compressed."""

    RESEARCH_RESULT = "research_result"
    OBSERVATION = "observation"
    ERROR_LOG = "error_log"
    CONVERSATION = "conversation"
    TOOL_OUTPUT = "tool_output"
    MIXED = "mixed"


@dataclass
class Artifact:
    """An artifact that can be recovered later."""

    artifact_type: str  # "url", "file", "reference"
    value: str
    context: str = ""  # Brief context about the artifact

    def to_handle(self) -> str:
        """Convert to a handle string for compression."""
        if self.artifact_type == "url":
            return f"[Source: {self.value[:60]}{'...' if len(self.value) > 60 else ''}]"
        elif self.artifact_type == "file":
            return f"[File: {self.value}]"
        else:
            return f"[Ref: {self.value[:40]}...]"


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    artifacts_preserved: List[Artifact]
    compression_level: CompressionLevel
    key_points_preserved: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens


class ContentAnalyzer:
    """Analyzes content to identify important elements."""

    IMPORTANCE_KEYWORDS = [
        "error",
        "warning",
        "critical",
        "important",
        "conclusion",
        "decision",
        "result",
        "key finding",
        "significant",
        "note:",
        "summary",
        "recommendation",
        "action required",
    ]

    ARTIFACT_PATTERNS = {
        "url": r"https?://[^\s\)\]\>]+",
        "file": r"[a-zA-Z0-9_\-./]+\.(md|txt|json|py|js|ts|yaml|yml|csv)",
        "reference": r"\[[^\]]+\]\([^\)]+\)",
    }

    @classmethod
    def extract_key_sentences(cls, content: str, max_sentences: int = 5) -> List[str]:
        """Extract the most important sentences from content."""
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        scored_sentences = []
        for sentence in sentences:
            score = cls._calculate_importance_score(sentence)
            scored_sentences.append((sentence, score))

        scored_sentences.sort(key=lambda x: -x[1])
        return [s for s, _ in scored_sentences[:max_sentences]]

    @classmethod
    def extract_artifacts(cls, content: str) -> List[Artifact]:
        """Extract all artifacts from content."""
        artifacts = []

        for artifact_type, pattern in cls.ARTIFACT_PATTERNS.items():
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                artifacts.append(Artifact(artifact_type=artifact_type, value=match))

        return artifacts

    @classmethod
    def _calculate_importance_score(cls, sentence: str) -> float:
        """Calculate importance score for a sentence."""
        score = 0.0
        sentence_lower = sentence.lower()

        for keyword in cls.IMPORTANCE_KEYWORDS:
            if keyword in sentence_lower:
                score += 1.0

        if re.search(r"\d+", sentence):
            score += 0.5

        if re.search(r"[A-Z]{2,}", sentence):
            score += 0.3

        length_score = min(len(sentence) / 100, 1.0)
        score += length_score * 0.5

        return score


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""

    @abstractmethod
    def compress(
        self, content: str, target_tokens: int, content_type: ContentType
    ) -> CompressionResult:
        """Compress content to target token count."""
        pass


class LightCompressionStrategy(CompressionStrategy):
    """Light compression - keeps most content."""

    def compress(
        self, content: str, target_tokens: int, content_type: ContentType
    ) -> CompressionResult:
        original_tokens = len(content) // 4

        key_sentences = ContentAnalyzer.extract_key_sentences(content, max_sentences=5)
        artifacts = ContentAnalyzer.extract_artifacts(content)

        lines = content.split("\n")
        compressed_lines = []

        for line in lines:
            if any(kw in line.lower() for kw in ContentAnalyzer.IMPORTANCE_KEYWORDS):
                compressed_lines.append(line)
            elif len(compressed_lines) < target_tokens // 20:
                compressed_lines.append(line)

        compressed = "\n".join(compressed_lines)
        compressed_tokens = len(compressed) // 4

        return CompressionResult(
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=(
                compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            ),
            artifacts_preserved=artifacts,
            compression_level=CompressionLevel.LIGHT,
            key_points_preserved=key_sentences,
        )


class MediumCompressionStrategy(CompressionStrategy):
    """Medium compression - balances content and brevity."""

    def compress(
        self, content: str, target_tokens: int, content_type: ContentType
    ) -> CompressionResult:
        original_tokens = len(content) // 4

        key_sentences = ContentAnalyzer.extract_key_sentences(content, max_sentences=5)
        artifacts = ContentAnalyzer.extract_artifacts(content)

        compressed_parts = []

        compressed_parts.append("## Key Findings\n")
        for i, sentence in enumerate(key_sentences[:3], 1):
            compressed_parts.append(f"{i}. {sentence}\n")

        if artifacts:
            compressed_parts.append("\n## Sources\n")
            for artifact in artifacts[:10]:
                compressed_parts.append(f"- {artifact.to_handle()}\n")

        compressed = "".join(compressed_parts)
        compressed_tokens = len(compressed) // 4

        return CompressionResult(
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=(
                compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            ),
            artifacts_preserved=artifacts,
            compression_level=CompressionLevel.MEDIUM,
            key_points_preserved=key_sentences,
        )


class HeavyCompressionStrategy(CompressionStrategy):
    """Heavy compression - minimal content with artifact handles."""

    def compress(
        self, content: str, target_tokens: int, content_type: ContentType
    ) -> CompressionResult:
        original_tokens = len(content) // 4

        key_sentences = ContentAnalyzer.extract_key_sentences(content, max_sentences=3)
        artifacts = ContentAnalyzer.extract_artifacts(content)

        compressed_parts = []

        compressed_parts.append("Summary:\n")
        for sentence in key_sentences[:2]:
            compressed_parts.append(f"- {sentence[:100]}\n")

        if artifacts:
            compressed_parts.append(f"\n[{len(artifacts)} sources available]\n")

        compressed = "".join(compressed_parts)
        compressed_tokens = len(compressed) // 4

        return CompressionResult(
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=(
                compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            ),
            artifacts_preserved=artifacts,
            compression_level=CompressionLevel.HEAVY,
            key_points_preserved=key_sentences,
        )


class SmartContextCompressor:
    """Smart context compressor with semantic importance awareness.

    Features:
    - Multiple compression levels
    - Artifact preservation
    - Key point extraction
    - Content type awareness
    """

    def __init__(self):
        self._strategies = {
            CompressionLevel.LIGHT: LightCompressionStrategy(),
            CompressionLevel.MEDIUM: MediumCompressionStrategy(),
            CompressionLevel.HEAVY: HeavyCompressionStrategy(),
        }

    def compress(
        self,
        content: str,
        target_tokens: int,
        content_type: ContentType = ContentType.MIXED,
        preserve_artifacts: bool = True,
    ) -> CompressionResult:
        """Compress content to target token count.

        Args:
            content: Content to compress
            target_tokens: Target token count
            content_type: Type of content
            preserve_artifacts: Whether to preserve artifact handles

        Returns:
            CompressionResult with compressed content and metadata
        """
        if not content:
            return CompressionResult(
                compressed_content="",
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                artifacts_preserved=[],
                compression_level=CompressionLevel.LIGHT,
                key_points_preserved=[],
            )

        original_tokens = len(content) // 4

        if original_tokens <= target_tokens:
            return CompressionResult(
                compressed_content=content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                artifacts_preserved=(
                    ContentAnalyzer.extract_artifacts(content)
                    if preserve_artifacts
                    else []
                ),
                compression_level=CompressionLevel.LIGHT,
                key_points_preserved=ContentAnalyzer.extract_key_sentences(content),
            )

        compression_ratio = target_tokens / original_tokens

        if compression_ratio > 0.7:
            level = CompressionLevel.LIGHT
        elif compression_ratio > 0.4:
            level = CompressionLevel.MEDIUM
        else:
            level = CompressionLevel.HEAVY

        strategy = self._strategies[level]
        result = strategy.compress(content, target_tokens, content_type)

        if not preserve_artifacts:
            result.artifacts_preserved = []

        logger.info(
            f"Compressed content: {original_tokens} -> {result.compressed_tokens} tokens "
            f"({result.compression_ratio:.2%}) using {level.value} strategy"
        )

        return result

    def compress_batch(
        self,
        contents: List[str],
        total_target_tokens: int,
        content_type: ContentType = ContentType.MIXED,
    ) -> List[CompressionResult]:
        """Compress multiple content items with shared token budget.

        Args:
            contents: List of content items
            total_target_tokens: Total token budget
            content_type: Type of content

        Returns:
            List of CompressionResults
        """
        if not contents:
            return []

        total_original = sum(len(c) // 4 for c in contents)
        if total_original <= total_target_tokens:
            return [self.compress(c, len(c) // 4, content_type) for c in contents]

        weights = [len(c) // 4 for c in contents]
        total_weight = sum(weights)

        results = []
        for content, weight in zip(contents, weights):
            target = int(total_target_tokens * weight / total_weight)
            result = self.compress(content, target, content_type)
            results.append(result)

        return results

    def estimate_compression_ratio(
        self, content: str, target_tokens: int
    ) -> Tuple[float, CompressionLevel]:
        """Estimate compression ratio without actually compressing.

        Args:
            content: Content to analyze
            target_tokens: Target token count

        Returns:
            Tuple of (estimated_ratio, recommended_level)
        """
        original_tokens = len(content) // 4

        if original_tokens <= target_tokens:
            return 1.0, CompressionLevel.LIGHT

        ratio = target_tokens / original_tokens

        if ratio > 0.7:
            return ratio, CompressionLevel.LIGHT
        elif ratio > 0.4:
            return ratio, CompressionLevel.MEDIUM
        else:
            return ratio, CompressionLevel.HEAVY
