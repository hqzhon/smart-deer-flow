"""Follow-up Query Result Merging Mechanism

This module provides intelligent Follow-up query result merging functionality, including:
- Content deduplication and similarity detection
- Intelligent result priority ranking
- Structured data merging
- Quality assessment and filtering
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MergedResult:
    """Merged result data structure"""

    content: str
    sources: List[str]
    confidence_score: float
    relevance_score: float
    key_points: List[str]
    metadata: Dict[str, Any]
    original_count: int
    merged_count: int


@dataclass
class ResultMetrics:
    """Result quality metrics"""

    content_length: int
    unique_information_ratio: float
    source_diversity: int
    temporal_relevance: float
    structural_completeness: float


class FollowUpResultMerger:
    """Follow-up query result intelligent merger"""

    def __init__(
        self,
        config: Optional[Any] = None,
        similarity_threshold: Optional[float] = None,
        min_content_length: Optional[int] = None,
        max_merged_results: Optional[int] = None,
        enable_semantic_grouping: Optional[bool] = None,
    ):
        """
        Initialize merger

        Args:
            config: Merger configuration object (highest priority)
            similarity_threshold: Content similarity threshold (backward compatibility)
            min_content_length: Minimum content length (backward compatibility)
            max_merged_results: Maximum merged results count (backward compatibility)
            enable_semantic_grouping: Whether to enable semantic grouping (backward compatibility)
        """
        # Import configuration (lazy import to avoid circular dependencies)
        if config is None:
            try:
                # Prefer new unified configuration system
                from src.config.config_loader import get_settings

                app_settings = get_settings()
                config = app_settings.get_followup_merger_config()
            except ImportError:
                try:
                    # Fallback to old configuration system
                    from src.config.follow_up_merger_config import (
                        get_active_merger_config,
                    )

                    config = get_active_merger_config()
                except ImportError:
                    # If configuration module is unavailable, use default values
                    from src.config.follow_up_merger_config import FollowUpMergerConfig

                    config = FollowUpMergerConfig()

        # Apply parameter overrides (backward compatibility)
        self.config = config
        if similarity_threshold is not None:
            self.config.similarity_threshold = similarity_threshold
        if min_content_length is not None:
            self.config.min_content_length = min_content_length
        if max_merged_results is not None:
            self.config.max_merged_results = max_merged_results
        if enable_semantic_grouping is not None:
            self.config.enable_semantic_grouping = enable_semantic_grouping

        # Configuration is automatically validated in Pydantic model

        # Content fingerprint cache
        self._content_fingerprints: Set[str] = set()
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

        # Performance statistics
        self._stats = {
            "total_merges": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_count": 0,
            "quality_filtered_count": 0,
        }

    def merge_follow_up_results(
        self,
        follow_up_results: List[Dict[str, Any]],
        original_findings: List[str],
        query_context: Optional[str] = None,
    ) -> List[MergedResult]:
        """
        Merge Follow-up query results

        Args:
            follow_up_results: Raw results from Follow-up queries
            original_findings: Original research findings
            query_context: Query context

        Returns:
            List of merged results
        """
        logger.info(f"Starting to merge {len(follow_up_results)} Follow-up results")

        # Update statistics
        self._stats["total_merges"] += 1

        # 1. Preprocessing and normalization
        normalized_results = self._normalize_results(follow_up_results)

        # 2. Content deduplication
        deduplicated_results = self._deduplicate_content(normalized_results)

        # 3. Deduplication against original findings
        filtered_results = self._filter_against_original(
            deduplicated_results, original_findings
        )

        # 4. Semantic grouping (if enabled)
        if self.config.enable_semantic_grouping:
            grouped_results = self._group_by_semantic_similarity(filtered_results)
        else:
            grouped_results = [[result] for result in filtered_results]

        # 5. Intra-group merging
        merged_groups = []
        for group in grouped_results:
            merged_result = self._merge_group(group, query_context)
            if merged_result:
                merged_groups.append(merged_result)

        # 6. Quality assessment and ranking
        scored_results = self._score_and_rank_results(merged_groups)

        # 7. Final filtering and quantity limitation
        final_results = self._apply_final_filters(scored_results)

        logger.info(
            f"Merging completed, merged {len(follow_up_results)} results into {len(final_results)} results"
        )
        return final_results

    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize result format"""
        normalized = []

        for i, result in enumerate(results):
            # Extract content
            content = ""
            if isinstance(result, dict):
                content = result.get("content", "")
                if not content:
                    # Try other fields
                    content = result.get("observation", "")
                    if not content and "update" in result:
                        update = result["update"]
                        if isinstance(update, dict):
                            observations = update.get("observations", [])
                            if observations:
                                content = " ".join(
                                    (
                                        obs.get("content", str(obs))
                                        if isinstance(obs, dict)
                                        else str(obs)
                                    )
                                    for obs in observations
                                )
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)

            # Clean content
            content = self._clean_content(content)

            if len(content) >= self.config.min_content_length:
                normalized.append(
                    {
                        "content": content,
                        "source": (
                            result.get("source", f"follow_up_{i+1}")
                            if isinstance(result, dict)
                            else f"follow_up_{i+1}"
                        ),
                        "metadata": result if isinstance(result, dict) else {},
                        "original_index": i,
                    }
                )

        return normalized

    def _clean_content(self, content: str) -> str:
        """Clean content text"""
        if not content:
            return ""

        # Remove Follow-up markers
        content = re.sub(r"\[Follow-up \d+\.\d+\]\s*", "", content)

        # Remove excess whitespace characters
        content = re.sub(r"\s+", " ", content).strip()

        # Remove duplicate sentence beginnings
        lines = content.split("\n")
        cleaned_lines = []
        prev_line = ""

        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line

        return "\n".join(cleaned_lines)

    def _deduplicate_content(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate based on content fingerprint"""
        deduplicated = []
        seen_fingerprints = set()

        for result in results:
            content = result["content"]
            fingerprint = self._generate_content_fingerprint(content)

            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                deduplicated.append(result)
            else:
                logger.debug(f"Found duplicate content, skipped: {content[:100]}...")

        logger.info(
            f"After deduplication, retained {len(deduplicated)}/{len(results)} results"
        )
        return deduplicated

    def _generate_content_fingerprint(self, content: str) -> str:
        """Generate content fingerprint"""
        # Normalize text
        normalized = re.sub(r"\W+", " ", content.lower()).strip()

        # Extract keywords (remove stop words)
        words = normalized.split()
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "是",
            "的",
            "了",
            "在",
            "有",
            "和",
            "与",
        }
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Take first 20 keywords to generate fingerprint
        key_content = " ".join(sorted(keywords[:20]))
        return hashlib.md5(key_content.encode()).hexdigest()

    def _filter_against_original(
        self, results: List[Dict[str, Any]], original_findings: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter content that duplicates original findings"""
        if not original_findings:
            return results

        # Check if duplicates original findings
        original_fingerprints = set()
        for finding in original_findings:
            if (
                isinstance(finding, str)
                and len(finding) >= self.config.min_content_length
            ):
                fingerprint = self._generate_content_fingerprint(finding)
                original_fingerprints.add(fingerprint)

        # Filter duplicate content
        filtered = []
        for result in results:
            content_fingerprint = self._generate_content_fingerprint(result["content"])

            # Check if duplicates original findings
            is_duplicate = content_fingerprint in original_fingerprints

            # Check similarity
            if not is_duplicate:
                max_similarity = 0.0
                for finding in original_findings:
                    if isinstance(finding, str):
                        similarity = self._calculate_similarity(
                            result["content"], finding
                        )
                        max_similarity = max(max_similarity, similarity)

                is_duplicate = max_similarity > self.config.similarity_threshold

            if not is_duplicate:
                filtered.append(result)
            else:
                self._stats["deduplication_count"] += 1
                logger.debug(
                    f"Filtered duplicate content: {result['content'][:100]}..."
                )

        logger.info(
            f"After original findings filtering, retained {len(filtered)}/{len(results)} results"
        )
        return filtered

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (with caching)
        """
        # Create cache key
        cache_key = (hash(text1), hash(text2))
        if cache_key[0] > cache_key[1]:
            cache_key = (cache_key[1], cache_key[0])

        # Check cache
        if cache_key in self._similarity_cache:
            self._stats["cache_hits"] += 1
            return self._similarity_cache[cache_key]

        self._stats["cache_misses"] += 1

        try:
            # Use simple vocabulary overlap calculation
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0

            # Cache result
            self._similarity_cache[cache_key] = similarity

            # Limit cache size
            if len(self._similarity_cache) > 1000:
                # Clean oldest half of cache
                keys_to_remove = list(self._similarity_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._similarity_cache[key]

            return similarity

        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def _group_by_semantic_similarity(
        self, results: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group based on semantic similarity"""
        if not results:
            return []

        groups = []
        ungrouped = results.copy()

        while ungrouped:
            # Take the first one as group seed
            seed = ungrouped.pop(0)
            current_group = [seed]

            # Find similar results
            remaining = []
            for result in ungrouped:
                similarity = self._calculate_similarity(
                    seed["content"], result["content"]
                )

                if similarity > self.config.similarity_threshold:
                    current_group.append(result)
                else:
                    remaining.append(result)

            ungrouped = remaining
            groups.append(current_group)

        logger.info(
            f"Semantic grouping completed: {len(results)} results grouped into {len(groups)} groups"
        )
        return groups

    def _merge_group(
        self, group: List[Dict[str, Any]], query_context: Optional[str] = None
    ) -> Optional[MergedResult]:
        """Merge results within the same group"""
        if not group:
            return None

        if len(group) == 1:
            # Single result direct conversion
            result = group[0]
            return MergedResult(
                content=result["content"],
                sources=[result["source"]],
                confidence_score=0.8,
                relevance_score=0.7,
                key_points=self._extract_key_points(result["content"]),
                metadata=result["metadata"],
                original_count=1,
                merged_count=1,
            )

        # Multiple results need merging
        merged_content = self._merge_content([r["content"] for r in group])
        sources = [r["source"] for r in group]

        # Merge metadata
        merged_metadata = {}
        for result in group:
            merged_metadata.update(result["metadata"])

        # Calculate confidence and relevance scores
        confidence_score = min(
            0.9, 0.6 + len(group) * 0.1
        )  # More sources = higher confidence
        relevance_score = self._calculate_relevance_score(merged_content, query_context)

        return MergedResult(
            content=merged_content,
            sources=sources,
            confidence_score=confidence_score,
            relevance_score=relevance_score,
            key_points=self._extract_key_points(merged_content),
            metadata=merged_metadata,
            original_count=len(group),
            merged_count=1,
        )

    def _merge_content(self, contents: List[str]) -> str:
        """Intelligently merge multiple contents"""
        if not contents:
            return ""

        if len(contents) == 1:
            return contents[0]

        # Sort by length, prioritize more detailed content
        sorted_contents = sorted(contents, key=len, reverse=True)

        # Extract all sentences
        all_sentences = []
        for content in sorted_contents:
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            all_sentences.extend(sentences)

        # Deduplicate sentences (maintain order)
        unique_sentences = []
        seen_sentences = set()

        for sentence in all_sentences:
            sentence_key = sentence.lower().strip()
            if sentence_key not in seen_sentences and len(sentence) > 10:
                seen_sentences.add(sentence_key)
                unique_sentences.append(sentence)

        # Reorganize content
        merged = ". ".join(unique_sentences)
        if merged and not merged.endswith("."):
            merged += "."

        return merged

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points"""
        if not content:
            return []

        sentences = [s.strip() for s in content.split(".") if s.strip()]

        # Simple key point extraction: select longer and information-rich sentences
        key_points = []
        for sentence in sentences[:5]:  # 最多5个关键点
            if (
                len(sentence) > 20
                and len(sentence) < 200
                and any(char.isdigit() or char.isupper() for char in sentence)
            ):
                key_points.append(sentence)

        return key_points[:3]  # Maximum 3 key points

    def _calculate_relevance_score(
        self, content: str, query_context: Optional[str] = None
    ) -> float:
        """Calculate relevance score"""
        if not query_context:
            return 0.7  # Default score

        # Simple keyword matching
        content_lower = content.lower()
        context_lower = query_context.lower()

        # Extract keywords
        context_words = set(re.findall(r"\w+", context_lower))
        content_words = set(re.findall(r"\w+", content_lower))

        if not context_words:
            return 0.7

        # Calculate intersection ratio
        intersection = context_words.intersection(content_words)
        relevance = len(intersection) / len(context_words)

        return min(1.0, relevance + 0.3)  # Base score + matching degree

    def _score_and_rank_results(
        self, results: List[MergedResult]
    ) -> List[MergedResult]:
        """Score and rank results"""
        if not results:
            return []

        # Calculate composite score
        for result in results:
            # Composite score = confidence * 0.4 + relevance * 0.4 + content quality * 0.2
            content_quality = self._calculate_content_quality(result.content)

            composite_score = (
                result.confidence_score * 0.4
                + result.relevance_score * 0.4
                + content_quality * 0.2
            )

            # Update metadata
            result.metadata["composite_score"] = composite_score
            result.metadata["content_quality"] = content_quality

        # Sort by composite score
        sorted_results = sorted(
            results, key=lambda x: x.metadata.get("composite_score", 0), reverse=True
        )

        return sorted_results

    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score"""
        if not content:
            return 0.0

        score = 0.0

        # Length score (moderate length is better)
        length = len(content)
        if 100 <= length <= 500:
            score += 0.3
        elif 500 < length <= 1000:
            score += 0.2
        elif length > 50:
            score += 0.1

        # Information density score
        if re.search(r"\d+", content):  # Contains numbers
            score += 0.2

        if re.search(r"[A-Z]{2,}", content):  # Contains abbreviations
            score += 0.1

        # Structural degree
        sentences = content.split(".")
        if len(sentences) >= 3:
            score += 0.2

        # Technical term density
        words = content.split()
        long_words = [w for w in words if len(w) > 6]
        if len(long_words) / len(words) > 0.2:
            score += 0.2

        return min(1.0, score)

    def _apply_final_filters(self, results: List[MergedResult]) -> List[MergedResult]:
        """Apply final filters"""
        if not results:
            return []

        # Filter low-quality results
        filtered = [
            result
            for result in results
            if result.metadata.get("content_quality", 0)
            >= self.config.quality_threshold
        ]

        # Count filtered results
        self._stats["quality_filtered_count"] += len(results) - len(filtered)

        # Limit quantity
        final_results = filtered[: self.config.max_merged_results]

        logger.info(
            f"Final filtering: {len(results)} -> {len(filtered)} -> {len(final_results)}"
        )
        return final_results

    def get_merge_statistics(self, results: List[MergedResult]) -> Dict[str, Any]:
        """Get merge statistics"""
        if not results:
            return {}

        total_original = sum(r.original_count for r in results)
        total_merged = len(results)

        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)

        source_diversity = len(
            set(source for result in results for source in result.sources)
        )

        return {
            "total_original_results": total_original,
            "total_merged_results": total_merged,
            "compression_ratio": (
                total_original / total_merged if total_merged > 0 else 0
            ),
            "average_confidence": avg_confidence,
            "average_relevance": avg_relevance,
            "source_diversity": source_diversity,
            "quality_distribution": {
                "high": len(
                    [r for r in results if r.metadata.get("content_quality", 0) > 0.7]
                ),
                "medium": len(
                    [
                        r
                        for r in results
                        if 0.4 <= r.metadata.get("content_quality", 0) <= 0.7
                    ]
                ),
                "low": len(
                    [r for r in results if r.metadata.get("content_quality", 0) < 0.4]
                ),
            },
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        """
        return {
            "total_merges": self._stats["total_merges"],
            "cache_performance": {
                "hits": self._stats["cache_hits"],
                "misses": self._stats["cache_misses"],
                "hit_rate": (
                    self._stats["cache_hits"]
                    / (self._stats["cache_hits"] + self._stats["cache_misses"])
                    if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                    else 0
                ),
                "cache_size": len(self._similarity_cache),
            },
            "filtering_stats": {
                "deduplication_count": self._stats["deduplication_count"],
                "quality_filtered_count": self._stats["quality_filtered_count"],
            },
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "min_content_length": self.config.min_content_length,
                "max_merged_results": self.config.max_merged_results,
                "enable_semantic_grouping": self.config.enable_semantic_grouping,
                "enable_intelligent_merging": self.config.enable_intelligent_merging,
                "enable_deduplication": self.config.enable_deduplication,
                "enable_quality_filtering": self.config.enable_quality_filtering,
            },
        }

    def reset_stats(self):
        """
        Reset performance statistics
        """
        self._stats = {
            "total_merges": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_count": 0,
            "quality_filtered_count": 0,
        }
        self._similarity_cache.clear()
        self._content_fingerprints.clear()
