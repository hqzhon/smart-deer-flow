"""Research result processor module - Process and integrate research results"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResearchResultProcessor:
    """Research result processor - Process and integrate research results"""

    def __init__(self, unified_config: Any):
        """Initialize research result processor

        Args:
            unified_config: Unified configuration object
        """
        self.unified_config = unified_config
        self.processed_results = []

        logger.info("ResearchResultProcessor initialized")

    def process_search_result(
        self, result: Any, query: str, iteration: int
    ) -> Dict[str, Any]:
        """Process single search result

        Args:
            result: Search result
            query: Query content
            iteration: Iteration count

        Returns:
            Processed result dictionary
        """
        try:
            processed_result = {
                "query": query,
                "iteration": iteration,
                "timestamp": datetime.now(),
                "raw_result": result,
                "processed_content": None,
                "metadata": {},
                "quality_score": 0.0,
            }

            # Extract content
            content = self._extract_content(result)
            processed_result["processed_content"] = content

            # Extract metadata
            metadata = self._extract_metadata(result)
            processed_result["metadata"] = metadata

            # Calculate quality score
            quality_score = self._calculate_quality_score(content, metadata)
            processed_result["quality_score"] = quality_score

            logger.debug(
                f"Processed search result for query '{query}' with quality score {quality_score:.2f}"
            )
            return processed_result

        except Exception as e:
            logger.error(f"Failed to process search result for query '{query}': {e}")
            return {
                "query": query,
                "iteration": iteration,
                "timestamp": datetime.now(),
                "raw_result": result,
                "processed_content": None,
                "metadata": {"error": str(e)},
                "quality_score": 0.0,
            }

    def _extract_content(self, result: Any) -> Optional[str]:
        """Extract content from result

        Args:
            result: Search result

        Returns:
            Extracted content string
        """
        if not result:
            return None

        # Handle string result
        if isinstance(result, str):
            return result.strip()

        # Handle dictionary result
        if isinstance(result, dict):
            # Try multiple possible content fields
            content_fields = ["content", "text", "body", "description", "summary"]
            for field in content_fields:
                if field in result and result[field]:
                    content = result[field]
                    if isinstance(content, str):
                        return content.strip()

        # Handle list result
        if isinstance(result, list) and result:
            # If it's a list, try to extract content from the first valid item
            for item in result:
                content = self._extract_content(item)
                if content:
                    return content

        # If unable to extract, return string representation
        try:
            return str(result).strip()
        except Exception:
            return None

    def _extract_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract metadata from result

        Args:
            result: Search result

        Returns:
            Metadata dictionary
        """
        metadata = {}

        if isinstance(result, dict):
            # Extract common metadata fields
            metadata_fields = [
                "url",
                "title",
                "source",
                "author",
                "date",
                "score",
                "relevance",
                "confidence",
                "type",
                "category",
            ]

            for field in metadata_fields:
                if field in result:
                    metadata[field] = result[field]

            # Calculate content length
            content = self._extract_content(result)
            if content:
                metadata["content_length"] = len(content)
                metadata["word_count"] = len(content.split())

        elif isinstance(result, str):
            metadata["content_length"] = len(result)
            metadata["word_count"] = len(result.split())
            metadata["type"] = "text"

        return metadata

    def _calculate_quality_score(
        self, content: Optional[str], metadata: Dict[str, Any]
    ) -> float:
        """Calculate result quality score

        Args:
            content: Content string
            metadata: Metadata dictionary

        Returns:
            Quality score (0.0 - 1.0)
        """
        score = 0.0

        # Content quality scoring
        if content:
            # Based on content length
            content_length = len(content)
            if content_length > 100:
                score += 0.3
            elif content_length > 50:
                score += 0.2
            elif content_length > 20:
                score += 0.1

            # Based on word count
            word_count = len(content.split())
            if word_count > 50:
                score += 0.2
            elif word_count > 20:
                score += 0.1

            # Check content quality indicators
            if any(
                keyword in content.lower()
                for keyword in ["detail", "comprehensive", "thorough", "detailed"]
            ):
                score += 0.1

        # Metadata quality scoring
        if metadata.get("url"):
            score += 0.1
        if metadata.get("title"):
            score += 0.1
        if metadata.get("source"):
            score += 0.1
        if metadata.get("score") and isinstance(metadata["score"], (int, float)):
            # If there's a raw score, normalize and weight it
            raw_score = float(metadata["score"])
            normalized_score = min(raw_score / 100.0, 1.0)  # Assume max raw score is 100
            score += normalized_score * 0.2

        return min(score, 1.0)

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple research results

        Args:
            results: List of results

        Returns:
            Aggregated result dictionary
        """
        if not results:
            return {
                "total_results": 0,
                "average_quality": 0.0,
                "aggregated_content": "",
                "metadata_summary": {},
                "quality_distribution": {},
            }

        # Calculate statistics
        total_results = len(results)
        quality_scores = [r.get("quality_score", 0.0) for r in results]
        average_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        # Aggregate content
        aggregated_content = self._aggregate_content(results)

        # Aggregate metadata
        metadata_summary = self._aggregate_metadata(results)

        # Quality distribution
        quality_distribution = self._calculate_quality_distribution(quality_scores)

        aggregated_result = {
            "total_results": total_results,
            "average_quality": average_quality,
            "aggregated_content": aggregated_content,
            "metadata_summary": metadata_summary,
            "quality_distribution": quality_distribution,
            "best_result": (
                max(results, key=lambda x: x.get("quality_score", 0.0))
                if results
                else None
            ),
            "aggregation_timestamp": datetime.now(),
        }

        logger.info(
            f"Aggregated {total_results} results with average quality {average_quality:.2f}"
        )
        return aggregated_result

    def _aggregate_content(self, results: List[Dict[str, Any]]) -> str:
        """Aggregate content

        Args:
            results: List of results

        Returns:
            Aggregated content string
        """
        contents = []

        # Sort by quality score, prioritize high-quality content
        sorted_results = sorted(
            results, key=lambda x: x.get("quality_score", 0.0), reverse=True
        )

        for result in sorted_results:
            content = result.get("processed_content")
            if content and content.strip():
                # Avoid duplicate content
                content_lower = content.lower()
                if not any(content_lower in existing.lower() for existing in contents):
                    contents.append(content.strip())

        # Limit total length
        max_length = getattr(
            self.unified_config, "max_aggregated_content_length", 10000
        )
        aggregated = "\n\n".join(contents)

        if len(aggregated) > max_length:
            aggregated = aggregated[:max_length] + "...[truncated]"

        return aggregated

    def _aggregate_metadata(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metadata

        Args:
            results: List of results

        Returns:
            Aggregated metadata dictionary
        """
        metadata_summary = {
            "sources": set(),
            "urls": set(),
            "total_content_length": 0,
            "total_word_count": 0,
            "content_types": {},
        }

        for result in results:
            metadata = result.get("metadata", {})

            if metadata.get("source"):
                metadata_summary["sources"].add(metadata["source"])

            if metadata.get("url"):
                metadata_summary["urls"].add(metadata["url"])

            if metadata.get("content_length"):
                metadata_summary["total_content_length"] += metadata["content_length"]

            if metadata.get("word_count"):
                metadata_summary["total_word_count"] += metadata["word_count"]

            content_type = metadata.get("type", "unknown")
            metadata_summary["content_types"][content_type] = (
                metadata_summary["content_types"].get(content_type, 0) + 1
            )

        # Convert sets to lists for serialization
        metadata_summary["sources"] = list(metadata_summary["sources"])
        metadata_summary["urls"] = list(metadata_summary["urls"])

        return metadata_summary

    def _calculate_quality_distribution(
        self, quality_scores: List[float]
    ) -> Dict[str, int]:
        """Calculate quality distribution

        Args:
            quality_scores: List of quality scores

        Returns:
            Quality distribution dictionary
        """
        distribution = {
            "excellent": 0,  # 0.8-1.0
            "good": 0,  # 0.6-0.8
            "fair": 0,  # 0.4-0.6
            "poor": 0,  # 0.0-0.4
        }

        for score in quality_scores:
            if score >= 0.8:
                distribution["excellent"] += 1
            elif score >= 0.6:
                distribution["good"] += 1
            elif score >= 0.4:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1

        return distribution

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary

        Returns:
            Processing summary dictionary
        """
        return {
            "processed_results_count": len(self.processed_results),
            "max_aggregated_content_length": getattr(
                self.unified_config, "max_aggregated_content_length", 10000
            ),
            "processor_initialized": True,
        }
