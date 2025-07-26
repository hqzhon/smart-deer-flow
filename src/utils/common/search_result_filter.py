# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..tokens.content_processor import ContentProcessor

logger = logging.getLogger(__name__)


class SearchResultCleaner:
    """Efficient local search result cleaner"""

    def __init__(self):
        # Improved noise patterns (match complete phrases)
        self.noise_patterns = [
            r"\b(?:click here|read more|learn more|see more|view all)\b[^.!?]*[.!?]?",
            r"\b(?:advertisement|sponsored|ads?)\b[^.!?]*[.!?]?",
            r"\b(?:cookie|privacy policy|terms of service)\b[^.!?]*[.!?]?",
            r"\b(?:subscribe|newsletter|sign up)\b[^.!?]*[.!?]?",
            r"\b(?:share|like|follow|tweet)\b[^.!?]*[.!?]?",
            r"\b(?:loading|please wait)\b[^.!?]*[.!?]?",
            r"\berror\s*:?\s*(?:loading|occurred|404|500|not found)\b[^.!?]*[.!?]?",  # 只匹配明确的错误消息
            r"\b(?:javascript|enable js|browser)\b[^.!?]*[.!?]?",
            # Contact information patterns
            r"\b(?:contact us|call us|email us)\b[^.!?]*[.!?]?",
            r"\b(?:at|or call)\s+[\w@.-]+\s*[.!?]?",
            # Timestamps, emails, phone numbers
            r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}",
            r"\b\w+@\w+\.\w+\b",
            r"\b\d{3}-\d{3}-\d{4}\b",
        ]

        # Compile regular expressions for better performance
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.noise_patterns
        ]

        # Stop words list (simplified version)
        self.stop_words = {
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
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        # Invalid domain patterns
        self.invalid_domains = {
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "linkedin.com",
            "pinterest.com",
            "youtube.com",
            "tiktok.com",
            "reddit.com",
        }

    def clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text or not isinstance(text, str):
            return ""

        # 0. Remove script and style content
        text = self._remove_scripts_and_styles(text)

        # 1. Basic cleaning
        text = self._basic_clean(text)

        # 2. Remove noise patterns
        text = self._remove_noise_patterns(text)

        # 3. Normalize whitespace
        text = self._normalize_whitespace(text)

        # 4. Remove content that is too short or too long
        if len(text.strip()) < 10 or len(text) > 5000:
            return ""

        return text.strip()

    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Handle garbled text and encoding issues
        text = self._fix_encoding_issues(text)

        # Remove special characters and control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        return text

    def _remove_noise_patterns(self, text: str) -> str:
        """Remove noise patterns"""
        for pattern in self.compiled_patterns:
            text = pattern.sub("", text)
        return self._post_clean_noise(text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple whitespace characters with single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading and trailing whitespace from lines
        text = "\n".join(line.strip() for line in text.split("\n"))

        # Remove excessive line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _post_clean_noise(self, text: str) -> str:
        """Post-processing of text after noise cleaning"""
        # Remove excessive conjunctions and punctuation
        text = re.sub(r"\b(?:at|or|call|to|and|the|a|an)\s+[.!?]", "", text)
        # Clean excessive spaces and punctuation
        text = re.sub(r"\s+[.!?]\s+", " ", text)
        # Remove isolated punctuation marks
        text = re.sub(r"\s+[.!?]$", "", text)
        text = re.sub(r"^[.!?]\s+", "", text)
        # Remove excessive spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _remove_scripts_and_styles(self, text: str) -> str:
        """Remove script and style content"""
        # Remove script tags and their content
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        # Remove style tags and their content
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        # Remove JavaScript code snippets
        text = re.sub(r"\balert\s*\([^)]*\)\s*;?", "", text)
        text = re.sub(r"\bconsole\.[a-z]+\s*\([^)]*\)\s*;?", "", text)
        return text

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix encoding issues and garbled text"""
        if not text:
            return text

        try:
            # 1. Handle Unicode replacement characters (clear garbled text markers)
            if "\ufffd" in text:
                text = text.replace("\ufffd", "")

            # 2. Handle HTML entity encoding (only process common HTML entities)
            import html

            # Only process clear HTML entities, avoid misprocessing normal text
            if "&" in text and (";" in text):
                # Check if it contains common HTML entities
                if any(
                    entity in text
                    for entity in ["&lt;", "&gt;", "&amp;", "&quot;", "&#"]
                ):
                    text = html.unescape(text)

            # 3. Handle URL encoding (more conservative approach)
            import urllib.parse

            # Only process when text looks like URL encoding
            if "%" in text:
                # Check for URL encoding patterns
                url_encoded_pattern = r"%[0-9A-Fa-f]{2}"
                url_matches = re.findall(url_encoded_pattern, text)
                if len(url_matches) >= 1:  # At least 1 encoded character
                    try:
                        decoded = urllib.parse.unquote(text, errors="ignore")
                        # Check if decoding makes sense:
                        # 1. Decoded result cannot be empty
                        # 2. Decoded result should contain readable characters
                        # 3. Prioritize decoding when URL encoded characters have high ratio, or decoded result is clearly more readable
                        url_encoded_ratio = len("".join(url_matches)) / len(text)
                        if (
                            decoded
                            and decoded.strip()
                            and (
                                len(decoded) > len(text) * 0.5
                                or url_encoded_ratio > 0.15
                                or (
                                    decoded != text
                                    and len(decoded.strip()) > len(text.strip()) * 0.8
                                )
                            )
                        ):
                            text = decoded
                    except Exception:
                        pass

            # 4. Remove clear garbled text patterns (more conservative)
            # Only remove 4 or more consecutive question marks (clear garbled text)
            text = re.sub(r"\?{4,}", "", text)

            # Remove block characters (both single and consecutive are garbled text)
            text = re.sub(r"[\u25a0\u25a1\u2588■▪▫]+", "", text)

            # Remove zero-width characters and other invisible characters
            text = re.sub(r"[\ufeff\u200b-\u200f\u2028\u2029]", "", text)

            # 5. Handle double encoding issues (more cautious)
            # Only process when clear double encoding patterns are detected
            double_encoding_indicators = ["Ã¡", "Ã©", "Ã­", "Ã³", "Ãº", "Ã±", "Ã§"]
            if any(indicator in text for indicator in double_encoding_indicators):
                # Calculate the density of these indicators
                indicator_count = sum(
                    text.count(indicator) for indicator in double_encoding_indicators
                )
                if indicator_count >= 2:  # Consider as double encoding only if appears at least 2 times
                    try:
                        text_bytes = text.encode("latin-1")
                        decoded = text_bytes.decode("utf-8", errors="ignore")
                        # Only use if decoded result looks more normal
                        if len(decoded) >= len(text) * 0.7:  # Decoded result cannot lose too much content
                            text = decoded
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass

            # 6. Remove control characters (preserve basic whitespace characters)
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

            # 7. Normalize special whitespace characters to regular spaces
            text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)

            # 8. Clean excessive spaces
            text = re.sub(r"\s+", " ", text).strip()

            return text

        except Exception as e:
            # If any error occurs during processing, return original text
            logger.warning(f"Error fixing encoding issues: {e}")
            return text

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords"""
        if not text:
            return []

        # Convert to lowercase and tokenize
        words = re.findall(r"\b\w{3,}\b", text.lower())

        # Filter stop words
        keywords = [word for word in words if word not in self.stop_words]

        # Count word frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        if not url or not isinstance(url, str):
            return False

        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False

            # Check if it's an invalid domain
            domain = parsed.netloc.lower()
            for invalid_domain in self.invalid_domains:
                if invalid_domain in domain:
                    return False

            return True
        except Exception:
            return False

    def calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score (0-1)"""
        if not content:
            return 0.0

        score = 0.0

        # Length score (moderate length gets higher score)
        length = len(content)
        if 50 <= length <= 1000:
            score += 0.3
        elif 1000 < length <= 2000:
            score += 0.2
        elif length > 2000:
            score += 0.1

        # Sentence completeness score
        sentences = content.split(".")
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        if complete_sentences >= 2:
            score += 0.3

        # Information density score (keyword density)
        keywords = self.extract_keywords(content, 5)
        if len(keywords) >= 3:
            score += 0.2

        # Readability score (based on punctuation and structure)
        if "." in content and ("," in content or ";" in content):
            score += 0.2

        return min(score, 1.0)


class SearchResultFilter:
    """Search result filter for content processing and formatting"""

    # Smart filtering threshold configuration
    SMART_FILTERING_THRESHOLD_RATIO = 0.6  # 60% threshold
    DEFAULT_BATCH_SIZE = 3
    MAX_SECONDARY_RESULTS = 5

    # Cleaning configuration options
    DEFAULT_CLEANING_CONFIG = {
        "min_quality_score": 0.3,  # Minimum quality score
        "max_results": 20,  # Maximum number of results
        "enable_keyword_extraction": True,  # Enable keyword extraction
        "enable_enhanced_key_points": True,  # Enable enhanced key point extraction
        "filter_invalid_urls": True,  # Filter invalid URLs
        "sort_by_quality": True,  # Sort by quality
    }

    def __init__(
        self,
        content_processor: ContentProcessor,
        cleaning_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize search result filter

        Args:
            content_processor: Content processor instance
            cleaning_config: Cleaning configuration options
        """
        self.content_processor = content_processor
        self.cleaner = SearchResultCleaner()  # Initialize cleaner
        self.cleaning_config = {
            **self.DEFAULT_CLEANING_CONFIG,
            **(cleaning_config or {}),
        }
        # Add new configuration options
        self.cleaning_config.setdefault(
            "filter_raw_content", True
        )  # Whether to filter raw_content
        self.cleaning_config.setdefault(
            "prefer_content_over_raw", True
        )  # Prefer content over raw_content

    def should_enable_smart_filtering(
        self, search_results: List[Dict[str, Any]], model_name: str
    ) -> bool:
        """Determine whether smart filtering should be enabled

        Args:
            search_results: List of search results
            model_name: Model name

        Returns:
            bool: Whether smart filtering should be enabled
        """
        if not search_results:
            return False

        # Always return True to enable smart filtering when there are search results
        # The actual enable/disable control is handled by the configuration
        return True

    def get_smart_filtering_threshold(self, model_name: str) -> int:
        """Get smart filtering threshold

        Args:
            model_name: Model name

        Returns:
            int: Smart filtering threshold (token count)
        """
        limits = self.content_processor.get_model_limits(model_name)
        return int(limits.input_limit * self.SMART_FILTERING_THRESHOLD_RATIO)

    def filter_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        model_name: str,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process and format search results without LLM filtering

        Args:
            query: User query
            search_results: Original search results
            model_name: Model name
            max_results: Maximum number of results

        Returns:
            Dictionary of processed results
        """
        if not search_results:
            return self._create_empty_result()

        # Limit search results count
        if max_results:
            search_results = search_results[:max_results]

        logger.info(f"Processing {len(search_results)} search results")

        # Process and clean search results
        processed_results = self._process_search_results(search_results)

        return {
            "filtered_results": processed_results,
            "summary": f"Processed {len(processed_results)} search results",
            "total_filtered": len(processed_results),
            "total_original": len(search_results),
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            "filtered_results": [],
            "summary": "No relevant search results found",
            "total_filtered": 0,
            "total_original": 0,
        }

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results as text"""
        formatted = []
        for i, result in enumerate(search_results, 1):
            title = self.content_processor.sanitize_content(
                str(result.get("title", "No title"))
            )
            content = self.content_processor.sanitize_content(
                str(result.get("content", result.get("snippet", "No content")))
            )
            url = self.content_processor.sanitize_content(
                str(result.get("url", "No link"))
            )

            formatted_result = f"""Result {i}:
Title: {title}
Link: {url}
Content: {content}
---"""
            formatted.append(formatted_result)

        return "\n\n".join(formatted)

    def _process_search_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process and clean search results with advanced cleaning"""
        processed_results = []

        for i, result in enumerate(search_results):
            # Get original content
            raw_title = str(result.get("title", f"Result {i+1}"))
            raw_content = str(result.get("content", result.get("snippet", "")))
            raw_url = str(result.get("url", ""))

            # Use new cleaner for deep cleaning
            cleaned_title = self.cleaner.clean_text(raw_title)
            cleaned_content = self.cleaner.clean_text(raw_content)

            # Validate URL validity
            if not self.cleaner.is_valid_url(raw_url):
                continue  # Skip results with invalid URLs

            # Calculate content quality score
            quality_score = self.cleaner.calculate_content_quality(cleaned_content)

            # Filter low-quality content
            if quality_score < 0.3 or not cleaned_content:
                continue

            # Extract keywords and key points
            keywords = self.cleaner.extract_keywords(cleaned_content, 8)
            key_points = self._extract_enhanced_key_points(cleaned_content, keywords)

            # Final cleaning (use original sanitize_content as supplement)
            title = self.content_processor.sanitize_content(cleaned_title)
            content = self.content_processor.sanitize_content(cleaned_content)
            url = self.content_processor.sanitize_content(raw_url)

            # Create processed result
            processed_result = {
                "title": title,
                "content": content,
                "url": url,
                "relevance_score": quality_score,
                "key_points": key_points,
                "keywords": keywords[:5],  # Keep top 5 keywords
                "content_length": len(content),
                "quality_score": quality_score,
            }

            processed_results.append(processed_result)

        # Sort by quality score
        processed_results.sort(key=lambda x: x["quality_score"], reverse=True)

        return processed_results

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content using simple text processing"""
        if not content:
            return []

        # Simple key point extraction - split by sentences and take first few
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        # Return first 3 sentences as key points, limited to reasonable length
        key_points = []
        for sentence in sentences[:3]:
            if (
                len(sentence) > 10 and len(sentence) < 200
            ):  # Filter reasonable length sentences
                key_points.append(sentence)

        return key_points[:3]  # Maximum 3 key points

    def _extract_enhanced_key_points(
        self, content: str, keywords: List[str]
    ) -> List[str]:
        """Enhanced key point extraction using keywords and sentence scoring"""
        if not content:
            return []

        # Split sentences
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        if not sentences:
            return []

        # Calculate score for each sentence
        sentence_scores = []
        for sentence in sentences:
            if len(sentence) < 15 or len(sentence) > 300:
                continue

            score = 0
            sentence_lower = sentence.lower()

            # Keyword matching score
            keyword_matches = sum(
                1 for keyword in keywords if keyword in sentence_lower
            )
            score += keyword_matches * 2

            # Sentence position score (earlier sentences are more important)
            position_score = max(0, 3 - sentences.index(sentence) * 0.5)
            score += position_score

            # Sentence length score (moderate length is better)
            length = len(sentence)
            if 50 <= length <= 150:
                score += 2
            elif 150 < length <= 250:
                score += 1

            # Information density score (contains numbers, technical terms, etc.)
            if re.search(r"\d+", sentence):  # Contains numbers
                score += 0.5
            if re.search(r"[A-Z]{2,}", sentence):  # Contains abbreviations
                score += 0.5

            sentence_scores.append((sentence, score))

        # Sort by score and return top 3
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        key_points = [sentence for sentence, score in sentence_scores[:3] if score > 0]

        return key_points

    def batch_clean_search_results(
        self, search_results_list: List[List[Dict[str, Any]]], model_name: str
    ) -> List[Dict[str, Any]]:
        """Batch clean multiple search result sets

        Args:
            search_results_list: List of multiple search result sets
            model_name: Model name

        Returns:
            List[Dict[str, Any]]: List of batch processed results
        """
        batch_results = []

        for i, search_results in enumerate(search_results_list):
            logger.debug(f"Processing batch {i+1}/{len(search_results_list)}")

            # Apply configuration limits
            max_results = self.cleaning_config.get("max_results", 20)
            limited_results = search_results[:max_results]

            # Process single result set
            processed_batch = self._process_search_results_with_config(limited_results)

            batch_results.append(
                {
                    "batch_id": i,
                    "original_count": len(search_results),
                    "processed_count": len(processed_batch),
                    "results": processed_batch,
                }
            )

        return batch_results

    def _process_search_results_with_config(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process search results using configuration options"""
        processed_results = []

        for i, result in enumerate(search_results):
            # Get original content, prefer content over raw_content
            raw_title = str(result.get("title", f"Result {i+1}"))
            # Prefer content field, avoid useless information in raw_content
            raw_content = str(result.get("content", result.get("snippet", "")))
            raw_url = str(result.get("url", ""))

            # If raw_content exists but no content, extract useful information from raw_content
            if not raw_content and result.get("raw_content"):
                raw_content = self._extract_useful_content_from_raw(
                    result.get("raw_content")
                )

            # Use cleaner for deep cleaning
            cleaned_title = self.cleaner.clean_text(raw_title)
            cleaned_content = self.cleaner.clean_text(raw_content)

            # Validate URL validity based on configuration
            if self.cleaning_config.get("filter_invalid_urls", True):
                if not self.cleaner.is_valid_url(raw_url):
                    continue

            # Calculate content quality score
            quality_score = self.cleaner.calculate_content_quality(cleaned_content)

            # Filter low-quality content based on configuration
            min_quality = self.cleaning_config.get("min_quality_score", 0.3)
            if quality_score < min_quality or not cleaned_content:
                continue

            # Extract keywords based on configuration
            keywords = []
            if self.cleaning_config.get("enable_keyword_extraction", True):
                keywords = self.cleaner.extract_keywords(cleaned_content, 8)

            # Extract key points based on configuration
            key_points = []
            if self.cleaning_config.get("enable_enhanced_key_points", True):
                key_points = self._extract_enhanced_key_points(
                    cleaned_content, keywords
                )
            else:
                key_points = self._extract_key_points(cleaned_content)

            # Final cleaning
            title = self.content_processor.sanitize_content(cleaned_title)
            content = self.content_processor.sanitize_content(cleaned_content)
            url = self.content_processor.sanitize_content(raw_url)

            # Create processed result without raw_content
            processed_result = {
                "title": title,
                "content": content,
                "url": url,
                "relevance_score": quality_score,
                "key_points": key_points,
                "keywords": keywords[:5],
                "content_length": len(content),
                "quality_score": quality_score,
            }

            processed_results.append(processed_result)

        # Sort by quality score based on configuration
        if self.cleaning_config.get("sort_by_quality", True):
            processed_results.sort(key=lambda x: x["quality_score"], reverse=True)

        return processed_results

    def format_filtered_results(self, filtered_data: Dict[str, Any]) -> str:
        """Format filtered results as text"""
        if not filtered_data.get("filtered_results"):
            return filtered_data.get("summary", "No relevant information found")

        formatted_parts = []

        # Add summary
        if filtered_data.get("summary"):
            formatted_parts.append(
                f"## Search Results Summary\n\n{filtered_data['summary']}\n"
            )

        # Add filtered results
        formatted_parts.append("## Relevant Information\n")

        for i, result in enumerate(filtered_data["filtered_results"], 1):
            title = result.get("title", f"Result {i}")
            content = result.get("content", "")
            url = result.get("url", "")
            relevance = result.get("relevance_score", "N/A")

            result_text = f"### {i}. {title}\n\n"
            if content:
                result_text += f"{content}\n\n"
            if url and url != "No link":
                result_text += f"**Source**: {url}\n\n"
            if relevance != "N/A":
                result_text += f"**Relevance**: {relevance}\n\n"

            formatted_parts.append(result_text)

        # Add statistics
        total_filtered = filtered_data.get("total_filtered", 0)
        total_original = filtered_data.get("total_original", 0)
        formatted_parts.append(
            f"---\n\n*Filtering Statistics: {total_filtered} relevant results filtered from {total_original} total results*"
        )

        return "\n".join(formatted_parts)

    def _extract_useful_content_from_raw(self, raw_content: str) -> str:
        """Extract useful information from raw content, filtering out ads, external links and other useless content

        Args:
            raw_content: Raw content string

        Returns:
            str: Extracted useful content
        """
        if not raw_content:
            return ""

        # Remove HTML tags
        import re

        text = re.sub(r"<[^>]+>", " ", raw_content)

        # Remove common ad and useless information patterns
        ad_patterns = [
            r"广告[^\n]*",  # Advertisement related
            r"Advertisement[^\n]*",
            r"Sponsored[^\n]*",
            r"推广[^\n]*",
            r"免责声明[^\n]*",
            r"Disclaimer[^\n]*",
            r"版权所有[^\n]*",
            r"Copyright[^\n]*",
            r"All rights reserved[^\n]*",
            r"更多.*?请访问[^\n]*",
            r"点击.*?了解更多[^\n]*",
            r"立即.*?咨询[^\n]*",
            r"联系我们[^\n]*",
            r"Contact us[^\n]*",
            r"订阅[^\n]*",  # Subscribe patterns
            r"Subscribe[^\n]*",
            r"关注我们[^\n]*",  # Follow us patterns
            r"Follow us[^\n]*",
            r"分享到[^\n]*",  # Share to patterns
            r"Share[^\n]*",
            r"相关推荐[^\n]*",  # Related recommendations patterns
            r"Related[^\n]*",
            r"热门.*?推荐[^\n]*",  # Popular recommendations patterns
            r"Popular[^\n]*",
            r"猜你喜欢[^\n]*",  # You might like patterns
            r"You might like[^\n]*",
        ]

        for pattern in ad_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove external link patterns
        link_patterns = [
            r"http[s]?://[^\s]+",  # HTTP links
            r"www\.[^\s]+",  # www links
            r"\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",  # Domain names
            r"点击.*?链接[^\n]*",  # Click link patterns
            r"Click.*?link[^\n]*",  # Click link patterns
            r"访问.*?网站[^\n]*",  # Visit website patterns
            r"Visit.*?website[^\n]*",
        ]

        for pattern in link_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove image-related information
        image_patterns = [
            r"\[图片\]",  # Image placeholders
            r"\[Image\]",  # Image placeholders
            r"\[img\]",
            r"图片来源[^\n]*",  # Image source patterns
            r"Image source[^\n]*",
            r"图：[^\n]*",  # Figure patterns
            r"Figure[^\n]*",
        ]

        for pattern in image_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove excessive whitespace characters
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)

        # Remove leading and trailing whitespace
        text = text.strip()

        # If processed text is too short, return empty string
        if len(text) < 20:
            return ""

        return text
