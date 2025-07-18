"""Enhanced Message Extractor for Robust LLM Context Management

Provides advanced message extraction capabilities to handle various complex message formats
and prevent token management failures due to message extraction issues.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import re
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from ..common.structured_logging import get_logger

logger = get_logger(__name__)


class MessagePattern(Enum):
    """Enumeration of supported message patterns"""

    DIRECT_KWARGS = "direct_kwargs"  # kwargs['messages']
    NESTED_INPUT = "nested_input"  # kwargs['input']['messages']
    ARGS_LIST = "args_list"  # args[i] as message list
    DICT_FORMAT = "dict_format"  # [{'role': 'user', 'content': '...'}]
    STRING_LIST = "string_list"  # ['message1', 'message2']
    SINGLE_MESSAGE = "single_message"  # Single message object
    CUSTOM_OBJECT = "custom_object"  # Custom message objects
    DEEP_NESTED = "deep_nested"  # Deeply nested structures


class ExtractionStrategy(Enum):
    """Extraction strategies in order of preference"""

    FAST = "fast"  # Quick extraction for common patterns
    COMPREHENSIVE = "comprehensive"  # Thorough search including recursion
    FALLBACK = "fallback"  # Last resort methods


@dataclass
class ExtractedMessages:
    """Container for extracted message information"""

    messages: Optional[List[BaseMessage]]
    pattern: Optional[MessagePattern]
    source_location: str
    extraction_time: float
    validation_passed: bool
    normalized: bool = False
    original_format: Optional[str] = None


@dataclass
class ModelInfo:
    """Container for extracted model information"""

    model_name: str
    source: str
    llm_instance: Optional[Any] = None


class EnhancedMessageExtractor:
    """Enhanced message extractor with comprehensive format support"""

    def __init__(self, max_search_depth: int = 8, enable_caching: bool = True):
        self.max_search_depth = max_search_depth
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
        self._extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "cache_hits": 0,
            "pattern_usage": {pattern: 0 for pattern in MessagePattern},
        }

    def extract_messages_and_model(
        self, args: tuple, kwargs: dict
    ) -> Tuple[ExtractedMessages, ModelInfo]:
        """Extract both messages and model information from function arguments

        Args:
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Tuple of (ExtractedMessages, ModelInfo)
        """
        start_time = time.time()
        self._extraction_stats["total_extractions"] += 1

        # Generate cache key if caching is enabled
        cache_key = None
        if self.enable_caching:
            cache_key = self._generate_cache_key(args, kwargs)
            if cache_key in self._cache:
                self._extraction_stats["cache_hits"] += 1
                logger.debug("Using cached extraction result")
                return self._cache[cache_key]

        try:
            # Extract messages using comprehensive strategy
            extracted_messages = self._extract_messages_comprehensive(
                args, kwargs, start_time
            )

            # Extract model information
            model_info = self._extract_model_info(args, kwargs)

            # Validate and normalize if extraction was successful
            if extracted_messages.messages:
                extracted_messages.validation_passed = self._validate_messages(
                    extracted_messages.messages
                )
                if extracted_messages.validation_passed:
                    extracted_messages.messages = self._normalize_messages(
                        extracted_messages.messages
                    )
                    extracted_messages.normalized = True
                    self._extraction_stats["successful_extractions"] += 1

            result = (extracted_messages, model_info)

            # Cache the result if caching is enabled
            if self.enable_caching and cache_key:
                self._cache[cache_key] = result

            # Log extraction statistics
            if extracted_messages.messages:
                logger.debug(
                    f"Message extraction successful: {len(extracted_messages.messages)} messages, "
                    f"pattern: {extracted_messages.pattern.value}, "
                    f"time: {extracted_messages.extraction_time:.3f}s"
                )
            else:
                logger.warning("Message extraction failed - no valid messages found")

            return result

        except Exception as e:
            logger.error(f"Message extraction failed with error: {e}")
            # Return empty result on failure
            empty_messages = ExtractedMessages(
                messages=None,
                pattern=None,
                source_location="extraction_failed",
                extraction_time=time.time() - start_time,
                validation_passed=False,
            )
            default_model = ModelInfo(
                model_name="deepseek-chat", source="default_fallback"
            )
            return empty_messages, default_model

    def _extract_messages_comprehensive(
        self, args: tuple, kwargs: dict, start_time: float
    ) -> ExtractedMessages:
        """Comprehensive message extraction using multiple strategies"""

        # Strategy 1: Fast extraction for common patterns
        result = self._try_fast_extraction(args, kwargs, start_time)
        if result.messages:
            return result

        # Strategy 2: Comprehensive search including recursion
        result = self._try_comprehensive_extraction(args, kwargs, start_time)
        if result.messages:
            return result

        # Strategy 3: Fallback methods
        result = self._try_fallback_extraction(args, kwargs, start_time)
        return result

    def _try_fast_extraction(
        self, args: tuple, kwargs: dict, start_time: float
    ) -> ExtractedMessages:
        """Fast extraction for common message patterns"""

        # Pattern 1: Direct kwargs['messages']
        if "messages" in kwargs:
            messages = kwargs["messages"]
            if self._is_valid_message_list(messages):
                self._extraction_stats["pattern_usage"][
                    MessagePattern.DIRECT_KWARGS
                ] += 1
                return ExtractedMessages(
                    messages=messages,
                    pattern=MessagePattern.DIRECT_KWARGS,
                    source_location="kwargs['messages']",
                    extraction_time=time.time() - start_time,
                    validation_passed=False,  # Will be validated later
                )

        # Pattern 2: Nested input kwargs['input']['messages']
        if (
            "input" in kwargs
            and isinstance(kwargs["input"], dict)
            and "messages" in kwargs["input"]
        ):
            messages = kwargs["input"]["messages"]
            if self._is_valid_message_list(messages):
                self._extraction_stats["pattern_usage"][
                    MessagePattern.NESTED_INPUT
                ] += 1
                return ExtractedMessages(
                    messages=messages,
                    pattern=MessagePattern.NESTED_INPUT,
                    source_location="kwargs['input']['messages']",
                    extraction_time=time.time() - start_time,
                    validation_passed=False,
                )

        # Pattern 3: Messages in args (first valid list found)
        for i, arg in enumerate(args):
            if self._is_valid_message_list(arg):
                self._extraction_stats["pattern_usage"][MessagePattern.ARGS_LIST] += 1
                return ExtractedMessages(
                    messages=arg,
                    pattern=MessagePattern.ARGS_LIST,
                    source_location=f"args[{i}]",
                    extraction_time=time.time() - start_time,
                    validation_passed=False,
                )

        # No messages found in fast extraction
        return ExtractedMessages(
            messages=None,
            pattern=None,
            source_location="fast_extraction_failed",
            extraction_time=time.time() - start_time,
            validation_passed=False,
        )

    def _try_comprehensive_extraction(
        self, args: tuple, kwargs: dict, start_time: float
    ) -> ExtractedMessages:
        """Comprehensive extraction including recursive search"""

        # Try to extract from dict format messages first (more specific)
        result = self._try_dict_format_extraction(args, kwargs, start_time)
        if result.messages:
            return result

        # Try to extract from string format (more specific)
        result = self._try_string_format_extraction(args, kwargs, start_time)
        if result.messages:
            return result

        # Search in kwargs recursively (more general)
        result = self._recursive_search_kwargs(kwargs, start_time, 0, "")
        if result.messages:
            return result

        # Search in args recursively (most general)
        result = self._recursive_search_args(args, start_time)
        if result.messages:
            return result

        return ExtractedMessages(
            messages=None,
            pattern=None,
            source_location="comprehensive_extraction_failed",
            extraction_time=time.time() - start_time,
            validation_passed=False,
        )

    def _try_fallback_extraction(
        self, args: tuple, kwargs: dict, start_time: float
    ) -> ExtractedMessages:
        """Fallback extraction methods for edge cases"""

        # Try to find any object with 'content' attribute
        for i, arg in enumerate(args):
            if hasattr(arg, "content"):
                # Single message object
                self._extraction_stats["pattern_usage"][
                    MessagePattern.SINGLE_MESSAGE
                ] += 1
                return ExtractedMessages(
                    messages=[arg],
                    pattern=MessagePattern.SINGLE_MESSAGE,
                    source_location=f"args[{i}] (single message)",
                    extraction_time=time.time() - start_time,
                    validation_passed=False,
                )

        # Check kwargs for single message objects
        for key, value in kwargs.items():
            if hasattr(value, "content"):
                self._extraction_stats["pattern_usage"][
                    MessagePattern.SINGLE_MESSAGE
                ] += 1
                return ExtractedMessages(
                    messages=[value],
                    pattern=MessagePattern.SINGLE_MESSAGE,
                    source_location=f"kwargs['{key}'] (single message)",
                    extraction_time=time.time() - start_time,
                    validation_passed=False,
                )

        # Final fallback - return empty result
        return ExtractedMessages(
            messages=None,
            pattern=None,
            source_location="all_extraction_methods_failed",
            extraction_time=time.time() - start_time,
            validation_passed=False,
        )

    def _recursive_search_kwargs(
        self, kwargs: dict, start_time: float, depth: int = 0, path_prefix: str = ""
    ) -> ExtractedMessages:
        """Recursively search kwargs for messages"""
        if depth >= self.max_search_depth:
            return ExtractedMessages(
                messages=None,
                pattern=None,
                source_location="max_depth_reached",
                extraction_time=time.time() - start_time,
                validation_passed=False,
            )

        for key, value in kwargs.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            # Check if value is a message list (including mixed formats)
            if self._is_valid_message_list(value):
                # Validate and normalize the messages
                if self._validate_messages(value):
                    normalized_messages = self._normalize_messages(value)
                    self._extraction_stats["pattern_usage"][
                        MessagePattern.DEEP_NESTED
                    ] += 1
                    return ExtractedMessages(
                        messages=normalized_messages,
                        pattern=MessagePattern.DEEP_NESTED,
                        source_location=current_path,
                        extraction_time=time.time() - start_time,
                        validation_passed=True,
                        normalized=True,
                    )
            elif self._is_mixed_message_list(value):
                # Handle mixed format messages
                normalized_messages = self._normalize_messages(value)
                if normalized_messages and self._validate_messages(normalized_messages):
                    self._extraction_stats["pattern_usage"][
                        MessagePattern.DEEP_NESTED
                    ] += 1
                    return ExtractedMessages(
                        messages=normalized_messages,
                        pattern=MessagePattern.DEEP_NESTED,
                        source_location=current_path,
                        extraction_time=time.time() - start_time,
                        validation_passed=True,
                        normalized=True,
                    )

            # Recurse into dictionaries
            elif isinstance(value, dict):
                result = self._recursive_search_kwargs(
                    value, start_time, depth + 1, current_path
                )
                if result.messages:
                    return result

            # Check lists for nested structures
            elif isinstance(value, (list, tuple)) and not self._is_valid_message_list(
                value
            ):
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    if isinstance(item, dict):
                        result = self._recursive_search_kwargs(
                            item, start_time, depth + 1, item_path
                        )
                        if result.messages:
                            return result
                    elif self._is_valid_message_list(
                        item
                    ) or self._is_mixed_message_list(item):
                        normalized_messages = self._normalize_messages(item)
                        if normalized_messages and self._validate_messages(
                            normalized_messages
                        ):
                            self._extraction_stats["pattern_usage"][
                                MessagePattern.DEEP_NESTED
                            ] += 1
                            return ExtractedMessages(
                                messages=normalized_messages,
                                pattern=MessagePattern.DEEP_NESTED,
                                source_location=item_path,
                                extraction_time=time.time() - start_time,
                                validation_passed=True,
                                normalized=True,
                            )

        return ExtractedMessages(
            messages=None,
            pattern=None,
            source_location="recursive_kwargs_failed",
            extraction_time=time.time() - start_time,
            validation_passed=False,
        )

    def _recursive_search_args(
        self, args: tuple, start_time: float, depth: int = 0
    ) -> ExtractedMessages:
        """Recursively search args for messages"""
        if depth >= self.max_search_depth:
            return ExtractedMessages(
                messages=None,
                pattern=None,
                source_location="max_depth_reached",
                extraction_time=time.time() - start_time,
                validation_passed=False,
            )

        for i, arg in enumerate(args):
            # Skip if already checked in fast extraction
            if depth == 0 and self._is_valid_message_list(arg):
                continue

            # Recurse into dictionaries
            if isinstance(arg, dict):
                result = self._recursive_search_kwargs(
                    arg, start_time, depth + 1, f"args[{i}]"
                )
                if result.messages:
                    return result

            # Check nested lists/tuples
            elif isinstance(arg, (list, tuple)) and not self._is_valid_message_list(
                arg
            ):
                for j, item in enumerate(arg):
                    if self._is_valid_message_list(item):
                        self._extraction_stats["pattern_usage"][
                            MessagePattern.DEEP_NESTED
                        ] += 1
                        return ExtractedMessages(
                            messages=item,
                            pattern=MessagePattern.DEEP_NESTED,
                            source_location=f"args[{i}][{j}] (depth {depth})",
                            extraction_time=time.time() - start_time,
                            validation_passed=False,
                        )

        return ExtractedMessages(
            messages=None,
            pattern=None,
            source_location="recursive_args_failed",
            extraction_time=time.time() - start_time,
            validation_passed=False,
        )

    def _try_dict_format_extraction(
        self, args: tuple, kwargs: dict, start_time: float
    ) -> ExtractedMessages:
        """Try to extract messages from dictionary format"""

        def convert_dict_to_message(msg_dict):
            """Convert dictionary format to BaseMessage"""
            if not isinstance(msg_dict, dict):
                return None

            role = msg_dict.get("role", "").lower()
            content = msg_dict.get("content", "")

            if not content:
                return None

            if role == "system":
                return SystemMessage(content=content)
            elif role == "assistant" or role == "ai":
                return AIMessage(content=content)
            else:  # Default to human message
                return HumanMessage(content=content)

        # Check all possible locations for dict format messages
        all_locations = [
            ("kwargs.messages", kwargs.get("messages")),
            (
                "kwargs.input.messages",
                (
                    kwargs.get("input", {}).get("messages")
                    if isinstance(kwargs.get("input"), dict)
                    else None
                ),
            ),
        ]

        # Add args locations
        for i, arg in enumerate(args):
            all_locations.append((f"args[{i}]", arg))

        for location, potential_messages in all_locations:
            if not isinstance(potential_messages, (list, tuple)):
                continue

            converted_messages = []
            all_converted = True

            for msg in potential_messages:
                converted = convert_dict_to_message(msg)
                if converted:
                    converted_messages.append(converted)
                else:
                    all_converted = False
                    break

            if all_converted and converted_messages:
                # Validate and normalize the converted messages
                if self._validate_messages(converted_messages):
                    normalized_messages = self._normalize_messages(converted_messages)
                    self._extraction_stats["pattern_usage"][
                        MessagePattern.DICT_FORMAT
                    ] += 1
                    return ExtractedMessages(
                        messages=normalized_messages,
                        pattern=MessagePattern.DICT_FORMAT,
                        source_location=location,
                        extraction_time=time.time() - start_time,
                        validation_passed=True,
                        normalized=True,
                        original_format="dict_format",
                    )

        return ExtractedMessages(
            messages=None,
            pattern=None,
            source_location="dict_format_failed",
            extraction_time=time.time() - start_time,
            validation_passed=False,
        )

    def _try_string_format_extraction(
        self, args: tuple, kwargs: dict, start_time: float
    ) -> ExtractedMessages:
        """Try to extract messages from string format"""

        # Check for string lists that could be converted to messages
        all_locations = [
            ("kwargs.messages", kwargs.get("messages")),
            (
                "kwargs.input.messages",
                (
                    kwargs.get("input", {}).get("messages")
                    if isinstance(kwargs.get("input"), dict)
                    else None
                ),
            ),
        ]

        for i, arg in enumerate(args):
            all_locations.append((f"args[{i}]", arg))

        for location, potential_messages in all_locations:
            if (
                isinstance(potential_messages, (list, tuple))
                and potential_messages
                and all(isinstance(msg, str) for msg in potential_messages)
            ):

                # Convert strings to HumanMessage objects
                converted_messages = [
                    HumanMessage(content=msg)
                    for msg in potential_messages
                    if msg.strip()
                ]

                if converted_messages:
                    # Validate the converted messages
                    if self._validate_messages(converted_messages):
                        normalized_messages = self._normalize_messages(
                            converted_messages
                        )
                        self._extraction_stats["pattern_usage"][
                            MessagePattern.STRING_LIST
                        ] += 1
                        return ExtractedMessages(
                            messages=normalized_messages,
                            pattern=MessagePattern.STRING_LIST,
                            source_location=location,
                            extraction_time=time.time() - start_time,
                            validation_passed=True,
                            normalized=True,
                            original_format="string_list",
                        )

        return ExtractedMessages(
            messages=None,
            pattern=None,
            source_location="string_format_failed",
            extraction_time=time.time() - start_time,
            validation_passed=False,
        )

    def _extract_model_info(self, args: tuple, kwargs: dict) -> ModelInfo:
        """Extract model information from function arguments"""
        model_name = "deepseek-chat"  # Default fallback
        source = "default"
        llm_instance = None

        # Check for LLM instance in args
        for arg in args:
            if hasattr(arg, "model_name"):
                model_name = arg.model_name
                source = "llm_instance.model_name"
                llm_instance = arg
                break
            elif hasattr(arg, "model"):
                model_name = arg.model
                source = "llm_instance.model"
                llm_instance = arg
                break

        # Check kwargs for model information
        if model_name == "deepseek-chat":  # Still using default
            # Check direct kwargs.model
            if "model" in kwargs:
                model_name = kwargs["model"]
                source = "kwargs.model"
            # Check kwargs.config.model
            elif (
                "config" in kwargs
                and isinstance(kwargs["config"], dict)
                and "model" in kwargs["config"]
            ):
                model_name = kwargs["config"]["model"]
                source = "kwargs.config.model"
            # Check nested structures
            else:
                model_result = self._search_nested_dict(kwargs, "model")
                if model_result:
                    model_name = model_result
                    source = "kwargs.nested.model"

        return ModelInfo(
            model_name=model_name, source=source, llm_instance=llm_instance
        )

    def _search_nested_dict(self, data: dict, key: str, max_depth: int = 8) -> Any:
        """Search for a key in nested dictionary structure"""

        def _recursive_search(obj: Any, target_key: str, current_depth: int = 0) -> Any:
            if current_depth >= max_depth:
                return None

            if isinstance(obj, dict):
                if target_key in obj:
                    return obj[target_key]

                for value in obj.values():
                    result = _recursive_search(value, target_key, current_depth + 1)
                    if result is not None:
                        return result

            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if isinstance(item, dict):
                        result = _recursive_search(item, target_key, current_depth + 1)
                        if result is not None:
                            return result
                    elif isinstance(item, (list, tuple)):
                        result = _recursive_search(item, target_key, current_depth + 1)
                        if result is not None:
                            return result

            return None

        return _recursive_search(data, key)

    def _is_valid_message_list(self, obj: Any) -> bool:
        """Check if object is a valid message list"""
        if not isinstance(obj, (list, tuple)):
            return False

        if not obj:  # Empty list
            return False

        # Check if all items have content attribute (BaseMessage-like)
        try:
            return all(hasattr(item, "content") for item in obj)
        except (TypeError, AttributeError):
            return False

    def _is_mixed_message_list(self, obj: Any) -> bool:
        """Check if object is a mixed format message list (dict, string, BaseMessage)"""
        if not isinstance(obj, (list, tuple)):
            return False

        if not obj:  # Empty list
            return False

        try:
            # Check if items are potential messages (dict with role/content, strings, or BaseMessage)
            for item in obj:
                if hasattr(item, "content"):  # BaseMessage
                    continue
                elif isinstance(item, dict) and "content" in item:  # Dict format
                    continue
                elif isinstance(item, str) and item.strip():  # Non-empty string
                    continue
                else:
                    return False
            return True
        except (TypeError, AttributeError):
            return False

    def _validate_messages(self, messages: List[BaseMessage]) -> bool:
        """Validate extracted messages"""
        if not messages:
            return False

        try:
            for msg in messages:
                # Check if message has content
                if not hasattr(msg, "content"):
                    return False

                # Check if content is not empty
                content = getattr(msg, "content", "")
                if not content or not str(content).strip():
                    return False

            return True
        except Exception as e:
            logger.debug(f"Message validation failed: {e}")
            return False

    def _normalize_messages(self, messages: List[Any]) -> List[BaseMessage]:
        """Normalize messages to BaseMessage format with intelligent deduplication"""
        normalized = []

        for msg in messages:
            try:
                if isinstance(msg, BaseMessage):
                    normalized.append(msg)
                elif isinstance(msg, dict):
                    # Convert dict format to BaseMessage
                    role = msg.get("role", "user").lower()
                    content = msg.get("content", "")

                    if not content or not isinstance(content, str):
                        continue  # Skip invalid messages

                    if role == "system":
                        normalized.append(SystemMessage(content=content))
                    elif role in ["assistant", "ai"]:
                        normalized.append(AIMessage(content=content))
                    else:  # user, human, or default
                        normalized.append(HumanMessage(content=content))
                elif isinstance(msg, str):
                    # Convert string to HumanMessage
                    if msg.strip():  # Only add non-empty strings
                        normalized.append(HumanMessage(content=msg))
                elif hasattr(msg, "content"):
                    # Custom message object with content attribute
                    content = str(msg.content)
                    if content.strip():
                        normalized.append(HumanMessage(content=content))
                else:
                    # Fallback: convert to string and create HumanMessage
                    content = str(msg)
                    if content.strip():
                        normalized.append(HumanMessage(content=content))
            except Exception as e:
                logger.warning(f"Failed to normalize message: {e}")
                continue

        # Apply intelligent deduplication
        return self._deduplicate_messages(normalized)

    def _generate_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Generate cache key for extraction results"""
        # Simple cache key based on structure, not content
        # to avoid performance issues with large content
        key_parts = [
            f"args_len:{len(args)}",
            f"kwargs_keys:{sorted(kwargs.keys())}",
        ]

        # Add type information for args
        for i, arg in enumerate(args[:3]):  # Limit to first 3 args
            key_parts.append(f"arg{i}_type:{type(arg).__name__}")

        return "|".join(key_parts)

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        stats = self._extraction_stats.copy()
        if stats["total_extractions"] > 0:
            stats["success_rate"] = (
                stats["successful_extractions"] / stats["total_extractions"]
            )
            # Calculate average extraction time
            if "total_extraction_time" in stats and stats["total_extraction_time"] > 0:
                stats["avg_extraction_time"] = (
                    stats["total_extraction_time"] / stats["total_extractions"]
                )
            else:
                stats["avg_extraction_time"] = 0.0
        else:
            stats["success_rate"] = 0.0
            stats["avg_extraction_time"] = 0.0
        return stats

    def clear_cache(self):
        """Clear extraction cache"""
        if self._cache:
            self._cache.clear()
            logger.debug("Message extraction cache cleared")

    def _deduplicate_messages(
        self, messages: List[BaseMessage], similarity_threshold: float = 0.85
    ) -> List[BaseMessage]:
        """Remove duplicate messages using text similarity detection"""
        if not messages:
            return messages

        deduplicated = []
        seen_contents = []

        for msg in messages:
            content = msg.content.strip()
            if not content:
                continue

            # Check for exact duplicates first (fast path)
            if content in seen_contents:
                logger.debug(f"Skipping exact duplicate message: {content[:50]}...")
                continue

            # Check for similar content using text similarity
            is_similar = False
            for existing_content in seen_contents:
                similarity = self._calculate_text_similarity(content, existing_content)
                if similarity >= similarity_threshold:
                    logger.debug(
                        f"Skipping similar message (similarity: {similarity:.2f}): {content[:50]}..."
                    )
                    is_similar = True
                    break

            if not is_similar:
                deduplicated.append(msg)
                seen_contents.append(content)

        if len(deduplicated) < len(messages):
            logger.info(
                f"Deduplicated {len(messages) - len(deduplicated)} messages, kept {len(deduplicated)}"
            )

        return deduplicated

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        # Normalize texts
        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)

        # If either text is too short, use exact match
        if len(text1_norm) < 10 or len(text2_norm) < 10:
            return 1.0 if text1_norm == text2_norm else 0.0

        # Calculate multiple similarity metrics
        jaccard_sim = self._jaccard_similarity(text1_norm, text2_norm)
        cosine_sim = self._cosine_similarity(text1_norm, text2_norm)

        # Weighted combination of similarities
        combined_similarity = (jaccard_sim * 0.4) + (cosine_sim * 0.6)

        return combined_similarity

    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove common punctuation that doesn't affect meaning
        text = re.sub(r'[.,!?;:"\'\'\"\`]', "", text)

        return text

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on word sets"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity based on word frequency vectors"""
        words1 = text1.split()
        words2 = text2.split()

        if not words1 or not words2:
            return 1.0 if not words1 and not words2 else 0.0

        # Create word frequency vectors
        all_words = set(words1 + words2)
        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


# Global instance for reuse
_global_extractor = None


def get_global_message_extractor() -> EnhancedMessageExtractor:
    """Get or create global message extractor instance"""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = EnhancedMessageExtractor()
        logger.debug("Created global message extractor instance")
    return _global_extractor
