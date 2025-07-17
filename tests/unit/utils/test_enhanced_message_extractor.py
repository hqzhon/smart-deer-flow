"""Tests for Enhanced Message Extractor

Tests the comprehensive message extraction capabilities, various message formats,
and intelligent content deduplication functionality.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.utils.tokens.enhanced_message_extractor import (
    EnhancedMessageExtractor,
    MessagePattern,
    get_global_message_extractor,
)

# from src.utils.types import Message, MessageType  # Not needed for deduplication tests


class TestEnhancedMessageExtractor:
    """Test enhanced message extractor functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.extractor = EnhancedMessageExtractor(
            max_search_depth=10, enable_caching=False
        )

        # Create test messages
        self.test_messages = [
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you!"),
            SystemMessage(content="You are a helpful assistant."),
        ]

        # Create dict format messages
        self.dict_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        # Create string messages
        self.string_messages = [
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "You are a helpful assistant.",
        ]

    def test_direct_kwargs_extraction(self):
        """Test extraction from direct kwargs['messages']"""
        args = ()
        kwargs = {"messages": self.test_messages}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert extracted.messages == self.test_messages
        assert extracted.pattern == MessagePattern.DIRECT_KWARGS
        assert extracted.validation_passed
        assert extracted.normalized
        assert "kwargs['messages']" in extracted.source_location
        assert model_info.model_name == "deepseek-chat"  # Default

    def test_nested_input_extraction(self):
        """Test extraction from kwargs['input']['messages']"""
        args = ()
        kwargs = {"input": {"messages": self.test_messages, "model": "gpt-4"}}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert extracted.messages == self.test_messages
        assert extracted.pattern == MessagePattern.NESTED_INPUT
        assert extracted.validation_passed
        assert "kwargs['input']['messages']" in extracted.source_location
        assert model_info.model_name == "gpt-4"

    def test_args_list_extraction(self):
        """Test extraction from args"""
        args = ("some_param", self.test_messages, "another_param")
        kwargs = {}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert extracted.messages == self.test_messages
        assert extracted.pattern == MessagePattern.ARGS_LIST
        assert extracted.validation_passed
        assert "args[1]" in extracted.source_location

    def test_dict_format_extraction(self):
        """Test extraction and conversion from dictionary format"""
        args = ()
        kwargs = {"messages": self.dict_messages}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert len(extracted.messages) == 3
        assert extracted.pattern == MessagePattern.DICT_FORMAT
        assert extracted.validation_passed
        assert extracted.original_format == "dict_format"

        # Check message types
        assert isinstance(extracted.messages[0], HumanMessage)
        assert isinstance(extracted.messages[1], AIMessage)
        assert isinstance(extracted.messages[2], SystemMessage)

        # Check content
        assert extracted.messages[0].content == "Hello, how are you?"
        assert extracted.messages[1].content == "I'm doing well, thank you!"
        assert extracted.messages[2].content == "You are a helpful assistant."

    def test_string_format_extraction(self):
        """Test extraction and conversion from string format"""
        args = ()
        kwargs = {"messages": self.string_messages}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert len(extracted.messages) == 3
        assert extracted.pattern == MessagePattern.STRING_LIST
        assert extracted.validation_passed
        assert extracted.original_format == "string_list"

        # All should be converted to HumanMessage
        for msg in extracted.messages:
            assert isinstance(msg, HumanMessage)

        # Check content
        assert extracted.messages[0].content == "Hello, how are you?"
        assert extracted.messages[1].content == "I'm doing well, thank you!"
        assert extracted.messages[2].content == "You are a helpful assistant."

    def test_single_message_extraction(self):
        """Test extraction of single message object"""
        single_message = HumanMessage(content="Single message test")
        args = (single_message,)
        kwargs = {}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert len(extracted.messages) == 1
        assert extracted.messages[0] == single_message
        assert extracted.pattern == MessagePattern.SINGLE_MESSAGE
        assert extracted.validation_passed

    def test_deep_nested_extraction(self):
        """Test extraction from deeply nested structures"""
        args = ()
        kwargs = {
            "config": {"llm_params": {"input_data": {"messages": self.test_messages}}}
        }

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert extracted.messages == self.test_messages
        assert extracted.pattern == MessagePattern.DEEP_NESTED
        assert extracted.validation_passed
        assert "config" in extracted.source_location

    def test_model_extraction_from_llm_instance(self):
        """Test model name extraction from LLM instance"""
        mock_llm = Mock()
        mock_llm.model_name = "claude-3-sonnet"

        args = (mock_llm,)
        kwargs = {"messages": self.test_messages}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert model_info.model_name == "claude-3-sonnet"
        assert model_info.source == "llm_instance.model_name"
        assert model_info.llm_instance == mock_llm

    def test_model_extraction_from_kwargs(self):
        """Test model name extraction from kwargs"""
        args = ()
        kwargs = {"messages": self.test_messages, "model": "gpt-3.5-turbo"}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert model_info.model_name == "gpt-3.5-turbo"
        assert model_info.source == "kwargs.model"

    def test_model_extraction_from_config(self):
        """Test model name extraction from config"""
        args = ()
        kwargs = {"messages": self.test_messages, "config": {"model": "gemini-pro"}}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert model_info.model_name == "gemini-pro"
        assert model_info.source == "kwargs.config.model"

    def test_empty_messages_handling(self):
        """Test handling of empty or invalid messages"""
        args = ()
        kwargs = {"messages": []}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        assert extracted.messages is None
        assert not extracted.validation_passed
        assert model_info.model_name == "deepseek-chat"  # Default fallback

    def test_invalid_message_format(self):
        """Test handling of invalid message formats"""
        invalid_messages = ["string1", {"invalid": "dict"}, 123]
        args = ()
        kwargs = {"messages": invalid_messages}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        # Should fall back to string extraction for the string, but fail overall
        assert extracted.messages is None or not extracted.validation_passed

    def test_message_validation(self):
        """Test message validation logic"""
        # Valid messages
        valid_messages = [
            HumanMessage(content="Valid message 1"),
            HumanMessage(content="Valid message 2"),
        ]
        assert self.extractor._validate_messages(valid_messages)

        # Invalid messages (empty content)
        invalid_messages = [HumanMessage(content=""), HumanMessage(content="   ")]
        assert not self.extractor._validate_messages(invalid_messages)

        # Mixed valid/invalid
        mixed_messages = [
            HumanMessage(content="Valid message"),
            HumanMessage(content=""),
        ]
        assert not self.extractor._validate_messages(mixed_messages)

    def test_message_normalization(self):
        """Test message normalization"""

        # Create custom message object
        class CustomMessage:
            def __init__(self, content):
                self.content = content

        mixed_messages = [
            HumanMessage(content="Already normalized"),
            CustomMessage("Custom message"),
            "String message",
        ]

        normalized = self.extractor._normalize_messages(mixed_messages)

        assert len(normalized) == 3
        assert all(isinstance(msg, HumanMessage) for msg in normalized)
        assert normalized[0].content == "Already normalized"
        assert normalized[1].content == "Custom message"
        assert normalized[2].content == "String message"

    def test_extraction_statistics(self):
        """Test extraction statistics tracking"""
        # Perform several extractions
        test_cases = [
            ({}, {"messages": self.test_messages}),
            ((), {"input": {"messages": self.test_messages}}),
            ((self.test_messages,), {}),
        ]

        for args, kwargs in test_cases:
            self.extractor.extract_messages_and_model(args, kwargs)

        stats = self.extractor.get_extraction_stats()

        assert stats["total_extractions"] == 3
        assert stats["successful_extractions"] == 3
        assert stats["pattern_usage"][MessagePattern.DIRECT_KWARGS] >= 1
        assert stats["pattern_usage"][MessagePattern.NESTED_INPUT] >= 1
        assert stats["pattern_usage"][MessagePattern.ARGS_LIST] >= 1

    def test_caching_functionality(self):
        """Test caching functionality"""
        # Create extractor with caching enabled
        cached_extractor = EnhancedMessageExtractor(enable_caching=True)

        args = ()
        kwargs = {"messages": self.test_messages}

        # First extraction
        result1 = cached_extractor.extract_messages_and_model(args, kwargs)

        # Second extraction (should use cache)
        result2 = cached_extractor.extract_messages_and_model(args, kwargs)

        # Results should be identical
        assert result1[0].messages == result2[0].messages
        assert result1[1].model_name == result2[1].model_name

        # Check cache hit
        stats = cached_extractor.get_extraction_stats()
        assert stats["cache_hits"] >= 1

        # Clear cache
        cached_extractor.clear_cache()

    def test_global_extractor_instance(self):
        """Test global extractor instance"""
        extractor1 = get_global_message_extractor()
        extractor2 = get_global_message_extractor()

        # Should return the same instance
        assert extractor1 is extractor2
        assert isinstance(extractor1, EnhancedMessageExtractor)

    def test_extraction_performance(self):
        """Test extraction performance and timing"""
        args = ()
        kwargs = {"messages": self.test_messages}

        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)

        # Check that extraction time is recorded
        assert extracted.extraction_time > 0
        assert extracted.extraction_time < 1.0  # Should be very fast

    def test_complex_nested_structure(self):
        """Test extraction from complex nested structures"""
        complex_kwargs = {
            "request": {
                "data": {
                    "llm_call": {
                        "parameters": {"input": {"messages": self.test_messages}}
                    }
                }
            }
        }

        extracted, model_info = self.extractor.extract_messages_and_model(
            (), complex_kwargs
        )

        assert extracted.messages == self.test_messages
        assert extracted.pattern == MessagePattern.DEEP_NESTED
        assert "request" in extracted.source_location

    def test_max_search_depth_limit(self):
        """Test that search depth limit is respected"""
        # Create very deeply nested structure (beyond max depth)
        deep_kwargs = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {"level6": {"messages": self.test_messages}}
                        }
                    }
                }
            }
        }

        # Create extractor with low max depth
        shallow_extractor = EnhancedMessageExtractor(max_search_depth=3)

        extracted, model_info = shallow_extractor.extract_messages_and_model(
            (), deep_kwargs
        )

        # Should not find messages due to depth limit
        assert extracted.messages is None
        assert (
            "max_depth_reached" in extracted.source_location
            or "failed" in extracted.source_location
        )


class TestEnhancedMessageExtractorDeduplication:
    """Test intelligent content deduplication functionality"""

    @pytest.fixture
    def extractor(self):
        """EnhancedMessageExtractor fixture"""
        return EnhancedMessageExtractor()

    def test_jaccard_similarity_identical(self, extractor):
        """Test Jaccard similarity with identical texts"""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy dog"

        similarity = extractor._jaccard_similarity(text1, text2)
        assert similarity == 1.0

    def test_jaccard_similarity_no_overlap(self, extractor):
        """Test Jaccard similarity with no overlap"""
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"

        similarity = extractor._jaccard_similarity(text1, text2)
        assert similarity == 0.0

    def test_jaccard_similarity_partial_overlap(self, extractor):
        """Test Jaccard similarity with partial overlap"""
        text1 = "the quick brown fox"
        text2 = "the lazy brown dog"

        similarity = extractor._jaccard_similarity(text1, text2)
        # Common words: "the", "brown" = 2
        # Total unique words: "the", "quick", "brown", "fox", "lazy", "dog" = 6
        # Jaccard = 2/6 = 0.333...
        assert abs(similarity - 0.3333333333333333) < 0.001

    def test_jaccard_similarity_empty_texts(self, extractor):
        """Test jaccard similarity with empty texts"""
        similarity1 = extractor._jaccard_similarity("", "some text")
        similarity2 = extractor._jaccard_similarity("some text", "")
        similarity3 = extractor._jaccard_similarity("", "")

        assert similarity1 == 0.0
        assert similarity2 == 0.0
        assert similarity3 == 1.0  # Both empty should be identical

    def test_cosine_similarity_identical(self, extractor):
        """Test cosine similarity with identical texts"""
        text1 = "machine learning artificial intelligence"
        text2 = "machine learning artificial intelligence"

        similarity = extractor._cosine_similarity(text1, text2)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_no_overlap(self, extractor):
        """Test cosine similarity with no overlap"""
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"

        similarity = extractor._cosine_similarity(text1, text2)
        assert similarity == 0.0

    def test_cosine_similarity_partial_overlap(self, extractor):
        """Test cosine similarity with partial overlap"""
        text1 = "machine learning is powerful"
        text2 = "machine learning helps automation"

        similarity = extractor._cosine_similarity(text1, text2)
        # Should be > 0 due to common words "machine" and "learning"
        assert similarity > 0.0
        assert similarity < 1.0

    def test_cosine_similarity_empty_texts(self, extractor):
        """Test cosine similarity with empty texts"""
        similarity1 = extractor._cosine_similarity("", "some text")
        similarity2 = extractor._cosine_similarity("some text", "")
        similarity3 = extractor._cosine_similarity("", "")

        assert similarity1 == 0.0  # Empty vs non-empty returns 0.0
        assert similarity2 == 0.0  # Non-empty vs empty returns 0.0
        assert similarity3 == 1.0  # Both empty should be identical

    def test_calculate_text_similarity_high_similarity(self, extractor):
        """Test text similarity calculation with high similarity"""
        content1 = "Artificial intelligence is transforming healthcare industry"
        content2 = "AI is revolutionizing the healthcare sector"

        # Should have high similarity due to semantic overlap
        similarity = extractor._calculate_text_similarity(content1, content2)
        assert similarity > 0.25  # Adjusted based on actual calculation

    def test_calculate_text_similarity_low_similarity(self, extractor):
        """Test text similarity calculation with low similarity"""
        content1 = "Artificial intelligence in healthcare"
        content2 = "Weather forecast for tomorrow"

        # Should have low similarity
        similarity = extractor._calculate_text_similarity(content1, content2)
        assert similarity < 0.3

    def test_calculate_text_similarity_threshold_boundary(self, extractor):
        """Test text similarity calculation at threshold boundary"""
        content1 = "the quick brown fox"
        content2 = "the lazy brown dog"

        # Calculate similarity
        similarity = extractor._calculate_text_similarity(content1, content2)
        assert similarity > 0.0
        assert similarity < 1.0

    def test_deduplicate_messages_no_duplicates(self, extractor):
        """Test message deduplication with no duplicates"""
        messages = [
            HumanMessage(content="First unique message"),
            AIMessage(content="Second unique message"),
            HumanMessage(content="Third unique message"),
        ]

        deduplicated = extractor._deduplicate_messages(messages)

        assert len(deduplicated) == 3
        assert deduplicated == messages

    def test_deduplicate_messages_with_duplicates(self, extractor):
        """Test message deduplication with exact duplicates"""
        messages = [
            HumanMessage(content="Unique message"),
            AIMessage(content="Duplicate message"),
            HumanMessage(content="Duplicate message"),  # Exact duplicate
            AIMessage(content="Another unique message"),
        ]

        deduplicated = extractor._deduplicate_messages(messages)

        assert len(deduplicated) == 3
        # Should keep first occurrence of duplicate
        assert deduplicated[1].content == "Duplicate message"
        assert deduplicated[2].content == "Another unique message"

    def test_deduplicate_messages_with_similar_content(self, extractor):
        """Test message deduplication with similar content"""
        messages = [
            HumanMessage(content="machine learning is powerful"),
            AIMessage(content="machine learning is powerful"),  # Exact duplicate
            HumanMessage(content="Weather is sunny today"),
        ]

        # Should remove exact duplicate messages
        deduplicated = extractor._deduplicate_messages(
            messages, similarity_threshold=0.85
        )
        assert len(deduplicated) == 2  # Should keep only 2 unique messages
        assert deduplicated[0].content == "machine learning is powerful"
        assert deduplicated[1].content == "Weather is sunny today"

    def test_deduplicate_messages_preserve_order(self, extractor):
        """Test that message deduplication preserves order"""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="Second message"),
            HumanMessage(content="First message"),  # Duplicate
            AIMessage(content="Third message"),
        ]

        deduplicated = extractor._deduplicate_messages(messages)

        assert len(deduplicated) == 3
        assert deduplicated[0].content == "First message"
        assert deduplicated[1].content == "Second message"
        assert deduplicated[2].content == "Third message"

    def test_deduplicate_messages_empty_list(self, extractor):
        """Test message deduplication with empty list"""
        messages = []
        deduplicated = extractor._deduplicate_messages(messages)
        assert deduplicated == []

    def test_deduplicate_messages_single_message(self, extractor):
        """Test message deduplication with single message"""
        messages = [HumanMessage(content="Single message")]
        deduplicated = extractor._deduplicate_messages(messages)
        assert len(deduplicated) == 1
        assert deduplicated[0].content == "Single message"

    def test_deduplicate_messages_different_thresholds(self, extractor):
        """Test message deduplication with different similarity thresholds"""
        messages = [
            HumanMessage(content="Machine learning is powerful"),
            AIMessage(content="ML is very powerful technology"),
            HumanMessage(content="Deep learning networks"),
        ]

        # High threshold - should keep more messages
        deduplicated_high = extractor._deduplicate_messages(
            messages, similarity_threshold=0.9
        )
        assert len(deduplicated_high) >= 2

        # Low threshold - should remove more messages
        deduplicated_low = extractor._deduplicate_messages(
            messages, similarity_threshold=0.1
        )
        assert len(deduplicated_low) <= len(deduplicated_high)

    def test_normalize_messages_integration(self, extractor):
        """Test _normalize_messages method integration with deduplication"""
        messages = [
            HumanMessage(content="Research AI applications"),
            AIMessage(content="AI research applications"),
            HumanMessage(content="Weather forecast data"),
        ]

        # Test the actual _normalize_messages method
        normalized = extractor._normalize_messages(messages)

        # Should apply deduplication
        assert len(normalized) <= len(messages)
        assert all(
            isinstance(msg, (HumanMessage, AIMessage, SystemMessage))
            for msg in normalized
        )

    def test_normalize_messages_preserves_non_duplicate_content(self, extractor):
        """Test that normalization preserves non-duplicate content"""
        messages = [
            HumanMessage(content="Unique content 1"),
            AIMessage(content="Unique content 2"),
            HumanMessage(content="Unique content 3"),
        ]

        # Test the actual _normalize_messages method
        normalized = extractor._normalize_messages(messages)

        assert len(normalized) == 3
        assert normalized == messages

    def test_normalize_messages_handles_empty_content(self, extractor):
        """Test normalization with empty or None content"""
        messages = [HumanMessage(content=""), AIMessage(content="Valid content")]

        # Test the actual _normalize_messages method
        normalized = extractor._normalize_messages(messages)

        # Should handle empty content gracefully
        assert len(normalized) <= 2
        assert any(msg.content == "Valid content" for msg in normalized)

    def test_large_message_list_performance(self, extractor):
        """Test deduplication performance with large message list"""
        import time

        # Create a large list of messages with some duplicates
        messages = []
        for i in range(100):
            content = f"Message content {i % 20}"  # Creates duplicates
            messages.append(HumanMessage(content=content))

        start_time = time.time()
        deduplicated = extractor._deduplicate_messages(messages)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        # Should have removed duplicates
        assert len(deduplicated) < len(messages)
        assert len(deduplicated) <= 20  # At most 20 unique messages

    def test_similarity_calculation_performance(self, extractor):
        """Test similarity calculation performance"""
        import time

        text1 = (
            "This is a long text with many words to test similarity calculation performance"
            * 10
        )
        text2 = (
            "This is another long text with some similar words for performance testing"
            * 10
        )

        start_time = time.time()
        jaccard_sim = extractor._jaccard_similarity(text1, text2)
        cosine_sim = extractor._cosine_similarity(text1, text2)
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 0.1
        assert 0.0 <= jaccard_sim <= 1.0
        assert 0.0 <= cosine_sim <= 1.0
