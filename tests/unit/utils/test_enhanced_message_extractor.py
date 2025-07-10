"""Tests for Enhanced Message Extractor

Tests the comprehensive message extraction capabilities and various message formats.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.utils.enhanced_message_extractor import (
    EnhancedMessageExtractor,
    MessagePattern,
    get_global_message_extractor
)


class TestEnhancedMessageExtractor:
    """Test enhanced message extractor functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.extractor = EnhancedMessageExtractor(max_search_depth=10, enable_caching=False)
        
        # Create test messages
        self.test_messages = [
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you!"),
            SystemMessage(content="You are a helpful assistant.")
        ]
        
        # Create dict format messages
        self.dict_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        # Create string messages
        self.string_messages = [
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "You are a helpful assistant."
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
        kwargs = {
            "input": {
                "messages": self.test_messages,
                "model": "gpt-4"
            }
        }
        
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
            "config": {
                "llm_params": {
                    "input_data": {
                        "messages": self.test_messages
                    }
                }
            }
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
        kwargs = {
            "messages": self.test_messages,
            "model": "gpt-3.5-turbo"
        }
        
        extracted, model_info = self.extractor.extract_messages_and_model(args, kwargs)
        
        assert model_info.model_name == "gpt-3.5-turbo"
        assert model_info.source == "kwargs.model"

    def test_model_extraction_from_config(self):
        """Test model name extraction from config"""
        args = ()
        kwargs = {
            "messages": self.test_messages,
            "config": {"model": "gemini-pro"}
        }
        
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
            HumanMessage(content="Valid message 2")
        ]
        assert self.extractor._validate_messages(valid_messages)
        
        # Invalid messages (empty content)
        invalid_messages = [
            HumanMessage(content=""),
            HumanMessage(content="   ")
        ]
        assert not self.extractor._validate_messages(invalid_messages)
        
        # Mixed valid/invalid
        mixed_messages = [
            HumanMessage(content="Valid message"),
            HumanMessage(content="")
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
            "String message"
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
            ((self.test_messages,), {})
        ]
        
        for args, kwargs in test_cases:
            self.extractor.extract_messages_and_model(args, kwargs)
        
        stats = self.extractor.get_extraction_stats()
        
        assert stats['total_extractions'] == 3
        assert stats['successful_extractions'] == 3
        assert stats['pattern_usage'][MessagePattern.DIRECT_KWARGS] >= 1
        assert stats['pattern_usage'][MessagePattern.NESTED_INPUT] >= 1
        assert stats['pattern_usage'][MessagePattern.ARGS_LIST] >= 1

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
        assert stats['cache_hits'] >= 1
        
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
                        "parameters": {
                            "input": {
                                "messages": self.test_messages
                            }
                        }
                    }
                }
            }
        }
        
        extracted, model_info = self.extractor.extract_messages_and_model((), complex_kwargs)
        
        assert extracted.messages == self.test_messages
        assert extracted.pattern == MessagePattern.DEEP_NESTED
        assert "request" in extracted.source_location

    def test_max_search_depth_limit(self):
        """Test that search depth limit is respected"""
        # Create very deeply nested structure (beyond max depth)
        deep_kwargs = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": {"messages": self.test_messages}}}}}}}
        
        # Create extractor with low max depth
        shallow_extractor = EnhancedMessageExtractor(max_search_depth=3)
        
        extracted, model_info = shallow_extractor.extract_messages_and_model((), deep_kwargs)
        
        # Should not find messages due to depth limit
        assert extracted.messages is None
        assert "max_depth_reached" in extracted.source_location or "failed" in extracted.source_location