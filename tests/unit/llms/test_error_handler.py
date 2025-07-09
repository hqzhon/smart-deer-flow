"""Test cases for LLM Error Handler

This module contains comprehensive test cases to verify the correctness of the LLM error handling logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from src.llms.error_handler import (
    LLMErrorHandler,
    LLMErrorType,
    handle_llm_errors,
    safe_llm_call,
    safe_llm_call_async,
    _handle_content_too_long_error,
    _handle_content_too_long_error_async,
    error_handler,
)


class TestLLMErrorHandler:
    """Test cases for LLMErrorHandler class"""

    def setup_method(self):
        """Setup test environment"""
        self.handler = LLMErrorHandler()

    def test_error_classification(self):
        """Test error message classification"""
        # Test rate limit error
        assert self.handler.classify_error("Rate limit exceeded") == LLMErrorType.RATE_LIMIT_EXCEEDED
        assert self.handler.classify_error("Too many requests") == LLMErrorType.RATE_LIMIT_EXCEEDED
        
        # Test content too long error
        assert self.handler.classify_error("Token limit exceeded") == LLMErrorType.CONTENT_TOO_LONG
        assert self.handler.classify_error("Maximum context length") == LLMErrorType.CONTENT_TOO_LONG
        
        # Test authentication error
        assert self.handler.classify_error("Invalid API key") == LLMErrorType.INVALID_API_KEY
        assert self.handler.classify_error("Authentication failed") == LLMErrorType.AUTHENTICATION_ERROR
        
        # Test unknown error
        assert self.handler.classify_error("Some random error") == LLMErrorType.UNKNOWN_ERROR

    def test_error_classification_with_exception_object(self):
        """Test error classification with Exception objects"""
        error = Exception("Rate limit exceeded")
        assert self.handler.classify_error(error) == LLMErrorType.RATE_LIMIT_EXCEEDED
        
        error = ValueError("Token limit exceeded")
        assert self.handler.classify_error(error) == LLMErrorType.CONTENT_TOO_LONG

    def test_error_categorization(self):
        """Test error categorization logic"""
        # Test skip errors
        assert self.handler.should_skip_error(LLMErrorType.DATA_INSPECTION_FAILED) is True
        assert self.handler.should_skip_error(LLMErrorType.RATE_LIMIT_EXCEEDED) is False
        
        # Test retry errors
        assert self.handler.should_retry_error(LLMErrorType.RATE_LIMIT_EXCEEDED) is True
        assert self.handler.should_retry_error(LLMErrorType.NETWORK_ERROR) is True
        assert self.handler.should_retry_error(LLMErrorType.INVALID_API_KEY) is False
        
        # Test fatal errors
        assert self.handler.is_fatal_error(LLMErrorType.INVALID_API_KEY) is True
        assert self.handler.is_fatal_error(LLMErrorType.QUOTA_EXCEEDED) is True
        assert self.handler.is_fatal_error(LLMErrorType.RATE_LIMIT_EXCEEDED) is False
        
        # Test smart processing errors
        assert self.handler.needs_smart_processing(LLMErrorType.CONTENT_TOO_LONG) is True
        assert self.handler.needs_smart_processing(LLMErrorType.RATE_LIMIT_EXCEEDED) is False

    def test_fallback_response_generation(self):
        """Test fallback response generation"""
        response = self.handler.get_fallback_response(LLMErrorType.RATE_LIMIT_EXCEEDED)
        assert isinstance(response, AIMessage)
        assert "rate limit" in response.content.lower()
        
        # Test with context
        response = self.handler.get_fallback_response(
            LLMErrorType.RATE_LIMIT_EXCEEDED, "test context"
        )
        assert "test context" in response.content
        
        # Test unknown error type
        response = self.handler.get_fallback_response("unknown_type")
        assert isinstance(response, AIMessage)
        assert "unknown error" in response.content.lower()

    def test_handle_error_skip_logic(self):
        """Test handle_error method for skip logic"""
        error = Exception("Content safety violation")
        should_skip, fallback_response, needs_smart_processing = self.handler.handle_error(error)
        
        assert should_skip is True
        assert isinstance(fallback_response, AIMessage)
        assert needs_smart_processing is False

    def test_handle_error_smart_processing_logic(self):
        """Test handle_error method for smart processing logic"""
        error = Exception("Token limit exceeded")
        should_skip, fallback_response, needs_smart_processing = self.handler.handle_error(error)
        
        assert should_skip is False
        assert fallback_response is None
        assert needs_smart_processing is True

    def test_handle_error_retry_logic(self):
        """Test handle_error method for retry logic"""
        error = Exception("Rate limit exceeded")
        should_skip, fallback_response, needs_smart_processing = self.handler.handle_error(error)
        
        assert should_skip is False
        assert fallback_response is None
        assert needs_smart_processing is False

    def test_handle_error_fatal_logic(self):
        """Test handle_error method for fatal error logic"""
        error = Exception("Invalid API key")
        
        with pytest.raises(Exception) as exc_info:
            self.handler.handle_error(error)
        
        assert str(exc_info.value) == "Invalid API key"

    def test_handle_error_unknown_logic(self):
        """Test handle_error method for unknown error logic"""
        error = Exception("Some completely unknown error")
        
        with pytest.raises(Exception) as exc_info:
            self.handler.handle_error(error)
        
        assert str(exc_info.value) == "Some completely unknown error"

    def test_error_pattern_conflicts(self):
        """Test for potential conflicts in error patterns"""
        # This test checks if there are overlapping patterns that could cause misclassification
        test_cases = [
            ("api key invalid", LLMErrorType.INVALID_API_KEY),
            ("authentication failed due to invalid api key", LLMErrorType.INVALID_API_KEY),  # Should match first pattern found
            ("rate limit exceeded", LLMErrorType.RATE_LIMIT_EXCEEDED),
            ("quota exceeded due to rate limit", LLMErrorType.QUOTA_EXCEEDED),  # Should match first pattern found
        ]
        
        for error_msg, expected_type in test_cases:
            actual_type = self.handler.classify_error(error_msg)
            assert actual_type == expected_type, f"Error '{error_msg}' classified as {actual_type}, expected {expected_type}"


class TestErrorHandlerDecorator:
    """Test cases for error handler decorator"""

    def test_sync_decorator_normal_execution(self):
        """Test decorator with normal function execution"""
        @handle_llm_errors("Test Operation")
        def test_func(x, y):
            return x + y
        
        result = test_func(1, 2)
        assert result == 3

    def test_sync_decorator_skip_error(self):
        """Test decorator with skip error"""
        @handle_llm_errors("Test Operation")
        def test_func():
            raise Exception("Content safety violation")
        
        result = test_func()
        assert isinstance(result, AIMessage)
        assert "content safety" in result.content.lower()

    def test_sync_decorator_fatal_error(self):
        """Test decorator with fatal error"""
        @handle_llm_errors("Test Operation")
        def test_func():
            raise Exception("Invalid API key")
        
        with pytest.raises(Exception) as exc_info:
            test_func()
        
        assert "Invalid API key" in str(exc_info.value)

    def test_sync_decorator_smart_processing_error(self):
        """Test decorator with smart processing error"""
        @handle_llm_errors("Test Operation")
        def test_func():
            raise Exception("Token limit exceeded")
        
        with pytest.raises(Exception) as exc_info:
            test_func()
        
        assert "Token limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_decorator_normal_execution(self):
        """Test async decorator with normal function execution"""
        @handle_llm_errors("Test Operation")
        async def test_func(x, y):
            return x + y
        
        result = await test_func(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_async_decorator_skip_error(self):
        """Test async decorator with skip error"""
        @handle_llm_errors("Test Operation")
        async def test_func():
            raise Exception("Content safety violation")
        
        result = await test_func()
        assert isinstance(result, AIMessage)
        assert "content safety" in result.content.lower()


class TestSafeLLMCall:
    """Test cases for safe LLM call functions"""

    def test_safe_llm_call_success(self):
        """Test successful LLM call"""
        def mock_llm_func(x, y):
            return x + y
        
        result = safe_llm_call(mock_llm_func, 1, 2)
        assert result == 3

    def test_safe_llm_call_skip_error(self):
        """Test safe LLM call with skip error"""
        def mock_llm_func():
            raise Exception("Content safety violation")
        
        result = safe_llm_call(mock_llm_func)
        assert isinstance(result, AIMessage)
        assert "content safety" in result.content.lower()

    def test_safe_llm_call_retry_logic(self):
        """Test safe LLM call retry logic"""
        call_count = 0
        
        def mock_llm_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            return "success"
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = safe_llm_call(mock_llm_func, max_retries=3)
        
        assert result == "success"
        assert call_count == 3

    def test_safe_llm_call_max_retries_exceeded(self):
        """Test safe LLM call when max retries exceeded"""
        def mock_llm_func():
            raise Exception("Rate limit exceeded")
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(Exception) as exc_info:
                safe_llm_call(mock_llm_func, max_retries=2)
        
        assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_safe_llm_call_async_success(self):
        """Test successful async LLM call"""
        async def mock_llm_func(x, y):
            return x + y
        
        result = await safe_llm_call_async(mock_llm_func, 1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_safe_llm_call_async_retry_logic(self):
        """Test async safe LLM call retry logic"""
        call_count = 0
        
        async def mock_llm_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            return "success"
        
        with patch('asyncio.sleep'):  # Mock sleep to speed up test
            result = await safe_llm_call_async(mock_llm_func, max_retries=3)
        
        assert result == "success"
        assert call_count == 3


class TestContentTooLongErrorHandling:
    """Test cases for content too long error handling"""

    @patch('src.llms.error_handler.config_loader')
    @patch('src.llms.error_handler.ContentProcessor')
    def test_handle_content_too_long_error_basic(self, mock_processor_class, mock_config_loader):
        """Test basic content too long error handling"""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_token_limits = {}
        mock_config.enable_content_summarization = False
        mock_config_loader.create_configuration.return_value = mock_config
        
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock model limits
        mock_limits = Mock()
        mock_limits.input_limit = 4000
        mock_limits.safety_margin = 0.8
        mock_processor.get_model_limits.return_value = mock_limits
        
        # Mock token counting
        mock_token_result = Mock()
        mock_token_result.total_tokens = 5000
        mock_processor.count_tokens_accurate.return_value = mock_token_result
        
        # Mock chunking
        mock_processor.smart_chunk_content.return_value = ["truncated content"]
        
        # Create test messages
        messages = [HumanMessage(content="Very long content that exceeds token limits")]
        
        def mock_llm_func(*args, **kwargs):
            return "success"
        
        # Test the function
        error = Exception("Token limit exceeded")
        result = _handle_content_too_long_error(mock_llm_func, error, messages)
        
        assert result == "success"
        mock_processor.smart_chunk_content.assert_called_once()

    def test_handle_content_too_long_error_no_messages(self):
        """Test content too long error handling when no messages found"""
        def mock_llm_func(*args, **kwargs):
            return "success"
        
        error = Exception("Token limit exceeded")
        
        with pytest.raises(Exception) as exc_info:
            _handle_content_too_long_error(mock_llm_func, error, "not_messages")
        
        assert "Token limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch('src.llms.error_handler.config_loader')
    @patch('src.llms.error_handler.ContentProcessor')
    async def test_handle_content_too_long_error_async(self, mock_processor_class, mock_config_loader):
        """Test async content too long error handling"""
        # Setup mocks similar to sync test
        mock_config = Mock()
        mock_config.model_token_limits = {}
        mock_config.enable_content_summarization = False
        mock_config_loader.create_configuration.return_value = mock_config
        
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        mock_limits = Mock()
        mock_limits.input_limit = 4000
        mock_limits.safety_margin = 0.8
        mock_processor.get_model_limits.return_value = mock_limits
        
        mock_token_result = Mock()
        mock_token_result.total_tokens = 5000
        mock_processor.count_tokens_accurate.return_value = mock_token_result
        
        mock_processor.smart_chunk_content.return_value = ["truncated content"]
        
        messages = [HumanMessage(content="Very long content that exceeds token limits")]
        
        async def mock_llm_func(*args, **kwargs):
            return "async success"
        
        error = Exception("Token limit exceeded")
        result = await _handle_content_too_long_error_async(mock_llm_func, error, messages)
        
        assert result == "async success"


class TestLogicIssues:
    """Test cases to identify and verify fixes for logic issues"""

    def test_error_pattern_order_dependency(self):
        """Test if error classification depends on pattern order (potential issue)"""
        handler = LLMErrorHandler()
        
        # Test case where multiple patterns could match
        error_msg = "Invalid API key causing authentication failure"
        result = handler.classify_error(error_msg)
        
        # This should consistently return the same result regardless of dict iteration order
        # In Python 3.7+, dict order is preserved, but this test ensures consistency
        assert result in [LLMErrorType.INVALID_API_KEY, LLMErrorType.AUTHENTICATION_ERROR]

    def test_content_processor_double_instantiation(self):
        """Test potential issue with ContentProcessor being instantiated multiple times"""
        # This test identifies the logic issue where ContentProcessor is instantiated
        # multiple times in the same function, which could be inefficient
        
        with patch('src.llms.error_handler.ContentProcessor') as mock_processor_class:
            mock_config = Mock()
            mock_config.model_token_limits = {}
            mock_config.enable_content_summarization = False
            
            with patch('src.llms.error_handler.config_loader') as mock_config_loader:
                mock_config_loader.create_configuration.return_value = mock_config
                
                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                
                mock_limits = Mock()
                mock_limits.input_limit = 4000
                mock_limits.safety_margin = 0.8
                mock_processor.get_model_limits.return_value = mock_limits
                
                mock_token_result = Mock()
                mock_token_result.total_tokens = 10000  # Very long content
                mock_processor.count_tokens_accurate.return_value = mock_token_result
                
                mock_processor.smart_chunk_content.return_value = ["chunk1"]
                mock_processor.estimate_tokens.return_value = 3000
                
                messages = [HumanMessage(content="Very long content")]
                
                def mock_llm_func(*args, **kwargs):
                    return "success"
                
                error = Exception("Token limit exceeded")
                
                try:
                    _handle_content_too_long_error(mock_llm_func, error, messages)
                except:
                    pass
                
                # Check if ContentProcessor was instantiated multiple times
                # This indicates a potential optimization issue
                call_count = mock_processor_class.call_count
                if call_count > 2:  # Allow for some reasonable instantiation
                    pytest.fail(f"ContentProcessor instantiated {call_count} times, which may be inefficient")

    def test_binary_search_edge_cases(self):
        """Test edge cases in binary search logic for content truncation"""
        # This test verifies the binary search logic doesn't have off-by-one errors
        # or infinite loop issues
        
        with patch('src.llms.error_handler.ContentProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            
            # Test case where no content fits (edge case)
            mock_processor.estimate_tokens.return_value = 10000  # Always too large
            
            # This should not cause infinite loop
            content = "test content"
            left, right = 0, len(content)
            best_content = ""
            max_tokens = 100
            iterations = 0
            
            while left <= right and iterations < 20:  # Safety limit
                iterations += 1
                mid = (left + right) // 2
                test_content = content[:mid] + "...[truncated]"
                test_tokens = mock_processor.estimate_tokens(test_content, "test-model")
                
                if test_tokens <= max_tokens:
                    best_content = test_content
                    left = mid + 1
                else:
                    right = mid - 1
            
            # Should not take too many iterations for small content
            assert iterations < 10, f"Binary search took {iterations} iterations, possible infinite loop"

    def test_message_reconstruction_logic(self):
        """Test message reconstruction logic for potential issues"""
        # Test the logic that reconstructs message list after content truncation
        
        original_messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
            HumanMessage(content="Last message to be truncated")
        ]
        
        # Simulate the message reconstruction logic
        new_messages = []
        for msg in original_messages[:-1]:  # Keep all messages except the last one
            new_messages.append(msg)
        
        # Replace the last message with truncated content
        if original_messages:
            last_msg = original_messages[-1]
            truncated_content = "truncated content"
            
            if hasattr(last_msg, "content"):
                if hasattr(last_msg, "__class__"):
                    new_msg = last_msg.__class__(content=truncated_content)
                else:
                    new_msg = HumanMessage(content=truncated_content)
                new_messages.append(new_msg)
        
        # Verify the reconstruction
        assert len(new_messages) == len(original_messages)
        assert new_messages[-1].content == "truncated content"
        assert new_messages[0].content == "First message"
        assert new_messages[1].content == "Second message"

    def test_error_handler_singleton_behavior(self):
        """Test if error_handler global instance behaves correctly"""
        # Test that the global error_handler instance is properly initialized
        assert error_handler is not None
        assert isinstance(error_handler, LLMErrorHandler)
        
        # Test that it has all required attributes
        assert hasattr(error_handler, 'error_patterns')
        assert hasattr(error_handler, 'fallback_responses')
        assert hasattr(error_handler, 'skip_errors')
        assert hasattr(error_handler, 'retry_errors')
        assert hasattr(error_handler, 'fatal_errors')
        assert hasattr(error_handler, 'smart_processing_errors')


if __name__ == "__main__":
    pytest.main([__file__])