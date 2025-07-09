"""Test cases for context management functionality

This module contains comprehensive tests to verify that context management
is properly applied to all LLM calls and prevents token limit errors.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.llms.error_handler import (
    safe_llm_call,
    safe_llm_call_async,
    _evaluate_and_optimize_context_before_call,
    _evaluate_and_optimize_context_before_call_sync,
    error_handler,
)
from src.utils.context_evaluator import (
    ContextEvaluationResult,
)
from src.utils.advanced_context_manager import CompressionStrategy


class TestContextManagement:
    """Test context management functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_llm = Mock()
        self.mock_llm.model_name = "deepseek-chat"
        self.mock_llm.invoke = Mock(return_value=AIMessage(content="Test response"))
        self.mock_llm.ainvoke = Mock(
            return_value=AIMessage(content="Test async response")
        )

        # Create test messages that would exceed token limits
        self.long_messages = [
            HumanMessage(content="A" * 100000),  # Very long content
            HumanMessage(content="B" * 100000),
            HumanMessage(content="C" * 100000),
        ]

        self.normal_messages = [
            HumanMessage(content="Short message 1"),
            HumanMessage(content="Short message 2"),
        ]

    @patch("src.utils.context_evaluator.get_global_context_evaluator")
    def test_context_evaluation_before_sync_call(self, mock_get_evaluator):
        """Test that context evaluation happens before synchronous LLM calls"""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_get_evaluator.return_value = mock_evaluator

        # Mock evaluation result indicating compression needed
        mock_metrics = Mock()
        mock_metrics.compression_needed = True
        mock_metrics.evaluation_result = ContextEvaluationResult.NEEDS_COMPRESSION
        mock_metrics.recommended_strategy = CompressionStrategy.TRUNCATE
        mock_metrics.current_tokens = 100000
        mock_metrics.max_tokens = 65536
        mock_metrics.utilization_ratio = 1.5

        mock_evaluator.evaluate_context_before_llm_call.return_value = mock_metrics

        # Mock optimization result
        optimized_messages = [HumanMessage(content="Optimized content")]
        optimization_info = {
            "original_tokens": 100000,
            "optimized_tokens": 30000,
            "tokens_saved": 70000,
        }
        mock_evaluator.optimize_context_for_llm_call.return_value = (
            optimized_messages,
            optimization_info,
        )

        # Test with messages in kwargs
        args = ()
        kwargs = {"messages": self.long_messages}

        result_args, result_kwargs = _evaluate_and_optimize_context_before_call_sync(
            self.mock_llm.invoke, args, kwargs, "Test Operation", "Test Context"
        )

        # Verify evaluator was called
        mock_evaluator.evaluate_context_before_llm_call.assert_called_once()
        mock_evaluator.optimize_context_for_llm_call.assert_called_once()

        # Verify messages were optimized
        assert result_kwargs["messages"] == optimized_messages

    @patch("src.utils.context_evaluator.get_global_context_evaluator")
    async def test_context_evaluation_before_async_call(self, mock_get_evaluator):
        """Test that context evaluation happens before asynchronous LLM calls"""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_get_evaluator.return_value = mock_evaluator

        # Mock evaluation result indicating no compression needed
        mock_metrics = Mock()
        mock_metrics.compression_needed = False
        mock_metrics.evaluation_result = ContextEvaluationResult.OPTIMAL
        mock_metrics.current_tokens = 30000
        mock_metrics.max_tokens = 65536
        mock_metrics.utilization_ratio = 0.45

        mock_evaluator.evaluate_context_before_llm_call.return_value = mock_metrics

        # Test with messages in args
        args = (self.normal_messages,)
        kwargs = {}

        result_args, result_kwargs = await _evaluate_and_optimize_context_before_call(
            self.mock_llm.ainvoke, args, kwargs, "Test Async Operation", "Test Context"
        )

        # Verify evaluator was called
        mock_evaluator.evaluate_context_before_llm_call.assert_called_once()

        # Verify original arguments returned (no optimization needed)
        assert result_args == args
        assert result_kwargs == kwargs

    @patch("src.utils.context_evaluator.get_global_context_evaluator")
    def test_safe_llm_call_with_context_evaluation(self, mock_get_evaluator):
        """Test that safe_llm_call applies context evaluation"""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_get_evaluator.return_value = mock_evaluator

        # Mock evaluation result
        mock_metrics = Mock()
        mock_metrics.compression_needed = False
        mock_metrics.evaluation_result = ContextEvaluationResult.OPTIMAL

        mock_evaluator.evaluate_context_before_llm_call.return_value = mock_metrics

        # Call safe_llm_call
        result = safe_llm_call(
            self.mock_llm.invoke,
            self.normal_messages,
            operation_name="Test Safe Call",
            context="Test Context",
        )

        # Verify LLM was called
        self.mock_llm.invoke.assert_called_once()

        # Verify evaluator was called
        mock_evaluator.evaluate_context_before_llm_call.assert_called_once()

        # Verify result
        assert result.content == "Test response"

    @patch("src.utils.context_evaluator.get_global_context_evaluator")
    async def test_safe_llm_call_async_with_context_evaluation(
        self, mock_get_evaluator
    ):
        """Test that safe_llm_call_async applies context evaluation"""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_get_evaluator.return_value = mock_evaluator

        # Mock evaluation result
        mock_metrics = Mock()
        mock_metrics.compression_needed = False
        mock_metrics.evaluation_result = ContextEvaluationResult.OPTIMAL

        mock_evaluator.evaluate_context_before_llm_call.return_value = mock_metrics

        # Make ainvoke async
        async def mock_ainvoke(*args, **kwargs):
            return AIMessage(content="Test async response")

        self.mock_llm.ainvoke = mock_ainvoke

        # Call safe_llm_call_async
        result = await safe_llm_call_async(
            self.mock_llm.ainvoke,
            self.normal_messages,
            operation_name="Test Safe Async Call",
            context="Test Context",
        )

        # Verify evaluator was called
        mock_evaluator.evaluate_context_before_llm_call.assert_called_once()

        # Verify result
        assert result.content == "Test async response"

    def test_context_evaluation_disabled(self):
        """Test that context evaluation can be disabled"""
        with patch(
            "src.utils.context_evaluator.get_global_context_evaluator"
        ) as mock_get_evaluator:
            mock_evaluator = Mock()
            mock_get_evaluator.return_value = mock_evaluator

            # Call safe_llm_call with context evaluation disabled
            result = safe_llm_call(
                self.mock_llm.invoke,
                self.normal_messages,
                operation_name="Test No Context Eval",
                context="Test Context",
            )

            # Verify LLM was called
            self.mock_llm.invoke.assert_called_once()

            # Verify evaluator was NOT called
            mock_evaluator.evaluate_context_before_llm_call.assert_not_called()

            # Verify result
            assert result.content == "Test response"

    def test_token_limit_error_handling(self):
        """Test that token limit errors trigger smart processing"""
        # Mock LLM to raise token limit error
        token_error = Exception(
            "This model's maximum context length is 65536 tokens. However, you requested 1228996 tokens"
        )
        self.mock_llm.invoke.side_effect = token_error

        with patch(
            "src.llms.error_handler._handle_content_too_long_error"
        ) as mock_smart_processing:
            mock_smart_processing.return_value = AIMessage(
                content="Smart processed response"
            )

            # Call safe_llm_call
            result = safe_llm_call(
                self.mock_llm.invoke,
                self.long_messages,
                operation_name="Test Token Limit Error",
                context="Test Context",
                enable_smart_processing=True,
            )

            # Verify smart processing was called
            mock_smart_processing.assert_called_once()

            # Verify result
            assert result.content == "Smart processed response"

    def test_error_classification(self):
        """Test that errors are correctly classified"""
        # Test token limit error
        token_error = "This model's maximum context length is 65536 tokens"
        error_type = error_handler.classify_error(token_error)
        assert error_type == "content_too_long"

        # Test rate limit error
        rate_error = "Rate limit exceeded"
        error_type = error_handler.classify_error(rate_error)
        assert error_type == "rate_limit_exceeded"

        # Test authentication error
        auth_error = "Invalid API key"
        error_type = error_handler.classify_error(auth_error)
        assert error_type == "invalid_api_key"

    def test_message_extraction_from_args(self):
        """Test that messages are correctly extracted from function arguments"""
        with patch(
            "src.utils.context_evaluator.get_global_context_evaluator"
        ) as mock_get_evaluator:
            mock_evaluator = Mock()
            mock_get_evaluator.return_value = mock_evaluator

            # Mock evaluation result
            mock_metrics = Mock()
            mock_metrics.compression_needed = False
            mock_metrics.evaluation_result = ContextEvaluationResult.OPTIMAL

            mock_evaluator.evaluate_context_before_llm_call.return_value = mock_metrics

            # Test with messages in args
            args = (self.normal_messages,)
            kwargs = {}

            result_args, result_kwargs = (
                _evaluate_and_optimize_context_before_call_sync(
                    self.mock_llm.invoke, args, kwargs, "Test Operation", "Test Context"
                )
            )

            # Verify evaluator was called with extracted messages
            mock_evaluator.evaluate_context_before_llm_call.assert_called_once()
            call_args = mock_evaluator.evaluate_context_before_llm_call.call_args[0]
            assert call_args[0] == self.normal_messages  # messages parameter

    def test_message_extraction_from_input_dict(self):
        """Test that messages are correctly extracted from input dictionary"""
        with patch(
            "src.utils.context_evaluator.get_global_context_evaluator"
        ) as mock_get_evaluator:
            mock_evaluator = Mock()
            mock_get_evaluator.return_value = mock_evaluator

            # Mock evaluation result
            mock_metrics = Mock()
            mock_metrics.compression_needed = False
            mock_metrics.evaluation_result = ContextEvaluationResult.OPTIMAL

            mock_evaluator.evaluate_context_before_llm_call.return_value = mock_metrics

            # Test with messages in input dict
            args = ()
            kwargs = {"input": {"messages": self.normal_messages}}

            result_args, result_kwargs = (
                _evaluate_and_optimize_context_before_call_sync(
                    self.mock_llm.invoke, args, kwargs, "Test Operation", "Test Context"
                )
            )

            # Verify evaluator was called with extracted messages
            mock_evaluator.evaluate_context_before_llm_call.assert_called_once()
            call_args = mock_evaluator.evaluate_context_before_llm_call.call_args[0]
            assert call_args[0] == self.normal_messages  # messages parameter

    def test_model_name_extraction(self):
        """Test that model name is correctly extracted from LLM instance"""
        with patch(
            "src.utils.context_evaluator.get_global_context_evaluator"
        ) as mock_get_evaluator:
            mock_evaluator = Mock()
            mock_get_evaluator.return_value = mock_evaluator

            # Mock evaluation result
            mock_metrics = Mock()
            mock_metrics.compression_needed = False
            mock_metrics.evaluation_result = ContextEvaluationResult.OPTIMAL

            mock_evaluator.evaluate_context_before_llm_call.return_value = mock_metrics

            # Test with LLM instance having model_name
            self.mock_llm.model_name = "test-model"

            args = (self.normal_messages,)
            kwargs = {}

            result_args, result_kwargs = (
                _evaluate_and_optimize_context_before_call_sync(
                    self.mock_llm.invoke, args, kwargs, "Test Operation", "Test Context"
                )
            )

            # Verify evaluator was called with correct model name
            mock_evaluator.evaluate_context_before_llm_call.assert_called_once()
            call_args = mock_evaluator.evaluate_context_before_llm_call.call_args[0]
            assert call_args[1] == "test-model"  # model_name parameter


class TestContextManagementIntegration:
    """Integration tests for context management"""

    def test_health_check_uses_safe_call(self):
        """Test that health check uses safe LLM call"""
        from src.utils.health_check import LLMHealthCheck

        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=AIMessage(content="pong"))

        health_check = LLMHealthCheck("test_llm", mock_llm)

        with patch("src.llms.error_handler.safe_llm_call") as mock_safe_call:
            mock_safe_call.return_value = AIMessage(content="pong")

            # Run health check
            import asyncio

            result = asyncio.run(health_check.check())

            # Verify safe_llm_call was used
            mock_safe_call.assert_called_once()

            # Verify health check passed
            assert result.status.value == "healthy"

    def test_background_investigation_uses_safe_call(self):
        """Test that background investigation uses safe LLM call for search tools"""
        from src.graph.nodes import background_investigation_node

        # Mock state and config
        state = {"research_topic": "test query"}
        config = Mock()

        mock_configurable = Mock()
        mock_configurable.max_search_results = 5

        with patch(
            "src.config.configuration.Configuration.from_runnable_config"
        ) as mock_from_config:
            mock_from_config.return_value = mock_configurable

            with patch("src.llms.error_handler.safe_llm_call") as mock_safe_call:
                mock_safe_call.return_value = [
                    {"title": "Test", "content": "Test content"}
                ]

                with patch("src.graph.nodes.SELECTED_SEARCH_ENGINE", "tavily"):
                    # Run background investigation
                    result = background_investigation_node(state, config)

                    # Verify safe_llm_call was used
                    mock_safe_call.assert_called_once()

                    # Verify result
                    assert "background_investigation_results" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
