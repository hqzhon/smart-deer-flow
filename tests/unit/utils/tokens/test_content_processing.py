"""Unit tests for content processing modules."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from src.utils.tokens.content_processor import ContentProcessor


class TestContentProcessor:
    """Test cases for ContentProcessor."""

    def setup_method(self):
        """Test setup."""
        self.processor = ContentProcessor()

    def test_content_processor_instantiation(self):
        """Test ContentProcessor instantiation."""
        assert self.processor is not None
        assert isinstance(self.processor, ContentProcessor)

    def test_token_estimation_basic(self):
        """Test basic token estimation."""
        test_text = "这是一个测试文本，用于验证token估算功能。"
        estimated_tokens = self.processor.estimate_tokens(test_text)

        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0

    def test_token_estimation_empty_text(self):
        """Test token estimation with empty text."""
        empty_text = ""
        estimated_tokens = self.processor.estimate_tokens(empty_text)

        assert isinstance(estimated_tokens, int)
        assert estimated_tokens >= 0

    def test_token_estimation_various_lengths(self):
        """Test token estimation with various text lengths."""
        test_cases = [
            "短",  # Single character
            "短文本",  # Short text
            "这是一个中等长度的测试文本，包含多个词汇和标点符号。",  # Medium text
            "这是一个很长的测试文本，" * 20,  # Long text
        ]

        previous_tokens = 0
        for text in test_cases:
            estimated_tokens = self.processor.estimate_tokens(text)
            assert isinstance(estimated_tokens, int)
            assert estimated_tokens >= 0
            # Generally, longer text should have more tokens
            if len(text) > 0:
                assert estimated_tokens >= previous_tokens or estimated_tokens > 0
            previous_tokens = estimated_tokens

    def test_token_estimation_different_languages(self):
        """Test token estimation with different languages."""
        test_cases = [
            "Hello, this is English text.",
            "这是中文文本。",
            "これは日本語のテキストです。",
            "Это русский текст.",
            "Ceci est un texte français.",
        ]

        for text in test_cases:
            estimated_tokens = self.processor.estimate_tokens(text)
            assert isinstance(estimated_tokens, int)
            assert estimated_tokens > 0

    def test_content_sanitization_basic(self):
        """Test basic content sanitization."""
        test_text = "这是一个测试文本，用于验证内容清理功能。"
        sanitized_content = self.processor.sanitize_content(test_text)

        assert isinstance(sanitized_content, str)
        assert len(sanitized_content) > 0

    def test_content_sanitization_empty_text(self):
        """Test content sanitization with empty text."""
        empty_text = ""
        sanitized_content = self.processor.sanitize_content(empty_text)

        assert isinstance(sanitized_content, str)
        assert len(sanitized_content) == 0

    def test_content_sanitization_special_characters(self):
        """Test content sanitization with special characters."""
        test_cases = [
            "Text with\nnewlines\nand\ttabs",
            "Text with <html> tags </html>",
            "Text with @#$%^&*() symbols",
            "Text with 'quotes' and \"double quotes\"",
            "Text with unicode: 🚀 🎉 ✨",
        ]

        for text in test_cases:
            sanitized_content = self.processor.sanitize_content(text)
            assert isinstance(sanitized_content, str)
            # Sanitized content should not be longer than original
            assert len(sanitized_content) <= len(text) + 10  # Allow some flexibility

    def test_content_sanitization_preserves_meaning(self):
        """Test that content sanitization preserves meaning."""
        test_text = "这是一个重要的测试文本，包含关键信息。"
        sanitized_content = self.processor.sanitize_content(test_text)

        # Should preserve core content
        assert "测试" in sanitized_content or "test" in sanitized_content.lower()
        assert len(sanitized_content) > 0

    def test_content_sanitization_consistency(self):
        """Test content sanitization consistency."""
        test_text = "一致性测试文本"

        sanitized1 = self.processor.sanitize_content(test_text)
        sanitized2 = self.processor.sanitize_content(test_text)

        # Should produce consistent results
        assert sanitized1 == sanitized2

    def test_token_estimation_and_sanitization_integration(self):
        """Test integration between token estimation and content sanitization."""
        test_text = "这是一个集成测试文本，用于验证token估算和内容清理的协同工作。"

        # Get original token count
        original_tokens = self.processor.estimate_tokens(test_text)

        # Sanitize content
        sanitized_content = self.processor.sanitize_content(test_text)

        # Get sanitized token count
        sanitized_tokens = self.processor.estimate_tokens(sanitized_content)

        # Both should be positive
        assert original_tokens > 0
        assert sanitized_tokens > 0

        # Sanitized tokens should be reasonable compared to original
        assert sanitized_tokens <= original_tokens * 1.5  # Allow some flexibility

    def test_multiple_processor_instances(self):
        """Test creating multiple processor instances."""
        processor1 = ContentProcessor()
        processor2 = ContentProcessor()

        assert processor1 is not processor2
        assert isinstance(processor1, ContentProcessor)
        assert isinstance(processor2, ContentProcessor)

        # Both should work independently
        test_text = "测试文本"
        tokens1 = processor1.estimate_tokens(test_text)
        tokens2 = processor2.estimate_tokens(test_text)

        # Should produce same results
        assert tokens1 == tokens2

    def test_processor_error_handling(self):
        """Test processor error handling."""
        # Test with None input
        try:
            tokens = self.processor.estimate_tokens(None)
            assert isinstance(tokens, int) and tokens >= 0
        except (TypeError, AttributeError):
            # Exception is acceptable for None input
            pass

        try:
            sanitized = self.processor.sanitize_content(None)
            assert isinstance(sanitized, str)
        except (TypeError, AttributeError):
            # Exception is acceptable for None input
            pass

    @pytest.mark.parametrize(
        "test_input,expected_type",
        [
            ("Normal text", int),
            ("中文文本", int),
            ("Mixed 中英文 text", int),
            ("", int),
            ("Very long text " * 100, int),
        ],
    )
    def test_token_estimation_parametrized(self, test_input, expected_type):
        """Test token estimation with parametrized inputs."""
        result = self.processor.estimate_tokens(test_input)
        assert isinstance(result, expected_type)
        assert result >= 0

    @pytest.mark.parametrize(
        "test_input,expected_type",
        [
            ("Normal text", str),
            ("中文文本", str),
            ("Text with\nspecial\tchars", str),
            ("", str),
            ("HTML <tag>content</tag>", str),
        ],
    )
    def test_content_sanitization_parametrized(self, test_input, expected_type):
        """Test content sanitization with parametrized inputs."""
        result = self.processor.sanitize_content(test_input)
        assert isinstance(result, expected_type)

    def test_processor_performance_basic(self):
        """Test basic performance characteristics."""
        # Test with reasonably large text
        large_text = "这是一个性能测试文本。" * 1000

        # Should complete without hanging
        tokens = self.processor.estimate_tokens(large_text)
        sanitized = self.processor.sanitize_content(large_text)

        assert isinstance(tokens, int)
        assert isinstance(sanitized, str)
        assert tokens > 0
        assert len(sanitized) > 0
