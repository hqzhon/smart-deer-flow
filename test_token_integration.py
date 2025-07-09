#!/usr/bin/env python3
"""
Test script to verify token counting integration
"""

import sys

sys.path.append(".")

from src.utils.token_counter import get_token_counter
from src.utils.content_processor import ContentProcessor


def test_token_counter():
    """Test basic token counter functionality"""
    print("=== Testing TokenCounter ===")

    # Test basic token counting
    tc = get_token_counter()

    test_text = "Hello world! This is a test message for token counting."
    result = tc.count_tokens(test_text, "deepseek-chat")

    print(f"Text: {test_text}")
    print(f"Token count: {result.total_tokens}")
    print(f"Model: {result.model_name}")
    print(f"Encoding: {result.encoding_name}")

    return result.total_tokens > 0


def test_content_processor():
    """Test ContentProcessor integration"""
    print("\n=== Testing ContentProcessor Integration ===")

    cp = ContentProcessor()

    test_text = "This is a longer test message to verify that the ContentProcessor can accurately count tokens using the new tiktoken integration."

    # Test accurate token counting
    accurate_result = cp.count_tokens_accurate(test_text, "deepseek-chat")
    print(f"Accurate token count: {accurate_result.total_tokens}")

    # Test estimate tokens (should now use accurate counting)
    estimated_tokens = cp.estimate_tokens(test_text, "deepseek-chat")
    print(f"Estimated tokens: {estimated_tokens}")

    # Test token limit checking
    within_limit, token_result = cp.check_content_token_limit(
        test_text, "deepseek-chat", 1000, 0.8
    )
    print(f"Within limit (1000): {within_limit}")
    print(f"Token details: {token_result.total_tokens} tokens")

    return accurate_result.total_tokens > 0


def main():
    """Main test function"""
    print("Starting token counting integration tests...\n")

    try:
        # Test TokenCounter
        tc_success = test_token_counter()
        print(f"TokenCounter test: {'PASSED' if tc_success else 'FAILED'}")

        # Test ContentProcessor
        cp_success = test_content_processor()
        print(f"ContentProcessor test: {'PASSED' if cp_success else 'FAILED'}")

        # Overall result
        overall_success = tc_success and cp_success
        print(f"\n=== Overall Result: {'PASSED' if overall_success else 'FAILED'} ===")

        return 0 if overall_success else 1

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
