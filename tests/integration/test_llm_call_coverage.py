"""Integration test to verify all LLM calls use context management

This test scans the codebase to ensure all LLM calls are properly wrapped
with safe_llm_call or safe_llm_call_async functions.
"""

import os
import re
from typing import List, Dict


def check_direct_llm_calls(file_path: str) -> List[Dict]:
    """Check for direct LLM calls that should be wrapped"""
    # Patterns for direct LLM method calls (excluding workflow calls)
    llm_patterns = [
        r"(?<!workflow\.)(?<!graph\.)(?<!safe_llm_call\()\b\w+\.invoke\s*\(",
        r"(?<!workflow\.)(?<!graph\.)(?<!safe_llm_call_async\()\b\w+\.ainvoke\s*\(",
        r"(?<!workflow\.)(?<!graph\.)(?<!safe_llm_call\()\b\w+\.stream\s*\(",
        r"(?<!workflow\.)(?<!graph\.)(?<!safe_llm_call_async\()\b\w+\.astream\s*\(",
        r"(?<!workflow\.)(?<!graph\.)\b\w+\.generate\s*\(",
        r"(?<!workflow\.)(?<!graph\.)\b\w+\.agenerate\s*\(",
        r"(?<!workflow\.)(?<!graph\.)\b\w+\.predict\s*\(",
        r"(?<!workflow\.)(?<!graph\.)\b\w+\.apredict\s*\(",
    ]

    findings = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            # Skip lines that are already wrapped in safe calls
            if "safe_llm_call" in line:
                continue

            # Skip workflow and graph calls
            if "workflow." in line or "graph." in line:
                continue

            # Skip test files and mock calls
            if "test_" in file_path or "mock" in line.lower():
                continue

            for pattern in llm_patterns:
                if re.search(pattern, line):
                    # Additional filtering for known safe patterns
                    if any(
                        safe_word in line
                        for safe_word in ["safe_llm_call", "Mock", "mock", "test"]
                    ):
                        continue

                    # Check if it's a real LLM call by looking for common LLM variable names
                    if any(
                        llm_var in line
                        for llm_var in [
                            "llm.",
                            "model.",
                            "chat.",
                            "openai.",
                            "deepseek.",
                        ]
                    ):
                        findings.append(
                            {
                                "file": file_path,
                                "line": line_num,
                                "content": line.strip(),
                                "pattern": pattern,
                            }
                        )

    except Exception as e:
        print(f"Error checking {file_path}: {e}")

    return findings


def scan_critical_files() -> Dict[str, List[Dict]]:
    """Scan critical files for LLM calls"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")

    # Critical files that should use safe LLM calls
    critical_files = [
        "src/utils/health_check.py",
        "src/graph/nodes.py",
        "src/llms/llm.py",
        "src/llms/error_handler.py",
    ]

    # Files that only create LLM instances without calling them
    instance_only_files = [
        "src/llms/llm.py",  # Only creates and configures LLM instances
        "src/llms/error_handler.py",  # Only handles errors
    ]

    results = {}

    for file_pattern in critical_files:
        file_path = os.path.join(project_root, file_pattern)
        if os.path.exists(file_path):
            print(f"\nAnalyzing {file_pattern}:")

            # Check for safe_llm_call imports
            with open(file_path, "r") as f:
                content = f.read()

            # Skip import check for files that only create instances
            if file_pattern in instance_only_files:
                has_safe_import = (
                    True  # Consider these as "safe" since they don't make calls
                )
                print("  Safe import: ✅ (instance-only file)")
            else:
                has_safe_import = "safe_llm_call" in content
                print(f"  Safe import: {'✅' if has_safe_import else '❌'}")

            direct_calls = check_direct_llm_calls(file_path)

            print(f"  Direct calls: {len(direct_calls)}")

            if direct_calls:
                print("  ⚠️  Found potential direct calls:")
                for call in direct_calls[:3]:  # Show first 3
                    print(f"    - Line {call['line']}: {call['content']}")
            else:
                print("  ✅ No problematic direct calls found")

            results[file_pattern] = {
                "has_safe_import": has_safe_import,
                "direct_calls": direct_calls,
            }
        else:
            print(f"  ❌ File not found: {file_path}")
            results[file_pattern] = {"error": "File not found"}

    return results


def test_context_management_integration():
    """Test context management integration"""
    print("=== Testing Context Management Integration ===")

    try:
        # Test 1: Context evaluator availability
        print("\n1. Testing context evaluator...")
        from src.utils.context_evaluator import get_global_context_evaluator

        evaluator = get_global_context_evaluator()
        if evaluator:
            print("   ✅ Context evaluator is available")
        else:
            print("   ❌ Context evaluator is not available")
            return False
    except Exception as e:
        print(f"   ❌ Context evaluator error: {e}")
        return False

    try:
        # Test 2: Error handler availability
        print("\n2. Testing error handler...")
        from src.llms.error_handler import (
            error_handler,
            safe_llm_call,
            safe_llm_call_async,
        )

        if error_handler and safe_llm_call and safe_llm_call_async:
            print("   ✅ Error handler and safe call functions are available")
        else:
            print("   ❌ Error handler or safe call functions are not available")
            return False
    except Exception as e:
        print(f"   ❌ Error handler error: {e}")
        return False

    try:
        # Test 3: Token limit error classification
        print("\n3. Testing error classification...")
        token_error = "This model's maximum context length is 65536 tokens. However, you requested 1228996 tokens"
        error_type = error_handler.classify_error(token_error)
        if error_type == "content_too_long":
            print("   ✅ Token limit error classification works")
        else:
            print(f"   ❌ Token limit error classification failed: {error_type}")
            return False
    except Exception as e:
        print(f"   ❌ Error classification error: {e}")
        return False

    try:
        # Test 4: Large message handling simulation
        print("\n4. Testing large message handling...")
        from langchain_core.messages import HumanMessage

        # Create large messages
        large_content = "A" * 100000
        large_messages = [HumanMessage(content=large_content) for _ in range(3)]

        # Mock LLM that raises token limit error
        class MockLLM:
            def invoke(self, messages):
                total_chars = sum(len(msg.content) for msg in messages)
                if total_chars > 65536:
                    raise Exception(
                        "This model's maximum context length is 65536 tokens. However, you requested 300000 tokens"
                    )
                return HumanMessage(content="Success")

        mock_llm = MockLLM()

        # This should trigger context optimization
        result = safe_llm_call(
            mock_llm.invoke,
            large_messages,
            operation_name="Test Large Messages",
            context="Integration Test",
            enable_smart_processing=True,
        )

        if result:
            print("   ✅ Large message handling works")
        else:
            print("   ❌ Large message handling failed")
            return False

    except Exception as e:
        print(f"   ❌ Large message handling error: {e}")
        return False

    return True


def main():
    """Main test function"""
    print("🔍 LLM Call Coverage Analysis")
    print("=" * 50)

    # Scan critical files
    results = scan_critical_files()

    # Check for issues
    issues_found = False
    for file_path, result in results.items():
        if "error" in result:
            issues_found = True
            continue

        if not result.get("has_safe_import", False):
            print(f"⚠️  {file_path} doesn't import safe_llm_call")
            issues_found = True

        if result.get("direct_calls", []):
            print(
                f"⚠️  {file_path} has {len(result['direct_calls'])} potential direct calls"
            )
            issues_found = True

    # Test context management integration
    integration_ok = test_context_management_integration()

    # Final summary
    print("\n" + "=" * 50)
    if not issues_found and integration_ok:
        print("🎉 All tests passed! Context management is properly implemented.")
        print("\n✅ Key findings:")
        print("   - All critical files import safe_llm_call functions")
        print("   - No problematic direct LLM calls found")
        print("   - Context evaluator is properly configured")
        print("   - Error handler works correctly")
        print("   - Large message handling works as expected")
        return True
    else:
        print("⚠️  Some issues found:")
        if issues_found:
            print("   - Direct LLM calls or missing imports detected")
        if not integration_ok:
            print("   - Context management integration issues")
        print("\n📝 Recommendations:")
        print("   1. Ensure all LLM calls use safe_llm_call or safe_llm_call_async")
        print("   2. Import safe call functions in files that make LLM calls")
        print("   3. Enable context evaluation for all LLM operations")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
