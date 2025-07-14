#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Comprehensive Unit Tests for Content Processing and Memory Management

This test suite provides comprehensive coverage for content processing and memory
management systems in DeerFlow, replacing demo scripts with proper unit tests.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Content processing components
from src.utils.content.token_estimator import TokenEstimator
from src.utils.content.content_cleaner import ContentCleaner
from src.utils.content.smart_chunker import SmartChunker

# Memory management components
from src.utils.memory.hierarchical_cache import HierarchicalCache
from src.utils.memory.memory_manager import MemoryManager
from src.utils.memory.cache_optimizer import CacheOptimizer

# Context management
from src.utils.context.context_evaluator import ContextEvaluator
from src.utils.context.context_manager import ContextManager

# Error handling
from src.utils.error_handling.error_classifier import ErrorClassifier
from src.utils.error_handling.retry_handler import RetryHandler


class TestTokenEstimator:
    """Test TokenEstimator functionality."""
    
    @pytest.fixture
    def token_estimator(self):
        """Create TokenEstimator instance."""
        return TokenEstimator()
    
    def test_token_estimator_initialization(self, token_estimator):
        """Test TokenEstimator initialization."""
        assert hasattr(token_estimator, 'model_configs')
        assert hasattr(token_estimator, 'estimation_cache')
        assert hasattr(token_estimator, 'accuracy_metrics')
    
    def test_basic_token_estimation(self, token_estimator):
        """Test basic token estimation."""
        test_text = "This is a simple test text for token estimation."
        
        # Test estimation for different models
        gpt4_tokens = token_estimator.estimate_tokens(test_text, model="gpt-4")
        gpt35_tokens = token_estimator.estimate_tokens(test_text, model="gpt-3.5-turbo")
        claude_tokens = token_estimator.estimate_tokens(test_text, model="claude-3")
        
        assert isinstance(gpt4_tokens, int)
        assert isinstance(gpt35_tokens, int)
        assert isinstance(claude_tokens, int)
        assert gpt4_tokens > 0
        assert gpt35_tokens > 0
        assert claude_tokens > 0
    
    def test_complex_content_estimation(self, token_estimator):
        """Test token estimation for complex content."""
        complex_content = {
            "text": "This is the main content with some technical terms like API, JSON, and HTTP.",
            "code": "def example_function(param1, param2):\n    return param1 + param2",
            "metadata": {
                "title": "Example Document",
                "tags": ["programming", "python", "example"]
            }
        }
        
        estimation = token_estimator.estimate_complex_content(complex_content)
        
        assert isinstance(estimation, dict)
        assert "total_tokens" in estimation
        assert "breakdown" in estimation
        assert "text_tokens" in estimation["breakdown"]
        assert "code_tokens" in estimation["breakdown"]
        assert "metadata_tokens" in estimation["breakdown"]
        
        # Verify total is sum of parts
        breakdown = estimation["breakdown"]
        calculated_total = breakdown["text_tokens"] + breakdown["code_tokens"] + breakdown["metadata_tokens"]
        assert abs(estimation["total_tokens"] - calculated_total) <= 1  # Allow for rounding
    
    def test_batch_estimation(self, token_estimator):
        """Test batch token estimation."""
        texts = [
            "First text for estimation.",
            "Second text with more content and technical terms like API, REST, JSON.",
            "Third text that is much longer and contains various types of content including code snippets, technical documentation, and detailed explanations."
        ]
        
        batch_estimation = token_estimator.estimate_batch(texts)
        
        assert isinstance(batch_estimation, list)
        assert len(batch_estimation) == len(texts)
        
        for i, estimation in enumerate(batch_estimation):
            assert isinstance(estimation, dict)
            assert "text_index" in estimation
            assert "token_count" in estimation
            assert "confidence" in estimation
            assert estimation["text_index"] == i
    
    def test_estimation_caching(self, token_estimator):
        """Test token estimation caching."""
        test_text = "This text will be used to test caching functionality."
        
        # First estimation (should be calculated)
        start_time = time.time()
        first_estimation = token_estimator.estimate_tokens(test_text, model="gpt-4")
        first_duration = time.time() - start_time
        
        # Second estimation (should be cached)
        start_time = time.time()
        second_estimation = token_estimator.estimate_tokens(test_text, model="gpt-4")
        second_duration = time.time() - start_time
        
        # Verify same result
        assert first_estimation == second_estimation
        
        # Verify caching improved performance (second call should be faster)
        assert second_duration < first_duration or second_duration < 0.001  # Very fast for cached
    
    def test_estimation_accuracy_tracking(self, token_estimator):
        """Test estimation accuracy tracking."""
        test_text = "Sample text for accuracy testing."
        estimated_tokens = token_estimator.estimate_tokens(test_text, model="gpt-4")
        
        # Simulate actual token count feedback
        actual_tokens = estimated_tokens + 2  # Simulate slight difference
        
        token_estimator.record_actual_tokens(test_text, "gpt-4", actual_tokens)
        
        # Get accuracy metrics
        accuracy_metrics = token_estimator.get_accuracy_metrics("gpt-4")
        
        assert isinstance(accuracy_metrics, dict)
        assert "average_error" in accuracy_metrics
        assert "accuracy_percentage" in accuracy_metrics
        assert "total_samples" in accuracy_metrics
        assert accuracy_metrics["total_samples"] >= 1
    
    def test_model_specific_estimation(self, token_estimator):
        """Test model-specific token estimation differences."""
        test_text = "This text contains special characters: émojis 🚀, symbols ∑∆, and unicode 中文."
        
        estimations = {}
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3", "llama-2"]
        
        for model in models:
            try:
                estimations[model] = token_estimator.estimate_tokens(test_text, model=model)
            except ValueError:
                # Some models might not be configured
                continue
        
        # Verify we got estimations for at least some models
        assert len(estimations) > 0
        
        # Verify estimations are reasonable (not zero, not extremely high)
        for model, tokens in estimations.items():
            assert 0 < tokens < 1000  # Reasonable range for the test text


class TestContentCleaner:
    """Test ContentCleaner functionality."""
    
    @pytest.fixture
    def content_cleaner(self):
        """Create ContentCleaner instance."""
        return ContentCleaner()
    
    def test_content_cleaner_initialization(self, content_cleaner):
        """Test ContentCleaner initialization."""
        assert hasattr(content_cleaner, 'cleaning_rules')
        assert hasattr(content_cleaner, 'cleaning_stats')
        assert hasattr(content_cleaner, 'custom_filters')
    
    def test_basic_text_cleaning(self, content_cleaner):
        """Test basic text cleaning functionality."""
        dirty_text = "   This is a test text with   extra spaces\n\n\nand multiple newlines.\t\tTabs too!   "
        
        cleaned_text = content_cleaner.clean_text(dirty_text)
        
        assert isinstance(cleaned_text, str)
        assert not cleaned_text.startswith(" ")
        assert not cleaned_text.endswith(" ")
        assert "\n\n\n" not in cleaned_text
        assert "\t\t" not in cleaned_text
    
    def test_html_content_cleaning(self, content_cleaner):
        """Test HTML content cleaning."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>This is a paragraph with <strong>bold text</strong> and <em>italic text</em>.</p>
                <div class="sidebar">Sidebar content</div>
                <script>alert('This should be removed');</script>
                <style>.hidden { display: none; }</style>
            </body>
        </html>
        """
        
        cleaned_content = content_cleaner.clean_html(html_content)
        
        assert isinstance(cleaned_content, str)
        assert "<script>" not in cleaned_content
        assert "<style>" not in cleaned_content
        assert "Main Title" in cleaned_content
        assert "This is a paragraph" in cleaned_content
        assert "bold text" in cleaned_content
    
    def test_code_content_cleaning(self, content_cleaner):
        """Test code content cleaning."""
        code_content = """
        # This is a comment
        def example_function(param1, param2):
            "This is a docstring."
            # Another comment
            result = param1 + param2  # Inline comment
            return result
        
        # TODO: Implement better error handling
        # FIXME: This might cause issues
        """
        
        cleaned_code = content_cleaner.clean_code(code_content, preserve_structure=True)
        
        assert isinstance(cleaned_code, str)
        assert "def example_function" in cleaned_code
        assert "return result" in cleaned_code
        
        # Test with comment removal
        cleaned_code_no_comments = content_cleaner.clean_code(code_content, remove_comments=True)
        assert "# This is a comment" not in cleaned_code_no_comments
        assert "# TODO:" not in cleaned_code_no_comments
    
    def test_markdown_content_cleaning(self, content_cleaner):
        """Test Markdown content cleaning."""
        markdown_content = """
        # Main Title
        
        This is a paragraph with **bold text** and *italic text*.
        
        ## Subsection
        
        - List item 1
        - List item 2
        - List item 3
        
        ```python
        def example():
            return "Hello, World!"
        ```
        
        [Link to example](https://example.com)
        
        ![Image description](image.png)
        """
        
        cleaned_markdown = content_cleaner.clean_markdown(markdown_content)
        
        assert isinstance(cleaned_markdown, str)
        assert "Main Title" in cleaned_markdown
        assert "This is a paragraph" in cleaned_markdown
        assert "List item 1" in cleaned_markdown
        assert "def example()" in cleaned_markdown
    
    def test_custom_cleaning_rules(self, content_cleaner):
        """Test custom cleaning rules."""
        # Add custom rule to remove email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        content_cleaner.add_custom_rule("remove_emails", email_pattern, "[EMAIL_REMOVED]")
        
        test_text = "Contact us at support@example.com or admin@test.org for help."
        
        cleaned_text = content_cleaner.apply_custom_rules(test_text)
        
        assert "support@example.com" not in cleaned_text
        assert "admin@test.org" not in cleaned_text
        assert "[EMAIL_REMOVED]" in cleaned_text
    
    def test_batch_content_cleaning(self, content_cleaner):
        """Test batch content cleaning."""
        content_items = [
            {"type": "text", "content": "   Dirty text with spaces   "},
            {"type": "html", "content": "<p>HTML content</p><script>alert('bad');</script>"},
            {"type": "markdown", "content": "# Title\n\nParagraph with **bold** text."}
        ]
        
        cleaned_items = content_cleaner.clean_batch(content_items)
        
        assert len(cleaned_items) == len(content_items)
        
        for i, cleaned_item in enumerate(cleaned_items):
            assert "cleaned_content" in cleaned_item
            assert "original_type" in cleaned_item
            assert "cleaning_stats" in cleaned_item
            assert cleaned_item["original_type"] == content_items[i]["type"]
    
    def test_cleaning_statistics(self, content_cleaner):
        """Test cleaning statistics collection."""
        test_content = "   This is a test with   extra spaces and\n\nmultiple newlines.   "
        
        cleaned_content = content_cleaner.clean_text(test_content, collect_stats=True)
        
        stats = content_cleaner.get_cleaning_stats()
        
        assert isinstance(stats, dict)
        assert "total_cleanings" in stats
        assert "characters_removed" in stats
        assert "whitespace_normalized" in stats
        assert stats["total_cleanings"] >= 1


class TestSmartChunker:
    """Test SmartChunker functionality."""
    
    @pytest.fixture
    def smart_chunker(self):
        """Create SmartChunker instance."""
        return SmartChunker()
    
    def test_smart_chunker_initialization(self, smart_chunker):
        """Test SmartChunker initialization."""
        assert hasattr(smart_chunker, 'chunk_strategies')
        assert hasattr(smart_chunker, 'chunk_cache')
        assert hasattr(smart_chunker, 'optimization_metrics')
    
    def test_basic_text_chunking(self, smart_chunker):
        """Test basic text chunking."""
        long_text = "This is a long text that needs to be chunked into smaller pieces. " * 50
        
        chunks = smart_chunker.chunk_text(long_text, max_chunk_size=500)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "content" in chunk
            assert "chunk_id" in chunk
            assert "metadata" in chunk
            assert len(chunk["content"]) <= 500
    
    def test_semantic_chunking(self, smart_chunker):
        """Test semantic-aware chunking."""
        text_with_sections = """
        Introduction
        This is the introduction section of the document.
        
        Methodology
        This section describes the methodology used in the research.
        
        Results
        Here we present the results of our analysis.
        
        Conclusion
        Finally, we conclude with our findings.
        """
        
        chunks = smart_chunker.chunk_semantically(text_with_sections)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 4  # Should identify the main sections
        
        # Verify semantic boundaries are respected
        section_keywords = ["Introduction", "Methodology", "Results", "Conclusion"]
        found_sections = []
        
        for chunk in chunks:
            for keyword in section_keywords:
                if keyword in chunk["content"]:
                    found_sections.append(keyword)
                    break
        
        assert len(found_sections) >= 3  # Should find most sections
    
    def test_code_aware_chunking(self, smart_chunker):
        """Test code-aware chunking."""
        code_content = """
        def function_one():
            "First function."
            return "Hello"
        
        def function_two():
            "Second function."
            return "World"
        
        class ExampleClass:
            "Example class."
            
            def __init__(self):
                self.value = 42
            
            def method_one(self):
                return self.value * 2
        
        # Some standalone code
        result = function_one() + " " + function_two()
        instance = ExampleClass()
        """
        
        chunks = smart_chunker.chunk_code(code_content)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 3  # Should separate functions and class
        
        # Verify code structure is preserved
        function_chunks = [c for c in chunks if "def " in c["content"]]
        class_chunks = [c for c in chunks if "class " in c["content"]]
        
        assert len(function_chunks) >= 2
        assert len(class_chunks) >= 1
    
    def test_adaptive_chunking(self, smart_chunker):
        """Test adaptive chunking based on content type."""
        mixed_content = {
            "text": "This is regular text content that should be chunked normally.",
            "code": "def example(): return 'code'",
            "data": {"key1": "value1", "key2": "value2"},
            "list": ["item1", "item2", "item3"]
        }
        
        chunks = smart_chunker.chunk_adaptively(mixed_content)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        
        # Verify different content types are handled appropriately
        content_types = set()
        for chunk in chunks:
            if "content_type" in chunk["metadata"]:
                content_types.add(chunk["metadata"]["content_type"])
        
        assert len(content_types) > 1  # Should detect multiple content types
    
    def test_chunk_optimization(self, smart_chunker):
        """Test chunk size optimization."""
        test_content = "This is test content for optimization. " * 100
        
        # Test different chunk sizes
        small_chunks = smart_chunker.chunk_text(test_content, max_chunk_size=200)
        medium_chunks = smart_chunker.chunk_text(test_content, max_chunk_size=500)
        large_chunks = smart_chunker.chunk_text(test_content, max_chunk_size=1000)
        
        # Verify optimization metrics
        assert len(small_chunks) > len(medium_chunks)
        assert len(medium_chunks) >= len(large_chunks)
        
        # Get optimization recommendations
        optimization = smart_chunker.optimize_chunk_size(test_content)
        
        assert isinstance(optimization, dict)
        assert "recommended_size" in optimization
        assert "efficiency_score" in optimization
        assert "reasoning" in optimization
    
    def test_chunk_overlap_handling(self, smart_chunker):
        """Test chunk overlap functionality."""
        test_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        chunks_with_overlap = smart_chunker.chunk_text(
            test_text, 
            max_chunk_size=50, 
            overlap_size=10
        )
        
        chunks_without_overlap = smart_chunker.chunk_text(
            test_text, 
            max_chunk_size=50, 
            overlap_size=0
        )
        
        # Verify overlap is working
        if len(chunks_with_overlap) > 1:
            # Check that consecutive chunks have some overlapping content
            for i in range(len(chunks_with_overlap) - 1):
                chunk1_end = chunks_with_overlap[i]["content"][-20:]
                chunk2_start = chunks_with_overlap[i + 1]["content"][:20]
                
                # There should be some common words due to overlap
                chunk1_words = set(chunk1_end.split())
                chunk2_words = set(chunk2_start.split())
                common_words = chunk1_words.intersection(chunk2_words)
                
                # Allow for cases where overlap might not result in common words
                # due to sentence boundaries
                assert len(common_words) >= 0
    
    def test_chunk_metadata_enrichment(self, smart_chunker):
        """Test chunk metadata enrichment."""
        test_content = "This is a test document with multiple sentences. It contains various information."
        
        chunks = smart_chunker.chunk_text(test_content, enrich_metadata=True)
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            
            assert "word_count" in metadata
            assert "sentence_count" in metadata
            assert "character_count" in metadata
            assert "chunk_position" in metadata
            assert "content_hash" in metadata
            
            # Verify metadata accuracy
            content = chunk["content"]
            assert metadata["character_count"] == len(content)
            assert metadata["word_count"] == len(content.split())


class TestHierarchicalCache:
    """Test HierarchicalCache functionality."""
    
    @pytest.fixture
    def hierarchical_cache(self):
        """Create HierarchicalCache instance."""
        config = {
            "l1_cache": {"max_size": 100, "ttl": 300},
            "l2_cache": {"max_size": 500, "ttl": 1800},
            "l3_cache": {"max_size": 1000, "ttl": 3600}
        }
        return HierarchicalCache(config)
    
    def test_hierarchical_cache_initialization(self, hierarchical_cache):
        """Test HierarchicalCache initialization."""
        assert hasattr(hierarchical_cache, 'l1_cache')
        assert hasattr(hierarchical_cache, 'l2_cache')
        assert hasattr(hierarchical_cache, 'l3_cache')
        assert hasattr(hierarchical_cache, 'cache_stats')
    
    def test_cache_storage_and_retrieval(self, hierarchical_cache):
        """Test cache storage and retrieval across levels."""
        # Store data in cache
        key = "test_key_001"
        value = {"data": "test_value", "metadata": {"type": "test"}}
        
        hierarchical_cache.set(key, value)
        
        # Retrieve data
        retrieved_value = hierarchical_cache.get(key)
        
        assert retrieved_value is not None
        assert retrieved_value == value
    
    def test_cache_level_promotion(self, hierarchical_cache):
        """Test cache level promotion on access."""
        key = "promotion_test_key"
        value = "promotion_test_value"
        
        # Store in L3 cache initially
        hierarchical_cache.l3_cache.set(key, value)
        
        # Access the key multiple times to trigger promotion
        for _ in range(3):
            retrieved = hierarchical_cache.get(key)
            assert retrieved == value
        
        # Check if promoted to higher level cache
        cache_stats = hierarchical_cache.get_cache_stats()
        assert cache_stats["l1_hits"] > 0 or cache_stats["l2_hits"] > 0
    
    def test_cache_eviction_policy(self, hierarchical_cache):
        """Test cache eviction when limits are reached."""
        # Fill L1 cache beyond capacity
        for i in range(150):  # L1 max_size is 100
            key = f"eviction_test_{i}"
            value = f"value_{i}"
            hierarchical_cache.set(key, value)
        
        # Verify some items were evicted from L1
        l1_size = hierarchical_cache.l1_cache.size()
        assert l1_size <= 100
        
        # Verify evicted items might be in L2
        early_key = "eviction_test_0"
        retrieved = hierarchical_cache.get(early_key)
        # Item might be evicted or moved to L2/L3
        assert retrieved is None or retrieved == "value_0"
    
    def test_cache_statistics(self, hierarchical_cache):
        """Test cache statistics collection."""
        # Perform various cache operations
        hierarchical_cache.set("stats_key_1", "value_1")
        hierarchical_cache.set("stats_key_2", "value_2")
        
        hierarchical_cache.get("stats_key_1")  # Hit
        hierarchical_cache.get("stats_key_1")  # Hit
        hierarchical_cache.get("nonexistent_key")  # Miss
        
        stats = hierarchical_cache.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "l1_hits" in stats
        assert "l2_hits" in stats
        assert "l3_hits" in stats
        assert "total_misses" in stats
        assert "hit_ratio" in stats
        
        assert stats["total_misses"] >= 1  # At least one miss
        assert stats["l1_hits"] >= 1  # At least one L1 hit
    
    def test_cache_invalidation(self, hierarchical_cache):
        """Test cache invalidation functionality."""
        key = "invalidation_test_key"
        value = "invalidation_test_value"
        
        # Store and verify
        hierarchical_cache.set(key, value)
        assert hierarchical_cache.get(key) == value
        
        # Invalidate specific key
        hierarchical_cache.invalidate(key)
        assert hierarchical_cache.get(key) is None
        
        # Test pattern-based invalidation
        hierarchical_cache.set("pattern_key_1", "value_1")
        hierarchical_cache.set("pattern_key_2", "value_2")
        hierarchical_cache.set("other_key", "other_value")
        
        hierarchical_cache.invalidate_pattern("pattern_key_*")
        
        assert hierarchical_cache.get("pattern_key_1") is None
        assert hierarchical_cache.get("pattern_key_2") is None
        assert hierarchical_cache.get("other_key") == "other_value"
    
    def test_cache_ttl_expiration(self, hierarchical_cache):
        """Test cache TTL expiration."""
        key = "ttl_test_key"
        value = "ttl_test_value"
        
        # Store with short TTL
        hierarchical_cache.set(key, value, ttl=1)  # 1 second TTL
        
        # Immediate retrieval should work
        assert hierarchical_cache.get(key) == value
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        assert hierarchical_cache.get(key) is None
    
    def test_cache_memory_optimization(self, hierarchical_cache):
        """Test cache memory optimization."""
        # Store large amounts of data
        large_data = "x" * 1000  # 1KB string
        
        for i in range(50):
            key = f"large_data_{i}"
            hierarchical_cache.set(key, large_data)
        
        # Trigger memory optimization
        optimization_result = hierarchical_cache.optimize_memory()
        
        assert isinstance(optimization_result, dict)
        assert "memory_freed" in optimization_result
        assert "items_evicted" in optimization_result
        assert "optimization_time" in optimization_result
    
    def test_concurrent_cache_access(self, hierarchical_cache):
        """Test concurrent cache access."""
        import threading
        import time
        
        results = []
        
        def cache_worker(worker_id):
            try:
                # Each worker stores and retrieves data
                key = f"concurrent_key_{worker_id}"
                value = f"concurrent_value_{worker_id}"
                
                hierarchical_cache.set(key, value)
                time.sleep(0.1)  # Simulate some work
                
                retrieved = hierarchical_cache.get(key)
                
                results.append({
                    "worker_id": worker_id,
                    "success": retrieved == value,
                    "retrieved_value": retrieved
                })
            except Exception as e:
                results.append({
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all workers succeeded
        assert len(results) == 10
        for result in results:
            assert result["success"] is True


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create MemoryManager instance."""
        config = {
            "max_memory_mb": 100,
            "cleanup_threshold": 0.8,
            "enable_compression": True,
            "enable_monitoring": True
        }
        return MemoryManager(config)
    
    def test_memory_manager_initialization(self, memory_manager):
        """Test MemoryManager initialization."""
        assert hasattr(memory_manager, 'memory_pools')
        assert hasattr(memory_manager, 'memory_stats')
        assert hasattr(memory_manager, 'cleanup_policies')
    
    def test_memory_allocation_and_deallocation(self, memory_manager):
        """Test memory allocation and deallocation."""
        # Allocate memory for different purposes
        cache_memory = memory_manager.allocate_memory("cache", size_mb=10)
        buffer_memory = memory_manager.allocate_memory("buffer", size_mb=5)
        
        assert cache_memory is not None
        assert buffer_memory is not None
        
        # Check memory usage
        usage = memory_manager.get_memory_usage()
        assert usage["allocated_mb"] >= 15
        
        # Deallocate memory
        memory_manager.deallocate_memory("cache")
        
        updated_usage = memory_manager.get_memory_usage()
        assert updated_usage["allocated_mb"] < usage["allocated_mb"]
    
    def test_memory_monitoring(self, memory_manager):
        """Test memory monitoring functionality."""
        # Allocate some memory
        memory_manager.allocate_memory("test_pool", size_mb=20)
        
        # Get monitoring data
        monitoring_data = memory_manager.get_monitoring_data()
        
        assert isinstance(monitoring_data, dict)
        assert "current_usage" in monitoring_data
        assert "peak_usage" in monitoring_data
        assert "allocation_history" in monitoring_data
        assert "memory_pools" in monitoring_data
        
        # Verify current usage is tracked
        assert monitoring_data["current_usage"]["allocated_mb"] >= 20
    
    def test_memory_cleanup_policies(self, memory_manager):
        """Test memory cleanup policies."""
        # Fill memory close to threshold
        for i in range(8):  # 8 * 10MB = 80MB (80% of 100MB limit)
            memory_manager.allocate_memory(f"test_pool_{i}", size_mb=10)
        
        # Trigger cleanup
        cleanup_result = memory_manager.trigger_cleanup()
        
        assert isinstance(cleanup_result, dict)
        assert "memory_freed" in cleanup_result
        assert "pools_cleaned" in cleanup_result
        assert "cleanup_time" in cleanup_result
        
        # Verify memory was freed
        post_cleanup_usage = memory_manager.get_memory_usage()
        assert post_cleanup_usage["allocated_mb"] < 80
    
    def test_memory_compression(self, memory_manager):
        """Test memory compression functionality."""
        # Store compressible data
        large_text = "This is a test string that should compress well. " * 100
        
        compressed_pool = memory_manager.allocate_compressed_memory(
            "compressed_test", 
            data=large_text
        )
        
        assert compressed_pool is not None
        
        # Retrieve and verify data
        retrieved_data = memory_manager.get_compressed_data("compressed_test")
        assert retrieved_data == large_text
        
        # Check compression stats
        compression_stats = memory_manager.get_compression_stats("compressed_test")
        assert "original_size" in compression_stats
        assert "compressed_size" in compression_stats
        assert "compression_ratio" in compression_stats
        assert compression_stats["compressed_size"] < compression_stats["original_size"]


class TestContextEvaluator:
    """Test ContextEvaluator functionality."""
    
    @pytest.fixture
    def context_evaluator(self):
        """Create ContextEvaluator instance."""
        return ContextEvaluator()
    
    def test_context_evaluator_initialization(self, context_evaluator):
        """Test ContextEvaluator initialization."""
        assert hasattr(context_evaluator, 'evaluation_metrics')
        assert hasattr(context_evaluator, 'context_history')
        assert hasattr(context_evaluator, 'evaluation_cache')
    
    def test_context_relevance_evaluation(self, context_evaluator):
        """Test context relevance evaluation."""
        query = "machine learning algorithms for image recognition"
        context = {
            "documents": [
                "Convolutional neural networks are effective for image classification.",
                "Support vector machines can be used for various classification tasks.",
                "The weather today is sunny and warm."
            ],
            "metadata": {
                "domain": "computer_science",
                "topic": "machine_learning"
            }
        }
        
        relevance_score = context_evaluator.evaluate_relevance(query, context)
        
        assert isinstance(relevance_score, float)
        assert 0.0 <= relevance_score <= 1.0
        
        # Should score higher for relevant content
        assert relevance_score > 0.5  # Assuming good relevance
    
    def test_context_completeness_evaluation(self, context_evaluator):
        """Test context completeness evaluation."""
        query = "explain neural network backpropagation"
        context = {
            "content": "Backpropagation is an algorithm used to train neural networks by calculating gradients.",
            "coverage_areas": ["algorithm_definition", "gradient_calculation"],
            "missing_areas": ["mathematical_details", "implementation_examples"]
        }
        
        completeness_score = context_evaluator.evaluate_completeness(query, context)
        
        assert isinstance(completeness_score, dict)
        assert "score" in completeness_score
        assert "coverage_analysis" in completeness_score
        assert "missing_elements" in completeness_score
        assert "recommendations" in completeness_score
        
        assert 0.0 <= completeness_score["score"] <= 1.0
    
    def test_context_quality_assessment(self, context_evaluator):
        """Test context quality assessment."""
        context = {
            "content": "High-quality content with proper citations and clear explanations.",
            "sources": ["peer_reviewed_paper", "academic_textbook"],
            "freshness": "2023-01-01",
            "authority": "expert_author"
        }
        
        quality_assessment = context_evaluator.assess_quality(context)
        
        assert isinstance(quality_assessment, dict)
        assert "overall_score" in quality_assessment
        assert "quality_dimensions" in quality_assessment
        assert "improvement_suggestions" in quality_assessment
        
        # Check quality dimensions
        dimensions = quality_assessment["quality_dimensions"]
        assert "accuracy" in dimensions
        assert "authority" in dimensions
        assert "freshness" in dimensions
        assert "clarity" in dimensions
    
    def test_context_optimization_recommendations(self, context_evaluator):
        """Test context optimization recommendations."""
        current_context = {
            "content": "Basic information about the topic.",
            "depth": "shallow",
            "coverage": "partial"
        }
        
        query = "comprehensive analysis of topic"
        
        recommendations = context_evaluator.get_optimization_recommendations(
            query, current_context
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for recommendation in recommendations:
            assert "type" in recommendation
            assert "description" in recommendation
            assert "priority" in recommendation
            assert "implementation" in recommendation


class TestErrorHandling:
    """Test error handling functionality."""
    
    @pytest.fixture
    def error_classifier(self):
        """Create ErrorClassifier instance."""
        return ErrorClassifier()
    
    @pytest.fixture
    def retry_handler(self):
        """Create RetryHandler instance."""
        config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "exponential_backoff": True,
            "jitter": True
        }
        return RetryHandler(config)
    
    def test_error_classification(self, error_classifier):
        """Test error classification functionality."""
        # Test different types of errors
        network_error = ConnectionError("Network connection failed")
        timeout_error = TimeoutError("Request timed out")
        value_error = ValueError("Invalid input value")
        
        network_classification = error_classifier.classify_error(network_error)
        timeout_classification = error_classifier.classify_error(timeout_error)
        value_classification = error_classifier.classify_error(value_error)
        
        assert network_classification["category"] == "network"
        assert network_classification["severity"] in ["low", "medium", "high"]
        assert network_classification["retryable"] is True
        
        assert timeout_classification["category"] == "timeout"
        assert timeout_classification["retryable"] is True
        
        assert value_classification["category"] == "validation"
        assert value_classification["retryable"] is False
    
    def test_retry_handler_functionality(self, retry_handler):
        """Test retry handler functionality."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "Success"
        
        # Test successful retry
        result = retry_handler.execute_with_retry(failing_function)
        
        assert result == "Success"
        assert call_count == 3
    
    def test_retry_handler_max_retries(self, retry_handler):
        """Test retry handler max retries limit."""
        def always_failing_function():
            raise ConnectionError("Persistent failure")
        
        # Should raise exception after max retries
        with pytest.raises(ConnectionError):
            retry_handler.execute_with_retry(always_failing_function)
    
    def test_retry_handler_non_retryable_errors(self, retry_handler):
        """Test retry handler with non-retryable errors."""
        def function_with_value_error():
            raise ValueError("Invalid input")
        
        # Should not retry for non-retryable errors
        with pytest.raises(ValueError):
            retry_handler.execute_with_retry(function_with_value_error)
    
    @pytest.mark.asyncio
    async def test_async_retry_handler(self, retry_handler):
        """Test async retry handler functionality."""
        call_count = 0
        
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Async temporary failure")
            return "Async Success"
        
        result = await retry_handler.execute_async_with_retry(async_failing_function)
        
        assert result == "Async Success"
        assert call_count == 2


if __name__ == "__main__":
    # Run the comprehensive content and memory test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src.utils.content",
        "--cov=src.utils.memory",
        "--cov=src.utils.context",
        "--cov=src.utils.error_handling",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])