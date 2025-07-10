"""Unified token management utilities for consistent token handling across modules."""

from typing import Optional
from dataclasses import dataclass

from .content_processor import ContentProcessor
from .structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenValidationResult:
    """Result of token validation operation."""
    is_valid: bool
    actual_tokens: int
    max_tokens: int
    overflow_tokens: int = 0
    truncated_content: Optional[str] = None
    
    @property
    def overflow_ratio(self) -> float:
        """Calculate overflow ratio."""
        if self.max_tokens <= 0:
            return 0.0
        return max(0.0, self.overflow_tokens / self.max_tokens)


class TokenManager:
    """Unified token management for consistent token handling across modules."""
    
    # Token estimation constants
    CHAR_TO_TOKEN_RATIO_CONSERVATIVE = 3.0  # Conservative estimate: 1 token ≈ 3 chars
    CHAR_TO_TOKEN_RATIO_ROUGH = 4.0  # Rough estimate: 1 token ≈ 4 chars
    TOKEN_SAFETY_MULTIPLIER = 0.9  # Safety margin for token calculations
    TRUNCATION_MARKER = "...[truncated]"
    MAX_BINARY_SEARCH_ITERATIONS = 20
    MIN_VIABLE_CONTENT_TOKENS = 50
    
    def __init__(self, content_processor: Optional[ContentProcessor] = None):
        """Initialize TokenManager.
        
        Args:
            content_processor: ContentProcessor instance for accurate token counting
        """
        self.content_processor = content_processor or ContentProcessor()
        
    def count_tokens_precise(self, content: str, model_name: str) -> int:
        """Count tokens precisely using the content processor.
        
        Args:
            content: Text content to count tokens for
            model_name: Model name for accurate tokenization
            
        Returns:
            Precise token count
        """
        if not content:
            return 0
            
        try:
            result = self.content_processor.count_tokens_accurate(content, model_name)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Failed to count tokens precisely: {e}, using estimation")
            return self.estimate_tokens_conservative(content)
    
    def estimate_tokens_conservative(self, content: str) -> int:
        """Estimate tokens using conservative character-to-token ratio.
        
        Args:
            content: Text content to estimate tokens for
            
        Returns:
            Conservative token estimate
        """
        if not content:
            return 0
        return max(1, int(len(content) / self.CHAR_TO_TOKEN_RATIO_CONSERVATIVE))
    
    def estimate_tokens_rough(self, content: str) -> int:
        """Estimate tokens using rough character-to-token ratio.
        
        Args:
            content: Text content to estimate tokens for
            
        Returns:
            Rough token estimate
        """
        if not content:
            return 0
        return max(1, int(len(content) / self.CHAR_TO_TOKEN_RATIO_ROUGH))
    
    def validate_token_budget(self, content: str, max_tokens: int, model_name: str) -> TokenValidationResult:
        """Validate if content fits within token budget.
        
        Args:
            content: Content to validate
            max_tokens: Maximum allowed tokens
            model_name: Model name for accurate tokenization
            
        Returns:
            TokenValidationResult with validation details
        """
        if not content:
            return TokenValidationResult(
                is_valid=True,
                actual_tokens=0,
                max_tokens=max_tokens
            )
        
        actual_tokens = self.count_tokens_precise(content, model_name)
        is_valid = actual_tokens <= max_tokens
        overflow_tokens = max(0, actual_tokens - max_tokens)
        
        return TokenValidationResult(
            is_valid=is_valid,
            actual_tokens=actual_tokens,
            max_tokens=max_tokens,
            overflow_tokens=overflow_tokens
        )
    
    def truncate_to_token_limit(self, content: str, max_tokens: int, model_name: str, 
                               preserve_start: bool = True, preserve_end: bool = True) -> str:
        """Truncate content to fit within token limit using binary search.
        
        Args:
            content: Content to truncate
            max_tokens: Maximum token limit
            model_name: Model name for accurate tokenization
            preserve_start: Whether to preserve content from the start
            preserve_end: Whether to preserve content from the end
            
        Returns:
            Truncated content that fits within token limit
        """
        if not content:
            return content
            
        # Check if content already fits
        validation = self.validate_token_budget(content, max_tokens, model_name)
        if validation.is_valid:
            return content
            
        # Reserve tokens for truncation marker
        marker_tokens = self.count_tokens_precise(self.TRUNCATION_MARKER, model_name)
        effective_limit = max(max_tokens - marker_tokens, max_tokens // 2)
        
        if effective_limit < self.MIN_VIABLE_CONTENT_TOKENS:
            logger.warning(f"Token limit too small for meaningful content: {max_tokens}")
            return self.TRUNCATION_MARKER
        
        # Apply truncation strategy based on preservation preferences
        if preserve_start and preserve_end:
            return self._truncate_preserve_both_ends(content, effective_limit, model_name)
        elif preserve_start:
            return self._truncate_preserve_start(content, effective_limit, model_name)
        elif preserve_end:
            return self._truncate_preserve_end(content, effective_limit, model_name)
        else:
            return self._truncate_middle(content, effective_limit, model_name)
    
    def _truncate_preserve_start(self, content: str, max_tokens: int, model_name: str) -> str:
        """Truncate content preserving the start."""
        left, right = 0, len(content)
        best_content = ""
        iterations = 0
        
        while left <= right and iterations < self.MAX_BINARY_SEARCH_ITERATIONS:
            iterations += 1
            mid = (left + right) // 2
            
            if mid == 0:
                test_content = self.TRUNCATION_MARKER
            else:
                test_content = content[:mid] + self.TRUNCATION_MARKER
                
            test_tokens = self.count_tokens_precise(test_content, model_name)
            
            if test_tokens <= max_tokens:
                best_content = test_content
                left = mid + 1
            else:
                right = mid - 1
                
        return best_content or self._fallback_truncation(content, max_tokens)
    
    def _truncate_preserve_end(self, content: str, max_tokens: int, model_name: str) -> str:
        """Truncate content preserving the end."""
        left, right = 0, len(content)
        best_content = ""
        iterations = 0
        
        while left <= right and iterations < self.MAX_BINARY_SEARCH_ITERATIONS:
            iterations += 1
            mid = (left + right) // 2
            
            if mid >= len(content):
                test_content = self.TRUNCATION_MARKER + content
            else:
                test_content = self.TRUNCATION_MARKER + content[mid:]
                
            test_tokens = self.count_tokens_precise(test_content, model_name)
            
            if test_tokens <= max_tokens:
                best_content = test_content
                right = mid - 1
            else:
                left = mid + 1
                
        return best_content or self._fallback_truncation(content, max_tokens)
    
    def _truncate_preserve_both_ends(self, content: str, max_tokens: int, model_name: str) -> str:
        """Truncate content preserving both start and end."""
        # Try to preserve equal portions from start and end
        content_length = len(content)
        best_content = ""
        
        for start_ratio in [0.4, 0.3, 0.5, 0.2, 0.6]:  # Try different ratios
            start_chars = int(content_length * start_ratio)
            end_chars = int(content_length * (1 - start_ratio))
            
            if start_chars + end_chars >= content_length:
                continue
                
            start_part = content[:start_chars]
            end_part = content[-end_chars:] if end_chars > 0 else ""
            test_content = start_part + self.TRUNCATION_MARKER + end_part
            
            test_tokens = self.count_tokens_precise(test_content, model_name)
            
            if test_tokens <= max_tokens:
                best_content = test_content
                break
                
        return best_content or self._truncate_preserve_start(content, max_tokens, model_name)
    
    def _truncate_middle(self, content: str, max_tokens: int, model_name: str) -> str:
        """Truncate content from the middle."""
        content_length = len(content)
        quarter = content_length // 4
        
        # Try preserving first and last quarters
        start_part = content[:quarter]
        end_part = content[-quarter:]
        test_content = start_part + self.TRUNCATION_MARKER + end_part
        
        test_tokens = self.count_tokens_precise(test_content, model_name)
        
        if test_tokens <= max_tokens:
            return test_content
        else:
            # Fallback to start preservation
            return self._truncate_preserve_start(content, max_tokens, model_name)
    
    def _fallback_truncation(self, content: str, max_tokens: int) -> str:
        """Fallback truncation using character estimation."""
        char_limit = int(max_tokens * self.CHAR_TO_TOKEN_RATIO_CONSERVATIVE * self.TOKEN_SAFETY_MULTIPLIER)
        if char_limit < len(self.TRUNCATION_MARKER):
            return self.TRUNCATION_MARKER
        
        truncated = content[:char_limit - len(self.TRUNCATION_MARKER)]
        return truncated + self.TRUNCATION_MARKER
    
    def calculate_safe_token_limit(self, max_tokens: int, safety_margin: float = 0.1) -> int:
        """Calculate safe token limit with safety margin.
        
        Args:
            max_tokens: Maximum token limit
            safety_margin: Safety margin ratio (0.1 = 10% margin)
            
        Returns:
            Safe token limit
        """
        return max(1, int(max_tokens * (1 - safety_margin)))
    
    def split_content_by_tokens(self, content: str, chunk_size: int, model_name: str, 
                               overlap_tokens: int = 0) -> list[str]:
        """Split content into chunks based on token limits.
        
        Args:
            content: Content to split
            chunk_size: Maximum tokens per chunk
            model_name: Model name for accurate tokenization
            overlap_tokens: Number of tokens to overlap between chunks
            
        Returns:
            List of content chunks
        """
        if not content:
            return []
            
        # Check if content fits in single chunk
        total_tokens = self.count_tokens_precise(content, model_name)
        if total_tokens <= chunk_size:
            return [content]
        
        chunks = []
        sentences = content.split('. ')
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_with_period = sentence + '. ' if not sentence.endswith('.') else sentence + ' '
            sentence_tokens = self.count_tokens_precise(sentence_with_period, model_name)
            
            if current_tokens + sentence_tokens <= chunk_size:
                current_chunk += sentence_with_period
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Handle overlap
                    if overlap_tokens > 0 and chunks:
                        overlap_content = self._extract_overlap(current_chunk, overlap_tokens, model_name)
                        current_chunk = overlap_content + sentence_with_period
                        current_tokens = self.count_tokens_precise(current_chunk, model_name)
                    else:
                        current_chunk = sentence_with_period
                        current_tokens = sentence_tokens
                else:
                    # Single sentence exceeds chunk size, truncate it
                    truncated = self.truncate_to_token_limit(sentence_with_period, chunk_size, model_name)
                    chunks.append(truncated)
                    current_chunk = ""
                    current_tokens = 0
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _extract_overlap(self, content: str, overlap_tokens: int, model_name: str) -> str:
        """Extract overlap content from the end of a chunk."""
        sentences = content.split('. ')
        overlap_content = ""
        overlap_count = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_with_period = sentence + '. ' if not sentence.endswith('.') else sentence + ' '
            sentence_tokens = self.count_tokens_precise(sentence_with_period, model_name)
            
            if overlap_count + sentence_tokens <= overlap_tokens:
                overlap_content = sentence_with_period + overlap_content
                overlap_count += sentence_tokens
            else:
                break
                
        return overlap_content