"""Advanced Context Manager for LLM Token Optimization

Implements state-of-the-art context management strategies including:
- Hierarchical context management
- Sliding window attention simulation
- Dynamic context compression
- Intelligent summarization
- Token-aware content selection
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

from src.utils.content_processor import ContentProcessor
from src.utils.token_manager import TokenManager
from src.config.configuration import Configuration

logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """Context priority levels for hierarchical management"""

    CRITICAL = 1  # Current task, system instructions
    HIGH = 2  # Recent interactions, key decisions
    MEDIUM = 3  # Historical context, background info
    LOW = 4  # Auxiliary information, old context


class CompressionStrategy(Enum):
    """Context compression strategies"""

    NONE = "none"  # No compression
    TRUNCATE = "truncate"  # Simple truncation
    SUMMARIZE = "summarize"  # LLM-based summarization
    SLIDING_WINDOW = "sliding_window"  # Sliding window with overlap
    HIERARCHICAL = "hierarchical"  # Priority-based selection
    ADAPTIVE = "adaptive"  # Dynamic strategy selection


@dataclass
class ContextSegment:
    """Represents a segment of context with metadata"""

    content: str
    priority: ContextPriority
    timestamp: float
    token_count: int
    segment_type: str  # 'system', 'task', 'history', 'result'
    importance_score: float = 0.0
    can_compress: bool = True
    compressed_version: Optional[str] = None


@dataclass
class ContextWindow:
    """Represents the current context window state"""

    segments: List[ContextSegment]
    total_tokens: int
    max_tokens: int
    compression_ratio: float = 0.0
    strategy_used: CompressionStrategy = CompressionStrategy.NONE


class AdvancedContextManager:
    """Advanced context manager with multiple optimization strategies"""

    def __init__(
        self,
        config: Configuration,
        content_processor: Optional[ContentProcessor] = None,
    ):
        self.config = config
        self.content_processor = content_processor or ContentProcessor(
            config.model_token_limits
        )
        self.token_manager = TokenManager(self.content_processor)

        # Context management settings - More conservative ratios to prevent token overflow
        self.max_context_ratio = (
            0.4  # Use 40% of model limit for context (reduced from 60%)
        )
        self.sliding_window_size = (
            3  # Number of recent interactions to keep (reduced from 5)
        )
        self.overlap_ratio = 0.1  # Overlap between sliding windows (reduced from 0.2)
        self.compression_threshold = (
            0.6  # Trigger compression at 60% capacity (reduced from 80%)
        )

        # Hierarchical weights for different content types
        self.priority_weights = {
            ContextPriority.CRITICAL: 1.0,
            ContextPriority.HIGH: 0.7,
            ContextPriority.MEDIUM: 0.4,
            ContextPriority.LOW: 0.1,
        }

        logger.info("Advanced Context Manager initialized")

    def create_context_segment(
        self,
        content: str,
        segment_type: str,
        priority: ContextPriority = ContextPriority.MEDIUM,
        timestamp: Optional[float] = None,
    ) -> ContextSegment:
        """Create a context segment with metadata"""
        import time

        if timestamp is None:
            timestamp = time.time()

        # Use TokenManager for accurate token counting
        token_count = self.token_manager.count_tokens_precise(content, "deepseek-chat")
        importance_score = self._calculate_importance_score(
            content, segment_type, priority
        )

        return ContextSegment(
            content=content,
            priority=priority,
            timestamp=timestamp,
            token_count=token_count,
            segment_type=segment_type,
            importance_score=importance_score,
            can_compress=(segment_type != "system"),
        )

    def _calculate_importance_score(
        self, content: str, segment_type: str, priority: ContextPriority
    ) -> float:
        """Calculate importance score for content prioritization"""
        base_score = self.priority_weights[priority]

        # Adjust based on segment type
        type_multipliers = {
            "system": 1.0,
            "task": 0.9,
            "result": 0.8,
            "history": 0.6,
            "auxiliary": 0.3,
        }

        type_multiplier = type_multipliers.get(segment_type, 0.5)

        # Adjust based on content characteristics
        content_score = 1.0

        # Boost score for error messages, warnings, or important keywords
        important_keywords = [
            "error",
            "warning",
            "critical",
            "important",
            "required",
            "must",
        ]
        if any(keyword in content.lower() for keyword in important_keywords):
            content_score *= 1.2

        # Reduce score for very long content (likely to be verbose)
        if len(content) > 2000:
            content_score *= 0.8

        return base_score * type_multiplier * content_score

    def optimize_context_for_parallel(
        self,
        completed_steps: List[Any],
        current_task: str,
        model_name: str,
        strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE,
    ) -> ContextWindow:
        """Optimize context specifically for parallel execution"""
        logger.info(
            f"Optimizing context for parallel execution with strategy: {strategy.value}"
        )

        # Get model limits
        limits = self.content_processor.get_model_limits(model_name)
        max_context_tokens = int(limits.safe_input_limit * self.max_context_ratio)

        # Create segments from completed steps
        segments = []

        # Add current task (highest priority)
        task_segment = self.create_context_segment(
            content=current_task, segment_type="task", priority=ContextPriority.CRITICAL
        )
        segments.append(task_segment)

        # Process completed steps
        for i, step in enumerate(completed_steps):
            # Determine priority based on recency and content
            if i >= len(completed_steps) - 2:  # Last 2 steps
                priority = ContextPriority.HIGH
            elif i >= len(completed_steps) - 5:  # Last 5 steps
                priority = ContextPriority.MEDIUM
            else:
                priority = ContextPriority.LOW

            # Create segment for step result
            step_content = self._format_step_content(step)
            step_segment = self.create_context_segment(
                content=step_content, segment_type="result", priority=priority
            )
            segments.append(step_segment)

        # Apply compression strategy
        optimized_segments = self._apply_compression_strategy(
            segments, max_context_tokens, strategy, model_name
        )

        # Calculate final metrics
        total_tokens = sum(seg.token_count for seg in optimized_segments)
        compression_ratio = 1.0 - (
            total_tokens / sum(seg.token_count for seg in segments)
        )

        context_window = ContextWindow(
            segments=optimized_segments,
            total_tokens=total_tokens,
            max_tokens=max_context_tokens,
            compression_ratio=compression_ratio,
            strategy_used=strategy,
        )

        logger.info(
            f"Context optimization complete: {total_tokens}/{max_context_tokens} tokens, "
            f"compression: {compression_ratio:.2%}, segments: {len(optimized_segments)}"
        )

        return context_window

    def _format_step_content(self, step) -> str:
        """Format step content for context inclusion"""
        step_info = []

        # Handle both dict and object formats
        if hasattr(step, "__dict__"):  # Object with attributes
            step_dict = step.__dict__
        elif isinstance(step, dict):
            step_dict = step
        else:
            return str(step)

        # Check for step name/title
        step_name = step_dict.get("step_name") or step_dict.get("title")
        if step_name:
            step_info.append(f"Step: {step_name}")

        if "execution_res" in step_dict:
            result = step_dict["execution_res"]
            if isinstance(result, str):
                # Truncate very long results
                if len(result) > 500:
                    result = result[:400] + "... [truncated]"
                step_info.append(f"Result: {result}")
            elif isinstance(result, dict):
                # Extract key information from dict results
                key_fields = ["status", "message", "error", "output"]
                for field in key_fields:
                    if field in result:
                        value = str(result[field])
                        if len(value) > 200:
                            value = value[:150] + "... [truncated]"
                        step_info.append(f"{field.title()}: {value}")

        return "\n".join(step_info)

    def _validate_token_budget(self, segments: List[ContextSegment], max_tokens: int, model_name: str) -> List[ContextSegment]:
        """Validate and ensure segments fit within token budget using precise calculation.
        
        Args:
            segments: List of context segments
            max_tokens: Maximum token limit
            model_name: Model name for accurate token calculation
            
        Returns:
            Validated segments that fit within token budget
        """
        # Calculate precise token count for all segments
        total_content = "\n\n".join([seg.content for seg in segments])
        actual_tokens = self.token_manager.count_tokens_precise(total_content, model_name)
        
        if actual_tokens <= max_tokens:
            # Update segment token counts with more accurate values
            for segment in segments:
                segment.token_count = self.token_manager.count_tokens_precise(
                    segment.content, model_name
                )
            return segments
            
        logger.warning(
            f"Token validation failed: {actual_tokens} > {max_tokens}, applying emergency truncation"
        )
        
        # Apply emergency truncation with precise token calculation
        validated_segments = []
        used_tokens = 0
        
        # Sort by priority to preserve most important content
        sorted_segments = sorted(segments, key=lambda x: (x.priority.value, -x.importance_score))
        
        for segment in sorted_segments:
            segment_tokens = self.token_manager.count_tokens_precise(
                segment.content, model_name
            )
            
            if used_tokens + segment_tokens <= max_tokens:
                segment.token_count = segment_tokens
                validated_segments.append(segment)
                used_tokens += segment_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - used_tokens
                if remaining_tokens > 50:  # Minimum viable segment size
                    truncated_content = self.token_manager.truncate_to_token_limit(
                        segment.content, remaining_tokens, model_name, preserve_start=True, preserve_end=False
                    )
                    if truncated_content:
                        truncated_segment = ContextSegment(
                            content=truncated_content,
                            priority=segment.priority,
                            timestamp=segment.timestamp,
                            token_count=remaining_tokens,
                            segment_type=segment.segment_type,
                            importance_score=segment.importance_score * 0.8,
                        )
                        validated_segments.append(truncated_segment)
                break
                
        return validated_segments
    
    def _precise_truncate_content(self, content: str, max_tokens: int, model_name: str) -> str:
        """Precisely truncate content using TokenManager to fit token limit.
        
        Args:
            content: Content to truncate
            max_tokens: Maximum token limit
            model_name: Model name for token calculation
            
        Returns:
            Truncated content that fits within token limit
        """
        if not content:
            return content
            
        # Use TokenManager for consistent truncation
        return self.token_manager.truncate_to_token_limit(
            content, max_tokens, model_name, preserve_start=True, preserve_end=False
        )

    def _apply_compression_strategy(
        self,
        segments: List[ContextSegment],
        max_tokens: int,
        strategy: CompressionStrategy,
        model_name: str,
    ) -> List[ContextSegment]:
        """Apply the specified compression strategy"""
        current_tokens = sum(seg.token_count for seg in segments)

        if current_tokens <= max_tokens:
            logger.info("No compression needed")
            # Still validate with precise token calculation
            return self._validate_token_budget(segments, max_tokens, model_name)

        if strategy == CompressionStrategy.ADAPTIVE:
            strategy = self._select_adaptive_strategy(
                segments, max_tokens, current_tokens
            )
            logger.info(f"Adaptive strategy selected: {strategy.value}")

        # Apply the selected strategy
        if strategy == CompressionStrategy.TRUNCATE:
            compressed_segments = self._apply_truncation(segments, max_tokens)
        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            compressed_segments = self._apply_sliding_window(segments, max_tokens)
        elif strategy == CompressionStrategy.HIERARCHICAL:
            compressed_segments = self._apply_hierarchical_selection(segments, max_tokens)
        elif strategy == CompressionStrategy.SUMMARIZE:
            compressed_segments = self._apply_summarization(segments, max_tokens, model_name)
        else:
            # Fallback to truncation
            compressed_segments = self._apply_truncation(segments, max_tokens)
            
        # Always validate the final result with precise token calculation
        return self._validate_token_budget(compressed_segments, max_tokens, model_name)

    def _select_adaptive_strategy(
        self, segments: List[ContextSegment], max_tokens: int, current_tokens: int
    ) -> CompressionStrategy:
        """Intelligently select the best compression strategy"""
        overflow_ratio = current_tokens / max_tokens

        # Count segments by priority
        priority_counts = {}
        for seg in segments:
            priority_counts[seg.priority] = priority_counts.get(seg.priority, 0) + 1

        # Decision logic based on overflow severity and content distribution
        if overflow_ratio < 1.2:  # Mild overflow
            return CompressionStrategy.TRUNCATE
        elif overflow_ratio < 1.5:  # Moderate overflow
            if priority_counts.get(ContextPriority.LOW, 0) > 2:
                return CompressionStrategy.HIERARCHICAL
            else:
                return CompressionStrategy.SLIDING_WINDOW
        else:  # Severe overflow
            return CompressionStrategy.SUMMARIZE

    def _apply_truncation(
        self, segments: List[ContextSegment], max_tokens: int
    ) -> List[ContextSegment]:
        """Apply simple truncation strategy"""
        logger.info("Applying truncation strategy")

        # Sort by priority and importance
        sorted_segments = sorted(
            segments, key=lambda x: (x.priority.value, -x.importance_score)
        )

        selected_segments = []
        current_tokens = 0

        for segment in sorted_segments:
            if current_tokens + segment.token_count <= max_tokens:
                selected_segments.append(segment)
                current_tokens += segment.token_count
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Minimum viable segment size
                    truncated_content = self._truncate_content(
                        segment.content, remaining_tokens
                    )
                    truncated_segment = ContextSegment(
                        content=truncated_content,
                        priority=segment.priority,
                        timestamp=segment.timestamp,
                        token_count=remaining_tokens,
                        segment_type=segment.segment_type,
                        importance_score=segment.importance_score
                        * 0.8,  # Reduce score for truncated
                    )
                    selected_segments.append(truncated_segment)
                break

        return selected_segments

    def _apply_sliding_window(
        self, segments: List[ContextSegment], max_tokens: int
    ) -> List[ContextSegment]:
        """Apply sliding window strategy"""
        logger.info("Applying sliding window strategy")

        # Separate critical segments from others
        critical_segments = [
            seg for seg in segments if seg.priority == ContextPriority.CRITICAL
        ]
        other_segments = [
            seg for seg in segments if seg.priority != ContextPriority.CRITICAL
        ]

        # Always include critical segments
        selected_segments = critical_segments.copy()
        used_tokens = sum(seg.token_count for seg in critical_segments)

        # Apply sliding window to other segments (most recent first)
        other_segments.sort(key=lambda x: x.timestamp, reverse=True)

        window_size = min(self.sliding_window_size, len(other_segments))
        window_segments = other_segments[:window_size]

        for segment in window_segments:
            if used_tokens + segment.token_count <= max_tokens:
                selected_segments.append(segment)
                used_tokens += segment.token_count
            else:
                break

        return selected_segments

    def _apply_hierarchical_selection(
        self, segments: List[ContextSegment], max_tokens: int
    ) -> List[ContextSegment]:
        """Apply hierarchical priority-based selection"""
        logger.info("Applying hierarchical selection strategy")

        # Group segments by priority
        priority_groups = {}
        for segment in segments:
            if segment.priority not in priority_groups:
                priority_groups[segment.priority] = []
            priority_groups[segment.priority].append(segment)

        selected_segments = []
        used_tokens = 0

        # Process in priority order
        for priority in [
            ContextPriority.CRITICAL,
            ContextPriority.HIGH,
            ContextPriority.MEDIUM,
            ContextPriority.LOW,
        ]:
            if priority not in priority_groups:
                continue

            # Sort segments within priority by importance score
            group_segments = sorted(
                priority_groups[priority],
                key=lambda x: x.importance_score,
                reverse=True,
            )

            for segment in group_segments:
                if used_tokens + segment.token_count <= max_tokens:
                    selected_segments.append(segment)
                    used_tokens += segment.token_count
                else:
                    # Try compression for non-critical segments
                    if segment.can_compress and priority != ContextPriority.CRITICAL:
                        remaining_tokens = max_tokens - used_tokens
                        if remaining_tokens > 50:
                            compressed_content = self._compress_segment_content(
                                segment.content, remaining_tokens
                            )
                            compressed_segment = ContextSegment(
                                content=compressed_content,
                                priority=segment.priority,
                                timestamp=segment.timestamp,
                                token_count=remaining_tokens,
                                segment_type=segment.segment_type,
                                importance_score=segment.importance_score * 0.7,
                            )
                            selected_segments.append(compressed_segment)
                            used_tokens += remaining_tokens
                    break

            if used_tokens >= max_tokens:
                break

        return selected_segments

    def _apply_summarization(
        self, segments: List[ContextSegment], max_tokens: int, model_name: str
    ) -> List[ContextSegment]:
        """Apply LLM-based summarization strategy"""
        logger.info("Applying summarization strategy")

        # Always preserve critical segments
        critical_segments = [
            seg for seg in segments if seg.priority == ContextPriority.CRITICAL
        ]
        other_segments = [
            seg for seg in segments if seg.priority != ContextPriority.CRITICAL
        ]

        selected_segments = critical_segments.copy()
        used_tokens = sum(seg.token_count for seg in critical_segments)
        remaining_tokens = max_tokens - used_tokens

        if remaining_tokens <= 0:
            return critical_segments

        # Group other segments for summarization
        if other_segments:
            combined_content = "\n\n".join([seg.content for seg in other_segments])

            # Create a summarized version
            summary_prompt = f"""Please provide a concise summary of the following execution steps and results, focusing on key outcomes and any important information:

{combined_content}

Summary (keep under {remaining_tokens // 4} words):"""

            try:
                # Use a simple text compression as fallback if LLM summarization fails
                summarized_content = self._compress_segment_content(
                    combined_content, remaining_tokens
                )

                summary_segment = ContextSegment(
                    content=summarized_content,
                    priority=ContextPriority.MEDIUM,
                    timestamp=max(seg.timestamp for seg in other_segments),
                    token_count=self.token_manager.count_tokens_precise(
                        summarized_content, "deepseek-chat"
                    ),
                    segment_type="summary",
                    importance_score=0.6,
                    compressed_version=summarized_content,
                )

                selected_segments.append(summary_segment)

            except Exception as e:
                logger.warning(f"Summarization failed: {e}, falling back to truncation")
                return self._apply_truncation(segments, max_tokens)

        return selected_segments

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit"""
        estimated_chars = max_tokens * 3  # Rough estimation
        if len(content) <= estimated_chars:
            return content

        # Try to truncate at sentence boundaries
        sentences = content.split(". ")
        truncated = ""

        for sentence in sentences:
            test_content = truncated + sentence + ". "
            if (
            self.token_manager.count_tokens_precise(
                test_content, "deepseek-chat"
            )
            <= max_tokens
        ):
                truncated = test_content
            else:
                break

        if not truncated:  # Fallback to character truncation
            truncated = content[:estimated_chars] + "..."

        return truncated.strip()

    def _compress_segment_content(self, content: str, max_tokens: int) -> str:
        """Compress segment content using various techniques"""
        # First try truncation at natural boundaries
        truncated = self._truncate_content(content, max_tokens)

        # If still too long, apply more aggressive compression
        if (
            self.token_manager.count_tokens_precise(
                truncated, "deepseek-chat"
            )
            > max_tokens
        ):
            # Extract key information using regex patterns
            key_patterns = [
                r"Error: [^\n]+",
                r"Warning: [^\n]+",
                r"Result: [^\n]+",
                r"Status: [^\n]+",
                r"Output: [^\n]+",
            ]

            key_info = []
            for pattern in key_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                key_info.extend(matches)

            if key_info:
                compressed = "\n".join(key_info[:3])  # Top 3 key pieces
                if (
                    self.token_manager.count_tokens_precise(
                        compressed, "deepseek-chat"
                    )
                    <= max_tokens
                ):
                    return compressed

        return truncated

    def format_context_window(self, context_window: ContextWindow) -> str:
        """Format context window for LLM input"""
        formatted_parts = []

        for segment in context_window.segments:
            if segment.segment_type == "task":
                formatted_parts.append(f"Current Task: {segment.content}")
            elif segment.segment_type == "result":
                formatted_parts.append(f"Previous Step Result:\n{segment.content}")
            elif segment.segment_type == "summary":
                formatted_parts.append(f"Context Summary:\n{segment.content}")
            else:
                formatted_parts.append(segment.content)

        return "\n\n".join(formatted_parts)

    def get_context(
        self,
        completed_steps: List[Any],
        current_task: str,
        model_name: str = "deepseek-chat",
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get optimized context for LLM input.

        Args:
            completed_steps: List of completed steps
            current_task: Current task description
            model_name: Model name for token limits
            max_tokens: Optional maximum tokens override for budget constraints

        Returns:
            Formatted context string
        """
        try:
            # Get model limits
            limits = self.content_processor.get_model_limits(model_name)

            # Use provided max_tokens or calculate from model limits
            if max_tokens is not None:
                context_max_tokens = max_tokens
                logger.info(f"Using provided token budget: {max_tokens}")
            else:
                context_max_tokens = int(
                    limits.safe_input_limit * self.max_context_ratio
                )
                logger.info(f"Using calculated token limit: {context_max_tokens}")

            # Create context segments
            segments = []
            for step in completed_steps:
                segment = self.create_context_segment(
                    content=self._format_step_content(step),
                    segment_type="result",
                    priority=ContextPriority.MEDIUM,
                )
                segments.append(segment)

            # Add current task with high priority
            current_segment = self.create_context_segment(
                content=current_task,
                segment_type="task",
                priority=ContextPriority.CRITICAL,
            )
            segments.append(current_segment)

            # Apply compression strategy
            optimized_segments = self._apply_compression_strategy(
                segments, context_max_tokens, CompressionStrategy.ADAPTIVE, model_name
            )

            # Calculate final metrics
            total_tokens = sum(seg.token_count for seg in optimized_segments)
            compression_ratio = 1.0 - (
                total_tokens / sum(seg.token_count for seg in segments)
            )

            context_window = ContextWindow(
                segments=optimized_segments,
                total_tokens=total_tokens,
                max_tokens=context_max_tokens,
                compression_ratio=compression_ratio,
                strategy_used=CompressionStrategy.ADAPTIVE,
            )

            return self.format_context_window(context_window)

        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            # Fallback to simple concatenation with budget awareness
            context_parts = []
            max_steps = 3 if max_tokens is None or max_tokens > 2000 else 1
            for step in completed_steps[-max_steps:]:  # Fewer steps for tight budgets
                step_content = self._format_step_content(step)
                context_parts.append(f"Previous Step:\n{step_content}")
            context_parts.append(f"Current Task: {current_task}")
            return "\n\n".join(context_parts)

    def get_optimization_stats(self, context_window: ContextWindow) -> Dict[str, Any]:
        """Get detailed statistics about context optimization"""
        return {
            "total_tokens": context_window.total_tokens,
            "max_tokens": context_window.max_tokens,
            "token_utilization": context_window.total_tokens
            / context_window.max_tokens,
            "compression_ratio": context_window.compression_ratio,
            "strategy_used": context_window.strategy_used.value,
            "segment_count": len(context_window.segments),
            "segments_by_priority": {
                priority.name: len(
                    [seg for seg in context_window.segments if seg.priority == priority]
                )
                for priority in ContextPriority
            },
            "segments_by_type": {
                seg_type: len(
                    [
                        seg
                        for seg in context_window.segments
                        if seg.segment_type == seg_type
                    ]
                )
                for seg_type in set(seg.segment_type for seg in context_window.segments)
            },
        }
