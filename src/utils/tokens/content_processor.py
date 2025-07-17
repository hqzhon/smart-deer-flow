# SPDX-License-Identifier: MIT
"""
Intelligent content chunking and summarization processing module
Provides content chunking, summarization and optimization based on different model token limits
"""

import logging
import re
import html
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from .token_counter import get_token_counter, TokenCountResult

logger = logging.getLogger(__name__)


@dataclass
class ModelTokenLimits:
    """Model token limit configuration"""

    input_limit: int  # Input token limit
    output_limit: int  # Output token limit
    context_window: int  # Context window size
    safety_margin: float = 0.8  # Safety margin, use 80% of the limit

    @property
    def safe_input_limit(self) -> int:
        """Safe input limit"""
        return int(self.input_limit * self.safety_margin)

    @property
    def safe_output_limit(self) -> int:
        """Safe output limit"""
        return int(self.output_limit * self.safety_margin)


class ContentProcessor:
    """Intelligent content processor with security validation"""

    # Security patterns for malicious content detection
    MALICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"on\w+\s*=",  # Event handlers
        r"data:text/html",  # Data URLs
        r"vbscript:",  # VBScript
        r"expression\s*\(",  # CSS expressions
        r"@import",  # CSS imports
        r"\\x[0-9a-fA-F]{2}",  # Hex encoded characters
        r"%[0-9a-fA-F]{2}",  # URL encoded characters (suspicious patterns)
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(union|select|insert|update|delete|drop|create|alter)\s+",
        r"(or|and)\s+\d+\s*=\s*\d+",
        r'[\'"]\s*(or|and)\s+[\'"]',
        r"--\s*$",  # SQL comments
        r"/\*.*?\*/",  # SQL block comments
        r"xp_cmdshell",  # SQL Server command execution
        r"sp_executesql",  # SQL Server dynamic SQL
    ]

    def __init__(self, model_limits: Optional[Dict[str, Any]] = None):
        # If no model_limits provided, try to get from llm registry first
        if model_limits is None or not model_limits:
            try:
                from src.llms.llm import get_model_token_limits_registry

                model_limits = get_model_token_limits_registry()
                if model_limits:
                    logger.info(
                        f"Loaded token limits from LLM registry for models: {list(model_limits.keys())}"
                    )
                else:
                    logger.warning(
                        "LLM registry returned empty model limits, trying config file"
                    )
                    # Try to load from config file as fallback
                    model_limits = self._load_model_limits_from_config()
            except ImportError:
                logger.warning("Could not import LLM registry, trying config file")
                model_limits = self._load_model_limits_from_config()
            except Exception as e:
                logger.warning(
                    f"Failed to get model limits from LLM registry: {e}, trying config file"
                )
                model_limits = self._load_model_limits_from_config()

        # Convert dictionary format to ModelTokenLimits objects if needed
        self.model_limits = self._normalize_model_limits(model_limits or {})
        self.default_limits = ModelTokenLimits(
            input_limit=65536,
            output_limit=8192,
            context_window=65536,
            safety_margin=0.8,
        )

        if self.model_limits:
            logger.info(
                f"ContentProcessor initialized with model limits for: {list(self.model_limits.keys())}"
            )
        else:
            logger.warning(
                "ContentProcessor initialized with empty model limits, will use default limits"
            )
        # Compile regex patterns for better performance
        self._malicious_regex = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.MALICIOUS_PATTERNS
        ]
        self._sql_injection_regex = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]

    def _normalize_model_limits(
        self, model_limits: Dict[str, Any]
    ) -> Dict[str, ModelTokenLimits]:
        """Convert various model limit formats to ModelTokenLimits objects.

        Args:
            model_limits: Model limits in various formats (dict or ModelTokenLimits)

        Returns:
            Dictionary mapping model names to ModelTokenLimits objects
        """
        normalized = {}

        for model_name, limits in model_limits.items():
            if isinstance(limits, ModelTokenLimits):
                # Already a ModelTokenLimits object
                normalized[model_name] = limits
            elif isinstance(limits, dict):
                # Convert dictionary to ModelTokenLimits
                try:
                    normalized[model_name] = ModelTokenLimits(
                        input_limit=limits.get("input_limit", 65536),
                        output_limit=limits.get("output_limit", 8192),
                        context_window=limits.get("context_window", 65536),
                        safety_margin=limits.get("safety_margin", 0.8),
                    )
                    logger.debug(
                        f"Converted dictionary to ModelTokenLimits for model: {model_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to convert model limits for {model_name}: {e}, using defaults"
                    )
                    normalized[model_name] = self.default_limits
            else:
                logger.warning(
                    f"Unknown model limits format for {model_name}: {type(limits)}, using defaults"
                )
                normalized[model_name] = self.default_limits

        return normalized

    def _load_model_limits_from_config(self) -> Dict[str, ModelTokenLimits]:
        """Load model token limits from new configuration system as fallback."""
        try:
            from src.config.config_loader import load_configuration

            # Load configuration using new unified system
            settings = load_configuration()

            model_limits = {}

            # Load model_token_limits from new configuration system
            for model_name, limits_config in settings.model_token_limits.items():
                model_limits[model_name] = ModelTokenLimits(
                    input_limit=limits_config.get("input_limit", 32000),
                    output_limit=limits_config.get("output_limit", 4096),
                    context_window=limits_config.get("context_window", 32000),
                    safety_margin=limits_config.get("safety_margin", 0.8),
                )
                logger.info(
                    f"Loaded token limits for {model_name} from new configuration system"
                )

            return model_limits

        except Exception as e:
            logger.error(f"Failed to load model limits from new config system: {e}")
            return {}

    def validate_input_security(self, content: str) -> Tuple[bool, List[str]]:
        """Validate input for security threats - optimized version with reduced false positives

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_safe, list_of_threats_found)
        """
        threats = []

        # First check if content matches safe patterns (whitelist)
        safe_patterns = [
            r"^https?://[\w.-]+/[\w./%-]*$",  # Standard HTTP URLs
            r'^[\w\s.,!?;:()\[\]{}"\'-]+$',  # Basic text content
            r'^[\u4e00-\u9fff\w\s.,!?;:()\[\]{}"\'-]+$',  # Text with Chinese characters
        ]

        for pattern in safe_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return True, []

        # Check for malicious patterns (more precise blacklist)
        precise_malicious_patterns = [
            # Script injection
            r"<script[^>]*>.*?</script>",
            r"javascript:(?!void\(0\))",  # Exclude common void(0)
            r'on\w+\s*=\s*["\'][^"\'>]*["\']',  # Event handlers
            # Data URLs (more strict)
            r"data:(?!image/)[^,]*,",  # Exclude image data URLs
            # URL encoding (more precise, only check dangerous encodings)
            r"%(?:2[2-9]|[3-6][0-9a-fA-F]|7[0-9a-eA-E])",  # Dangerous URL encodings
            # XSS (more precise)
            r"<(?:iframe|object|embed|form)[^>]*>",
        ]

        for pattern in precise_malicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats.append(f"Malicious pattern detected: {pattern}")

        # Check for SQL injection patterns (more precise)
        precise_sql_patterns = [
            r"\b(?:union|select|insert|update|delete|drop)\s+(?:all\s+)?(?:distinct\s+)?\w+",
            r"(or|and)\s+\d+\s*=\s*\d+",
            r'[\'"]\s*(or|and)\s+[\'"]',
            r"--\s*$",  # SQL comments
            r"/\*.*?\*/",  # SQL block comments
        ]

        for pattern in precise_sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats.append(f"SQL injection pattern detected: {pattern}")

        # Check for excessive length (potential DoS)
        if len(content) > 1000000:  # 1MB limit
            threats.append("Content exceeds maximum allowed length")

        # Check for suspicious encoding (improved)
        try:
            # Try to decode potential encoded content
            decoded = html.unescape(content)
            if decoded != content:
                # Re-check decoded content for malicious patterns
                for pattern in precise_malicious_patterns:
                    if re.search(pattern, decoded, re.IGNORECASE):
                        threats.append(
                            f"Malicious pattern in decoded content: {pattern}"
                        )
        except Exception:
            # If decoding fails, it might be suspicious
            threats.append("Suspicious encoding detected")

        # If threats detected but content looks like safe URL encoding, reduce threat level
        if threats and self._is_likely_safe_url_encoding(content):
            # Filter out URL encoding related threats
            threats = [t for t in threats if "URL encoding" not in t and "%" not in t]

        return len(threats) == 0, threats

    def _is_likely_safe_url_encoding(self, content: str) -> bool:
        """Check if content is likely safe URL encoding"""
        # If content mainly contains URL encoded normal characters
        url_encoded_chars = re.findall(r"%[0-9a-fA-F]{2}", content)
        if url_encoded_chars:
            # Check if encoded characters are common safe characters
            safe_encoded_patterns = [
                r"%20",  # Space
                r"%2[1-9]",  # Common punctuation
                r"%3[0-9a-fA-F]",  # Numbers and some symbols
            ]

            safe_count = 0
            for encoded in url_encoded_chars:
                for safe_pattern in safe_encoded_patterns:
                    if re.match(safe_pattern, encoded, re.IGNORECASE):
                        safe_count += 1
                        break

            # If most encoded characters are safe, consider it safe URL encoding
            return safe_count / len(url_encoded_chars) > 0.7

        return False

    def sanitize_content(self, content: str) -> str:
        """Sanitize content by removing potentially dangerous elements

        Args:
            content: Content to sanitize

        Returns:
            Sanitized content
        """
        if not content:
            return content

        # HTML escape the content
        sanitized = html.escape(content)

        # Remove potential script tags and event handlers
        sanitized = re.sub(
            r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL
        )
        sanitized = re.sub(
            r'on\w+\s*=\s*["\'][^"\'>]*["\']', "", sanitized, flags=re.IGNORECASE
        )
        sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"vbscript:", "", sanitized, flags=re.IGNORECASE)

        # Remove SQL injection patterns
        for pattern in self.SQL_INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    def get_model_limits(self, model_name: str) -> ModelTokenLimits:
        """Get model token limits"""
        logger.debug(f"Looking for token limits for model: '{model_name}'")
        logger.debug(
            f"Available model configurations: {list(self.model_limits.keys())}"
        )

        # Try exact match
        if model_name in self.model_limits:
            logger.debug(f"Found exact match for model '{model_name}'")
            return self.model_limits[model_name]

        # Try fuzzy match with improved logic
        for key, limits in self.model_limits.items():
            # Check if model names match (case-insensitive)
            if key.lower() == model_name.lower():
                logger.debug(
                    f"Found case-insensitive match: '{key}' for model '{model_name}'"
                )
                return limits

            # Check if one is contained in the other
            if key.lower() in model_name.lower() or model_name.lower() in key.lower():
                logger.debug(f"Found fuzzy match: '{key}' for model '{model_name}'")
                return limits

            # Check for common model name variations
            # Remove common prefixes/suffixes and compare
            clean_key = (
                key.lower()
                .replace("-chat", "")
                .replace("_chat", "")
                .replace("-v3", "")
                .replace("-v2", "")
                .replace("-v1", "")
            )
            clean_model = (
                model_name.lower()
                .replace("-chat", "")
                .replace("_chat", "")
                .replace("-v3", "")
                .replace("-v2", "")
                .replace("-v1", "")
            )

            if (
                clean_key == clean_model
                or clean_key in clean_model
                or clean_model in clean_key
            ):
                logger.debug(
                    f"Found normalized match: '{key}' (normalized: '{clean_key}') for model '{model_name}' (normalized: '{clean_model}')"
                )
                return limits

        logger.warning(
            f"Token limit configuration not found for model '{model_name}' in available configurations: {list(self.model_limits.keys())}, using default configuration"
        )
        return self.default_limits

    def estimate_tokens(self, text: str, model_name: str = "deepseek-chat") -> int:
        """Get accurate token count using tiktoken"""
        try:
            token_counter = get_token_counter()
            result = token_counter.count_tokens(text, model_name)
            logger.debug(
                f"Accurate token count for model {model_name}: {result.total_tokens} tokens"
            )
            return result.total_tokens
        except Exception as e:
            logger.warning(
                f"Failed to get accurate token count, falling back to estimation: {e}"
            )
            # Fallback to conservative estimation
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            english_chars = len(text) - chinese_chars

            # Apply conservative multipliers
            chinese_tokens = int(chinese_chars * 1.2)
            english_tokens = int(english_chars / 3)

            # Add safety buffer
            total_tokens = chinese_tokens + english_tokens
            safety_buffer = int(total_tokens * 0.15)  # 15% safety buffer for fallback

            return total_tokens + safety_buffer

    def count_tokens_accurate(self, text: str, model_name: str) -> TokenCountResult:
        """Get detailed token count information"""
        token_counter = get_token_counter()
        return token_counter.count_tokens(text, model_name)

    def check_content_token_limit(
        self, content: str, model_name: str, max_tokens: int, safety_margin: float = 0.9
    ) -> Tuple[bool, TokenCountResult]:
        """Check if content exceeds token limit with detailed information"""
        token_counter = get_token_counter()
        return token_counter.check_token_limit(
            content, model_name, max_tokens, safety_margin
        )

    def chunk_text_by_sentences(
        self, text: str, max_tokens: int, model_name: str = "deepseek-chat"
    ) -> List[str]:
        """Chunk text by sentences with accurate token counting"""
        # Split by sentences
        sentences = re.split(r"[.!?。！？]\s*", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence = sentence.strip() + "。"  # Add period
            sentence_tokens = self.count_tokens_accurate(
                sentence, model_name
            ).total_tokens
            current_tokens = (
                self.count_tokens_accurate(current_chunk, model_name).total_tokens
                if current_chunk
                else 0
            )

            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text_by_paragraphs(
        self, text: str, max_tokens: int, model_name: str = "deepseek-chat"
    ) -> List[str]:
        """Chunk text by paragraphs with accurate token counting"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            paragraph_tokens = self.count_tokens_accurate(
                paragraph, model_name
            ).total_tokens
            current_tokens = (
                self.count_tokens_accurate(current_chunk, model_name).total_tokens
                if current_chunk
                else 0
            )

            if current_tokens + paragraph_tokens <= max_tokens:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If a single paragraph is too long, chunk by sentences
                if paragraph_tokens > max_tokens:
                    sentence_chunks = self.chunk_text_by_sentences(
                        paragraph, max_tokens, model_name
                    )
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def smart_chunk_content(
        self, content: str, model_name: str, chunk_strategy: str = "auto"
    ) -> List[str]:
        """Smart chunk content with security validation and accurate token counting

        Args:
            content: Content to be chunked
            model_name: Model name
            chunk_strategy: Chunking strategy ("sentences", "paragraphs", "auto", "aggressive")

        Returns:
            List of chunked content

        Raises:
            ValueError: If content contains security threats
        """
        # Validate input security first
        is_safe, threats = self.validate_input_security(content)
        if not is_safe:
            logger.warning(f"Security threats detected in content: {threats}")
            # Sanitize content instead of rejecting it completely
            content = self.sanitize_content(content)
            logger.info("Content has been sanitized")

        limits = self.get_model_limits(model_name)
        max_tokens = limits.safe_input_limit

        # Preemptive token check with accurate counting
        is_within_limit, token_result = self.check_content_token_limit(
            content, model_name, max_tokens, limits.safety_margin
        )

        if is_within_limit:
            logger.debug(
                f"Content within token limit ({token_result.total_tokens} <= {max_tokens}), no chunking needed"
            )
            return [content]

        logger.info(
            f"Content exceeds token limit for model {model_name} "
            f"({token_result.total_tokens} > {max_tokens}), starting chunking process"
        )

        if chunk_strategy == "auto":
            # Auto strategy: try paragraph chunking first, then sentence chunking if chunks are too large
            chunks = self.chunk_text_by_paragraphs(content, max_tokens, model_name)

            # Check if any chunks are still too large using accurate counting
            final_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_within_limit, chunk_token_result = self.check_content_token_limit(
                    chunk, model_name, max_tokens, limits.safety_margin
                )

                if not chunk_within_limit:
                    logger.debug(
                        f"Chunk {i+1} still too large ({chunk_token_result.total_tokens} tokens), using sentence chunking"
                    )
                    sentence_chunks = self.chunk_text_by_sentences(
                        chunk, max_tokens, model_name
                    )
                    final_chunks.extend(sentence_chunks)
                else:
                    final_chunks.append(chunk)

            return final_chunks

        elif chunk_strategy == "paragraphs":
            return self.chunk_text_by_paragraphs(content, max_tokens, model_name)

        elif chunk_strategy == "sentences":
            return self.chunk_text_by_sentences(content, max_tokens, model_name)

        elif chunk_strategy == "aggressive":
            # Aggressive strategy: use much smaller chunks for extremely long content
            aggressive_max_tokens = min(
                max_tokens // 4, 2000
            )  # Use 1/4 of normal limit or 2000 tokens max
            logger.info(
                f"Using aggressive chunking with max {aggressive_max_tokens} tokens per chunk"
            )

            # Try sentence chunking first with reduced limit
            chunks = self.chunk_text_by_sentences(
                content, aggressive_max_tokens, model_name
            )

            # If still too large, use character-based chunking
            final_chunks = []
            for chunk in chunks:
                chunk_within_limit, chunk_token_result = self.check_content_token_limit(
                    chunk, model_name, aggressive_max_tokens, limits.safety_margin
                )

                if not chunk_within_limit:
                    logger.debug(
                        f"Chunk still too large ({chunk_token_result.total_tokens} tokens), using character-based chunking"
                    )
                    # Character-based chunking as last resort
                    # Use precise token-based truncation instead of character estimation
                    # Binary search for precise truncation
                    content = chunk
                    left, right = 0, len(content)
                    best_content = ""

                    while left <= right:
                        mid = (left + right) // 2
                        test_content = content[:mid] + "...[truncated]"
                        test_tokens = self.estimate_tokens(test_content, model_name)

                        if test_tokens <= aggressive_max_tokens:
                            best_content = test_content
                            left = mid + 1
                        else:
                            right = mid - 1

                    if best_content:
                        final_chunks.append(best_content)
                    else:
                        # Fallback to conservative character limit
                        char_limit = (
                            aggressive_max_tokens * 2.5
                        )  # More conservative estimate
                    chunk_parts = [
                        chunk[i : i + char_limit]
                        for i in range(0, len(chunk), char_limit)
                    ]
                    final_chunks.extend(chunk_parts)
                else:
                    final_chunks.append(chunk)

            return final_chunks[
                :3
            ]  # Return only first 3 chunks for aggressive strategy

        else:
            raise ValueError(f"Unsupported chunking strategy: {chunk_strategy}")

    def create_summary_prompt(
        self, content: str, summary_type: str = "comprehensive"
    ) -> str:
        """Create summary prompt"""
        if summary_type == "comprehensive":
            return f"""Please provide a comprehensive summary of the following content, retaining key information and important details:

{content}

Summary requirements:
1. Retain all important facts and data
2. Maintain clear logical structure
3. Use concise and clear language
4. Ensure summary content is accurate

Summary:"""

        elif summary_type == "key_points":
            return f"""Please extract key points from the following content:

{content}

Requirements:
1. List in bullet point format
2. Each point should be concise and clear
3. Retain the most important information
4. Sort by importance

Key points:"""

        elif summary_type == "abstract":
            return f"""Please write a concise abstract for the following content:

{content}

Abstract requirements:
1. Keep within 200 words
2. Highlight core viewpoints
3. Use concise and accurate language

Abstract:"""

        else:
            raise ValueError(f"Unsupported summary type: {summary_type}")

    def summarize_content(
        self,
        content: str,
        llm: BaseChatModel,
        model_name: str,
        summary_type: str = "comprehensive",
    ) -> str:
        """Summarize content with accurate token counting and preemptive checks

        Args:
            content: Content to be summarized
            llm: Language model
            model_name: Model name
            summary_type: Summary type

        Returns:
            Summarized content
        """
        limits = self.get_model_limits(model_name)

        # Preemptive token check for content
        is_within_limit, token_result = self.check_content_token_limit(
            content, model_name, limits.safe_input_limit, limits.safety_margin
        )

        if is_within_limit:
            logger.debug(
                f"Content within token limit ({token_result.total_tokens} tokens), direct summarization"
            )
            prompt = self.create_summary_prompt(content, summary_type)

            # Check prompt token count before sending
            prompt_within_limit, prompt_token_result = self.check_content_token_limit(
                prompt, model_name, limits.safe_input_limit, limits.safety_margin
            )

            if not prompt_within_limit:
                logger.warning(
                    f"Prompt exceeds token limit ({prompt_token_result.total_tokens} > {limits.safe_input_limit}), "
                    "falling back to chunked summarization"
                )
            else:
                messages = [HumanMessage(content=prompt)]

                try:
                    from src.llms.error_handler import safe_llm_call

                    response = safe_llm_call(
                        llm.invoke,
                        messages,
                        operation_name="Content Summarization",
                        context="Summarizing content within token limits",
                        max_retries=3,
                    )
                    return response.content
                except Exception as e:
                    logger.error(f"Error occurred while summarizing content: {e}")
                    return content  # Return original content

        # Content too long, need chunked summarization
        logger.info(
            f"Content too long ({token_result.total_tokens} > {limits.safe_input_limit}), "
            "using chunked summarization strategy"
        )
        chunks = self.smart_chunk_content(content, model_name)

        summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")

            prompt = self.create_summary_prompt(chunk, summary_type)

            # Check each prompt before sending
            prompt_within_limit, prompt_token_result = self.check_content_token_limit(
                prompt, model_name, limits.safe_input_limit, limits.safety_margin
            )

            if not prompt_within_limit:
                logger.warning(
                    f"Chunk {i+1} prompt exceeds token limit ({prompt_token_result.total_tokens} tokens), "
                    "using original chunk content"
                )
                summaries.append(chunk)
                continue

            messages = [HumanMessage(content=prompt)]

            try:
                from src.llms.error_handler import safe_llm_call

                response = safe_llm_call(
                    llm.invoke,
                    messages,
                    operation_name="Chunk Summarization",
                    context=f"Summarizing chunk {i+1}/{len(chunks)}",
                    max_retries=3,
                )
                summaries.append(response.content)
            except Exception as e:
                logger.error(f"Error occurred while summarizing chunk {i+1}: {e}")
                summaries.append(chunk)  # Use original content

        # Merge all summaries
        combined_summary = "\n\n".join(summaries)

        # Check if merged summary is still too long
        combined_within_limit, combined_token_result = self.check_content_token_limit(
            combined_summary, model_name, limits.safe_input_limit, limits.safety_margin
        )

        if not combined_within_limit:
            logger.info(
                f"Merged summary still too long ({combined_token_result.total_tokens} > {limits.safe_input_limit}), "
                "performing secondary summarization"
            )
            return self.summarize_content(combined_summary, llm, model_name, "abstract")

        return combined_summary

    def process_search_results(
        self,
        search_results: List[Dict[str, Any]],
        llm: BaseChatModel,
        model_name: str,
        max_results: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        """Process search results with chunking, summarization and security validation

        Note: Smart filtering is now handled exclusively at the SmartSearchTool level.
        This method focuses on traditional content processing.

        Args:
            search_results: List of search results
            llm: Language model
            model_name: Model name
            max_results: Maximum number of results
            query: Original user query (for logging purposes)

        Returns:
            Processed search results text
        """
        if not search_results:
            return "No relevant search results found."

        # Limit number of search results
        if max_results:
            search_results = search_results[:max_results]

        # First format results to check content length
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "No title")
            content = result.get("content", result.get("snippet", "No content"))
            url = result.get("url", "No link")

            # Validate and sanitize each field
            title = self.sanitize_content(str(title))
            content = self.sanitize_content(str(content))
            url = self.sanitize_content(str(url))

            # Additional URL validation
            if not self._is_safe_url(url):
                logger.warning(f"Potentially unsafe URL detected: {url}")
                url = "[URL removed for security]"

            formatted_result = f"""Search Result {i}:
Title: {title}
Link: {url}
Content: {content}
"""
            formatted_results.append(formatted_result)

        combined_results = "\n\n".join(formatted_results)

        # Get model limits for token checking
        limits = self.get_model_limits(model_name)

        # Preemptive token check for combined results
        results_within_limit, results_token_result = self.check_content_token_limit(
            combined_results, model_name, limits.safe_input_limit, limits.safety_margin
        )

        # Note: Smart filtering is now handled exclusively at the SmartSearchTool level
        # This method focuses on traditional content processing (chunking/summarization)
        logger.info(
            f"Processing search results with traditional methods "
            f"(tokens: {results_token_result.total_tokens}, limit: {limits.safe_input_limit})"
        )

        # Check if summarization is needed
        if not results_within_limit:
            logger.info(
                f"Search results too long ({results_token_result.total_tokens} > {limits.safe_input_limit}), "
                "starting intelligent summarization"
            )
            return self.summarize_content(
                combined_results, llm, model_name, "key_points"
            )

        # Final security check on combined results
        is_safe, threats = self.validate_input_security(combined_results)
        if not is_safe:
            logger.warning(
                f"Security threats detected in combined search results: {threats}"
            )
            combined_results = self.sanitize_content(combined_results)

        return combined_results

    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe

        Args:
            url: URL to check

        Returns:
            True if URL appears safe
        """
        if not url or url == "No link":
            return True

        # Check for suspicious URL patterns
        suspicious_patterns = [
            r"javascript:",
            r"data:",
            r"vbscript:",
            r"file:",
            r"ftp://.*@",  # FTP with credentials
            r"\d+\.\d+\.\d+\.\d+",  # Raw IP addresses (potentially suspicious)
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        return True
