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
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

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
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'data:text/html',  # Data URLs
        r'vbscript:',  # VBScript
        r'expression\s*\(',  # CSS expressions
        r'@import',  # CSS imports
        r'\\x[0-9a-fA-F]{2}',  # Hex encoded characters
        r'%[0-9a-fA-F]{2}',  # URL encoded characters (suspicious patterns)
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(union|select|insert|update|delete|drop|create|alter)\s+',
        r'(or|and)\s+\d+\s*=\s*\d+',
        r'[\'"]\s*(or|and)\s+[\'"]',
        r'--\s*$',  # SQL comments
        r'/\*.*?\*/',  # SQL block comments
        r'xp_cmdshell',  # SQL Server command execution
        r'sp_executesql',  # SQL Server dynamic SQL
    ]
    
    def __init__(self, model_limits: Optional[Dict[str, ModelTokenLimits]] = None):
        # If no model_limits provided, try to get from llm registry first
        if model_limits is None or not model_limits:
            try:
                from src.llms.llm import get_model_token_limits_registry
                model_limits = get_model_token_limits_registry()
                if model_limits:
                    logger.info(f"Loaded token limits from LLM registry for models: {list(model_limits.keys())}")
                else:
                    logger.warning("LLM registry returned empty model limits, trying config file")
                    # Try to load from config file as fallback
                    model_limits = self._load_model_limits_from_config()
            except ImportError:
                logger.warning("Could not import LLM registry, trying config file")
                model_limits = self._load_model_limits_from_config()
            except Exception as e:
                logger.warning(f"Failed to get model limits from LLM registry: {e}, trying config file")
                model_limits = self._load_model_limits_from_config()
        
        self.model_limits = model_limits or {}
        self.default_limits = ModelTokenLimits(
            input_limit=65536,
            output_limit=8192,
            context_window=65536,
            safety_margin=0.8
        )
        
        if self.model_limits:
            logger.info(f"ContentProcessor initialized with model limits for: {list(self.model_limits.keys())}")
        else:
            logger.warning("ContentProcessor initialized with empty model limits, will use default limits")
        # Compile regex patterns for better performance
        self._malicious_regex = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in self.MALICIOUS_PATTERNS]
        self._sql_injection_regex = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in self.SQL_INJECTION_PATTERNS]
    
    def _load_model_limits_from_config(self) -> Dict[str, ModelTokenLimits]:
        """Load model token limits from configuration file as fallback."""
        try:
            import yaml
            import os
            from pathlib import Path
            
            # Find config file
            config_paths = [
                Path("conf.yaml"),
                Path("config.yaml"),
                Path("src/config/conf.yaml"),
                Path("../conf.yaml")
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if not config_file:
                logger.warning("Could not find configuration file")
                return {}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            model_limits = {}
            
            # Load BASIC_MODEL configuration
            if 'BASIC_MODEL' in config:
                basic_config = config['BASIC_MODEL']
                if 'token_limits' in basic_config:
                    token_config = basic_config['token_limits']
                    model_name = basic_config.get('model_name', 'deepseek-chat')
                    model_limits[model_name] = ModelTokenLimits(
                        input_limit=token_config.get('input_limit', 65536),
                        output_limit=token_config.get('output_limit', 8192),
                        context_window=token_config.get('context_window', 65536),
                        safety_margin=token_config.get('safety_margin', 0.8)
                    )
                    logger.info(f"Loaded token limits for {model_name} from BASIC_MODEL config")
            
            # Load REASONING_MODEL configuration
            if 'REASONING_MODEL' in config:
                reasoning_config = config['REASONING_MODEL']
                if 'token_limits' in reasoning_config:
                    token_config = reasoning_config['token_limits']
                    model_name = reasoning_config.get('model_name', 'deepseek-reasoner')
                    model_limits[model_name] = ModelTokenLimits(
                        input_limit=token_config.get('input_limit', 65536),
                        output_limit=token_config.get('output_limit', 8192),
                        context_window=token_config.get('context_window', 65536),
                        safety_margin=token_config.get('safety_margin', 0.8)
                    )
                    logger.info(f"Loaded token limits for {model_name} from REASONING_MODEL config")
            
            return model_limits
            
        except Exception as e:
            logger.error(f"Failed to load model limits from config file: {e}")
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
            r'^https?://[\w.-]+/[\w./%-]*$',  # Standard HTTP URLs
            r'^[\w\s.,!?;:()\[\]{}"\'-]+$',  # Basic text content
            r'^[\u4e00-\u9fff\w\s.,!?;:()\[\]{}"\'-]+$',  # Text with Chinese characters
        ]
        
        for pattern in safe_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return True, []
        
        # Check for malicious patterns (more precise blacklist)
        precise_malicious_patterns = [
            # Script injection
            r'<script[^>]*>.*?</script>',
            r'javascript:(?!void\(0\))',  # Exclude common void(0)
            r'on\w+\s*=\s*["\'][^"\'>]*["\']',  # Event handlers
            
            # Data URLs (more strict)
            r'data:(?!image/)[^,]*,',  # Exclude image data URLs
            
            # URL encoding (more precise, only check dangerous encodings)
            r'%(?:2[2-9]|[3-6][0-9a-fA-F]|7[0-9a-eA-E])',  # Dangerous URL encodings
            
            # XSS (more precise)
            r'<(?:iframe|object|embed|form)[^>]*>',
        ]
        
        for pattern in precise_malicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats.append(f"Malicious pattern detected: {pattern}")
        
        # Check for SQL injection patterns (more precise)
        precise_sql_patterns = [
            r'\b(?:union|select|insert|update|delete|drop)\s+(?:all\s+)?(?:distinct\s+)?\w+',
            r'(or|and)\s+\d+\s*=\s*\d+',
            r'[\'"]\s*(or|and)\s+[\'"]',
            r'--\s*$',  # SQL comments
            r'/\*.*?\*/',  # SQL block comments
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
                        threats.append(f"Malicious pattern in decoded content: {pattern}")
        except Exception:
            # If decoding fails, it might be suspicious
            threats.append("Suspicious encoding detected")
        
        # If threats detected but content looks like safe URL encoding, reduce threat level
        if threats and self._is_likely_safe_url_encoding(content):
            # Filter out URL encoding related threats
            threats = [t for t in threats if 'URL encoding' not in t and '%' not in t]
        
        return len(threats) == 0, threats
    
    def _is_likely_safe_url_encoding(self, content: str) -> bool:
        """Check if content is likely safe URL encoding"""
        # If content mainly contains URL encoded normal characters
        url_encoded_chars = re.findall(r'%[0-9a-fA-F]{2}', content)
        if url_encoded_chars:
            # Check if encoded characters are common safe characters
            safe_encoded_patterns = [
                r'%20',  # Space
                r'%2[1-9]',  # Common punctuation
                r'%3[0-9a-fA-F]',  # Numbers and some symbols
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
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\'>]*["\']', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove SQL injection patterns
        for pattern in self.SQL_INJECTION_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def get_model_limits(self, model_name: str) -> ModelTokenLimits:
        """Get model token limits"""
        logger.debug(f"Looking for token limits for model: '{model_name}'")
        logger.debug(f"Available model configurations: {list(self.model_limits.keys())}")
        
        # Try exact match
        if model_name in self.model_limits:
            logger.debug(f"Found exact match for model '{model_name}'")
            return self.model_limits[model_name]
        
        # Try fuzzy match with improved logic
        for key, limits in self.model_limits.items():
            # Check if model names match (case-insensitive)
            if key.lower() == model_name.lower():
                logger.debug(f"Found case-insensitive match: '{key}' for model '{model_name}'")
                return limits
            
            # Check if one is contained in the other
            if key.lower() in model_name.lower() or model_name.lower() in key.lower():
                logger.debug(f"Found fuzzy match: '{key}' for model '{model_name}'")
                return limits
            
            # Check for common model name variations
            # Remove common prefixes/suffixes and compare
            clean_key = key.lower().replace('-chat', '').replace('_chat', '').replace('-v3', '').replace('-v2', '').replace('-v1', '')
            clean_model = model_name.lower().replace('-chat', '').replace('_chat', '').replace('-v3', '').replace('-v2', '').replace('-v1', '')
            
            if clean_key == clean_model or clean_key in clean_model or clean_model in clean_key:
                logger.debug(f"Found normalized match: '{key}' (normalized: '{clean_key}') for model '{model_name}' (normalized: '{clean_model}')")
                return limits
        
        logger.warning(f"Token limit configuration not found for model '{model_name}' in available configurations: {list(self.model_limits.keys())}, using default configuration")
        return self.default_limits
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate text token count (simple estimation: 1 token ≈ 4 characters)"""
        # For Chinese, 1 character is approximately equal to 1 token
        # For English, 1 token is approximately equal to 4 characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(text) - chinese_chars
        return chinese_chars + (english_chars // 4)
    
    def chunk_text_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        """Chunk text by sentences"""
        # Split by sentences
        sentences = re.split(r'[.!?。！？]\s*', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence = sentence.strip() + "。"  # Add period
            sentence_tokens = self.estimate_tokens(sentence)
            current_tokens = self.estimate_tokens(current_chunk)
            
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_text_by_paragraphs(self, text: str, max_tokens: int) -> List[str]:
        """Chunk text by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            paragraph_tokens = self.estimate_tokens(paragraph)
            current_tokens = self.estimate_tokens(current_chunk)
            
            if current_tokens + paragraph_tokens <= max_tokens:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If a single paragraph is too long, chunk by sentences
                if paragraph_tokens > max_tokens:
                    sentence_chunks = self.chunk_text_by_sentences(paragraph, max_tokens)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def smart_chunk_content(self, content: str, model_name: str, chunk_strategy: str = "auto") -> List[str]:
        """Smart chunk content with security validation
        
        Args:
            content: Content to be chunked
            model_name: Model name
            chunk_strategy: Chunking strategy ("sentences", "paragraphs", "auto")
        
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
        
        # If content doesn't exceed limit, return directly
        if self.estimate_tokens(content) <= max_tokens:
            return [content]
        
        logger.info(f"Content exceeds token limit for model {model_name}, starting chunking process")
        
        if chunk_strategy == "auto":
            # Auto strategy: try paragraph chunking first, then sentence chunking if chunks are too large
            chunks = self.chunk_text_by_paragraphs(content, max_tokens)
            
            # Check if any chunks are still too large
            final_chunks = []
            for chunk in chunks:
                if self.estimate_tokens(chunk) > max_tokens:
                    sentence_chunks = self.chunk_text_by_sentences(chunk, max_tokens)
                    final_chunks.extend(sentence_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
        
        elif chunk_strategy == "paragraphs":
            return self.chunk_text_by_paragraphs(content, max_tokens)
        
        elif chunk_strategy == "sentences":
            return self.chunk_text_by_sentences(content, max_tokens)
        
        elif chunk_strategy == "aggressive":
            # Aggressive strategy: use much smaller chunks for extremely long content
            aggressive_max_tokens = min(max_tokens // 4, 2000)  # Use 1/4 of normal limit or 2000 tokens max
            logger.info(f"Using aggressive chunking with max {aggressive_max_tokens} tokens per chunk")
            
            # Try sentence chunking first with reduced limit
            chunks = self.chunk_text_by_sentences(content, aggressive_max_tokens)
            
            # If still too large, use character-based chunking
            final_chunks = []
            for chunk in chunks:
                if self.estimate_tokens(chunk) > aggressive_max_tokens:
                    # Character-based chunking as last resort
                    char_limit = aggressive_max_tokens * 3  # Rough estimate: 1 token ≈ 3-4 characters
                    chunk_parts = [chunk[i:i+char_limit] for i in range(0, len(chunk), char_limit)]
                    final_chunks.extend(chunk_parts)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks[:3]  # Return only first 3 chunks for aggressive strategy
        
        else:
            raise ValueError(f"Unsupported chunking strategy: {chunk_strategy}")
    
    def create_summary_prompt(self, content: str, summary_type: str = "comprehensive") -> str:
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
    
    def summarize_content(self, 
                         content: str, 
                         llm: BaseChatModel, 
                         model_name: str,
                         summary_type: str = "comprehensive") -> str:
        """Summarize content
        
        Args:
            content: Content to be summarized
            llm: Language model
            model_name: Model name
            summary_type: Summary type
        
        Returns:
            Summarized content
        """
        limits = self.get_model_limits(model_name)
        
        # If content doesn't exceed limit, can summarize directly
        if self.estimate_tokens(content) <= limits.safe_input_limit:
            prompt = self.create_summary_prompt(content, summary_type)
            messages = [HumanMessage(content=prompt)]
            
            try:
                from src.llms.error_handler import safe_llm_call
                response = safe_llm_call(
                    llm.invoke,
                    messages,
                    operation_name="Content Summarization",
                    context="Summarizing content within token limits",
                    max_retries=3,
                    enable_context_evaluation=True
                )
                return response.content
            except Exception as e:
                logger.error(f"Error occurred while summarizing content: {e}")
                return content  # Return original content
        
        # Content too long, need chunked summarization
        logger.info("Content too long, using chunked summarization strategy")
        chunks = self.smart_chunk_content(content, model_name)
        
        summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            
            prompt = self.create_summary_prompt(chunk, summary_type)
            messages = [HumanMessage(content=prompt)]
            
            try:
                from src.llms.error_handler import safe_llm_call
                response = safe_llm_call(
                    llm.invoke,
                    messages,
                    operation_name="Chunk Summarization",
                    context=f"Summarizing chunk {i+1}/{len(chunks)}",
                    max_retries=3,
                    enable_context_evaluation=True
                )
                summaries.append(response.content)
            except Exception as e:
                logger.error(f"Error occurred while summarizing chunk {i+1}: {e}")
                summaries.append(chunk)  # Use original content
        
        # Merge all summaries
        combined_summary = "\n\n".join(summaries)
        
        # If merged summary is still too long, summarize again
        if self.estimate_tokens(combined_summary) > limits.safe_input_limit:
            logger.info("Merged summary still too long, performing secondary summarization")
            return self.summarize_content(combined_summary, llm, model_name, "abstract")
        
        return combined_summary
    
    def process_search_results(self, 
                              search_results: List[Dict[str, Any]], 
                              llm: BaseChatModel,
                              model_name: str,
                              max_results: Optional[int] = None,
                              query: Optional[str] = None,
                              enable_smart_filtering: bool = True) -> str:
        """Process search results with intelligent filtering, chunking, summarization and security validation
        
        Args:
            search_results: List of search results
            llm: Language model
            model_name: Model name
            max_results: Maximum number of results
            query: Original user query for relevance filtering
            enable_smart_filtering: Whether to use LLM-based filtering
        
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
            title = result.get('title', 'No title')
            content = result.get('content', result.get('snippet', 'No content'))
            url = result.get('url', 'No link')
            
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
        
        # Use SearchResultFilter to determine if smart filtering should be enabled
        should_use_smart_filtering = False
        if enable_smart_filtering and query:
            from src.utils.search_result_filter import SearchResultFilter
            filter_instance = SearchResultFilter(self)
            should_use_smart_filtering = filter_instance.should_enable_smart_filtering(search_results, model_name)
        
        # Get model limits for token checking
        limits = self.get_model_limits(model_name)
        
        # Use smart filtering if enabled, query is provided, and threshold is met
        if enable_smart_filtering and query and should_use_smart_filtering:
            try:
                # Get threshold info for logging
                from src.utils.search_result_filter import SearchResultFilter
                filter_instance = SearchResultFilter(self)
                current_tokens = self.estimate_tokens(combined_results)
                smart_filtering_threshold = filter_instance.get_smart_filtering_threshold(model_name)
                logger.info(f"Using smart filtering for search results (tokens: {current_tokens}, threshold: {smart_filtering_threshold})")
                
                filtered_data = filter_instance.filter_search_results(
                    query=query,
                    search_results=search_results,
                    llm=llm,
                    model_name=model_name,
                    max_results=max_results
                )
                
                # Format filtered results
                filtered_formatted_results = filter_instance.format_filtered_results(filtered_data)
                
                # Check if the filtered results are within token limits
                if self.estimate_tokens(filtered_formatted_results) <= limits.safe_input_limit:
                    logger.info(f"Smart filtering successful: {filtered_data.get('total_filtered', 0)}/{filtered_data.get('total_original', 0)} results")
                    return filtered_formatted_results
                else:
                    logger.info("Filtered results still too long, applying additional summarization")
                    return self.summarize_content(filtered_formatted_results, llm, model_name, "key_points")
                    
            except Exception as e:
                logger.warning(f"Smart filtering failed: {e}, falling back to traditional processing")
                # Fall back to traditional processing with the already formatted results
        
        # Traditional processing (fallback or when smart filtering is disabled)
        # Use the already formatted results from above
        
        # Check if summarization is needed
        if self.estimate_tokens(combined_results) > limits.safe_input_limit:
            logger.info("Search results too long, starting intelligent summarization")
            return self.summarize_content(combined_results, llm, model_name, "key_points")
        
        # Final security check on combined results
        is_safe, threats = self.validate_input_security(combined_results)
        if not is_safe:
            logger.warning(f"Security threats detected in combined search results: {threats}")
            combined_results = self.sanitize_content(combined_results)
        
        return combined_results
    
    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe
        
        Args:
            url: URL to check
            
        Returns:
            True if URL appears safe
        """
        if not url or url == 'No link':
            return True
        
        # Check for suspicious URL patterns
        suspicious_patterns = [
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'file:',
            r'ftp://.*@',  # FTP with credentials
            r'\d+\.\d+\.\d+\.\d+',  # Raw IP addresses (potentially suspicious)
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True