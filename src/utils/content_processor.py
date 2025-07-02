# SPDX-License-Identifier: MIT
"""
Intelligent content chunking and summarization processing module
Provides content chunking, summarization and optimization based on different model token limits
"""

import logging
import re
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
    """Intelligent content processor"""
    
    def __init__(self, model_limits: Dict[str, ModelTokenLimits]):
        self.model_limits = model_limits
        self.default_limits = ModelTokenLimits(
            input_limit=4000,
            output_limit=1000,
            context_window=8000
        )
    
    def get_model_limits(self, model_name: str) -> ModelTokenLimits:
        """Get model token limits"""
        # Try exact match
        if model_name in self.model_limits:
            return self.model_limits[model_name]
        
        # Try fuzzy match
        for key, limits in self.model_limits.items():
            if key.lower() in model_name.lower() or model_name.lower() in key.lower():
                return limits
        
        logger.warning(f"Token limit configuration not found for model {model_name}, using default configuration")
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
        """Smart chunk content
        
        Args:
            content: Content to be chunked
            model_name: Model name
            chunk_strategy: Chunking strategy ("sentences", "paragraphs", "auto")
        
        Returns:
            List of chunked content
        """
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
                    max_retries=3
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
                    max_retries=3
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
                              max_results: Optional[int] = None) -> str:
        """Process search results with intelligent chunking and summarization
        
        Args:
            search_results: List of search results
            llm: Language model
            model_name: Model name
            max_results: Maximum number of results
        
        Returns:
            Processed search results text
        """
        if not search_results:
            return "No relevant search results found."
        
        # Limit number of search results
        if max_results:
            search_results = search_results[:max_results]
        
        # Format search results
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', result.get('snippet', 'No content'))
            url = result.get('url', 'No link')
            
            formatted_result = f"""Search Result {i}:
Title: {title}
Link: {url}
Content: {content}
"""
            formatted_results.append(formatted_result)
        
        combined_results = "\n\n".join(formatted_results)
        
        # Check if summarization is needed
        limits = self.get_model_limits(model_name)
        if self.estimate_tokens(combined_results) > limits.safe_input_limit:
            logger.info("Search results too long, starting intelligent summarization")
            return self.summarize_content(combined_results, llm, model_name, "key_points")
        
        return combined_results