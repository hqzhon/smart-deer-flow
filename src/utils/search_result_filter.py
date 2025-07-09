# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_core.language_models.base import BaseLanguageModel

from src.llms.error_handler import safe_llm_call
from src.utils.content_processor import ContentProcessor

logger = logging.getLogger(__name__)


class SearchResultFilter:
    """Smart search result filter that uses LLM to filter and merge relevant information"""
    
    # Smart filtering threshold configuration
    SMART_FILTERING_THRESHOLD_RATIO = 0.6  # 60% threshold
    DEFAULT_BATCH_SIZE = 3
    MAX_SECONDARY_RESULTS = 5
    
    def __init__(self, content_processor: ContentProcessor):
        self.content_processor = content_processor
        self.filter_prompt_template = self._create_filter_prompt_template()
        
    def _create_filter_prompt_template(self) -> str:
        """Create the filter prompt template"""
        return """
You are a professional information filtering assistant. Please filter the most relevant information from the following search results based on the user query.

User Query: {query}

Search Results:
{search_results}

Please filter information according to the following requirements:
1. Only keep information directly related to the user query
2. Remove duplicate or similar content
3. Preserve important facts, data, and reference sources
4. Sort by relevance, with the most relevant first
5. If a search result is completely irrelevant, ignore it completely

Please return the filtered results in JSON format:
{{
  "filtered_results": [
    {{
      "title": "Filtered title",
      "content": "Filtered relevant content",
      "url": "Original URL",
      "relevance_score": 0.9,
      "key_points": ["Key point 1", "Key point 2"]
    }}
  ],
  "summary": "Brief summary of filtered results",
  "total_filtered": 3,
  "total_original": 5
}}
"""
        
    def should_enable_smart_filtering(self, search_results: List[Dict[str, Any]], model_name: str) -> bool:
        """Determine whether smart filtering should be enabled
        
        Args:
            search_results: List of search results
            model_name: Model name
            
        Returns:
            bool: Whether smart filtering should be enabled
        """
        if not search_results:
            return False
            
        # Format search results to estimate token count
        formatted_content = self._format_search_results(search_results)
        
        limits = self.content_processor.get_model_limits(model_name)
        
        # Use accurate token counting
        content_within_limit, token_result = self.content_processor.check_content_token_limit(
            formatted_content, model_name, limits.input_limit, limits.safety_margin
        )
        
        # Calculate smart filtering threshold
        smart_filtering_threshold = int(limits.input_limit * self.SMART_FILTERING_THRESHOLD_RATIO)
        
        logger.debug(
            f"Token estimation: {token_result.total_tokens}, "
            f"smart filtering threshold: {smart_filtering_threshold}"
        )
        
        return token_result.total_tokens > smart_filtering_threshold
        
    def get_smart_filtering_threshold(self, model_name: str) -> int:
        """Get smart filtering threshold
        
        Args:
            model_name: Model name
            
        Returns:
            int: Smart filtering threshold (token count)
        """
        limits = self.content_processor.get_model_limits(model_name)
        return int(limits.input_limit * self.SMART_FILTERING_THRESHOLD_RATIO)
    
    def filter_search_results(self, 
                            query: str,
                            search_results: List[Dict[str, Any]], 
                            llm: BaseLanguageModel,
                            model_name: str,
                            max_results: Optional[int] = None) -> Dict[str, Any]:
        """Filter search results using LLM
        
        Args:
            query: User query
            search_results: Original search results
            llm: Language model
            model_name: Model name
            max_results: Maximum number of results
            
        Returns:
            Dictionary of filtered results
        """
        if not search_results:
            return self._create_empty_result()
        
        # Limit search results count
        if max_results:
            search_results = search_results[:max_results]
        
        logger.info(f"Starting to filter {len(search_results)} search results")
        
        # Check if batch processing is needed
        formatted_results = self._format_search_results(search_results)
        filter_prompt = self.filter_prompt_template.format(
            query=query,
            search_results=formatted_results
        )
        
        limits = self.content_processor.get_model_limits(model_name)
        
        # Use accurate token counting for filter prompt
        prompt_within_limit, prompt_token_result = self.content_processor.check_content_token_limit(
            filter_prompt, model_name, limits.safe_input_limit, limits.safety_margin
        )
        
        if not prompt_within_limit:
            logger.info(
                f"Search results too long ({prompt_token_result.total_tokens} > {limits.safe_input_limit}), "
                "using batch filtering strategy"
            )
            return self._batch_filter_results(query, search_results, llm, model_name)
        
        # Direct filtering
        return self._single_filter_call(filter_prompt, llm, len(search_results))
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            "filtered_results": [],
            "summary": "No relevant search results found",
            "total_filtered": 0,
            "total_original": 0
        }
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results as text"""
        formatted = []
        for i, result in enumerate(search_results, 1):
            title = self.content_processor.sanitize_content(str(result.get('title', 'No title')))
            content = self.content_processor.sanitize_content(str(result.get('content', result.get('snippet', 'No content'))))
            url = self.content_processor.sanitize_content(str(result.get('url', 'No link')))
            
            formatted_result = f"""Result {i}:
Title: {title}
Link: {url}
Content: {content}
---"""
            formatted.append(formatted_result)
        
        return "\n\n".join(formatted)
    
    def _single_filter_call(self, filter_prompt: str, llm: BaseLanguageModel, original_count: int) -> Dict[str, Any]:
        """Single LLM call for filtering"""
        try:
            messages = [HumanMessage(content=filter_prompt)]
            response = safe_llm_call(
                llm.invoke,
                messages,
                operation_name="search_result_filtering",
                context="Filter and merge search results",
                max_retries=3,
    
            )
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            filtered_data = self._parse_json_response(response_content)
            
            if filtered_data and 'filtered_results' in filtered_data:
                logger.info(f"Successfully filtered: {len(filtered_data['filtered_results'])}/{original_count} results")
                return filtered_data
            
            # Fallback if parsing fails
            return {
                "filtered_results": [],
                "summary": response_content,
                "total_filtered": 0,
                "total_original": original_count
            }
            
        except Exception as e:
            logger.error(f"Search result filtering failed: {e}")
            return {
                "filtered_results": [],
                "summary": f"Error occurred during filtering: {str(e)}",
                "total_filtered": 0,
                "total_original": original_count,
                "error": str(e)
            }
    
    def _parse_json_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response with fallback strategies"""
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*({.*?})\s*```', response_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON-like structure
        json_match = re.search(r'{[^{}]*"filtered_results"[^{}]*}', response_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse JSON response: {response_content[:200]}...")
        return None
    
    def _batch_filter_results(self, 
                            query: str,
                            search_results: List[Dict[str, Any]], 
                            llm: BaseLanguageModel,
                            model_name: str) -> Dict[str, Any]:
        """Filter search results in batches"""
        logger.info("Using batch filtering strategy")
        
        # Split search results into small batches
        batches = [search_results[i:i + self.DEFAULT_BATCH_SIZE] 
                  for i in range(0, len(search_results), self.DEFAULT_BATCH_SIZE)]
        
        all_filtered_results = []
        batch_summaries = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            
            formatted_batch = self._format_search_results(batch)
            batch_prompt = self.filter_prompt_template.format(
                query=query,
                search_results=formatted_batch
            )
            
            batch_result = self._single_filter_call(batch_prompt, llm, len(batch))
            
            if batch_result.get('filtered_results'):
                all_filtered_results.extend(batch_result['filtered_results'])
            
            if batch_result.get('summary'):
                batch_summaries.append(batch_result['summary'])
        
        # Secondary filtering if too many results
        if len(all_filtered_results) > self.MAX_SECONDARY_RESULTS:
            logger.info("Too many filtered results, applying secondary filtering")
            all_filtered_results = self._apply_secondary_filter(query, all_filtered_results, llm)
        
        combined_summary = "\n\n".join(batch_summaries) if batch_summaries else "Batch filtering completed"
        
        return {
            "filtered_results": all_filtered_results,
            "summary": combined_summary,
            "total_filtered": len(all_filtered_results),
            "total_original": len(search_results)
        }
    
    def _apply_secondary_filter(self, 
                              query: str,
                              filtered_results: List[Dict[str, Any]], 
                              llm: BaseLanguageModel) -> List[Dict[str, Any]]:
        """Apply secondary filtering to already filtered results"""
        try:
            results_text = "\n\n".join([
                f"Result {i+1}:\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')}\nRelevance: {r.get('relevance_score', 'N/A')}"
                for i, r in enumerate(filtered_results)
            ])
            
            secondary_prompt = f"""
User Query: {query}

The following are already filtered results. Please select the {self.MAX_SECONDARY_RESULTS} most relevant results:

{results_text}

Please return the most relevant result numbers (1-{len(filtered_results)}), separated by commas, e.g.: 1,3,5,7,9
"""
            
            messages = [HumanMessage(content=secondary_prompt)]
            response = safe_llm_call(
                llm.invoke,
                messages,
                operation_name="secondary_filtering",
                context="Select most relevant content from filtered results",
                max_retries=2,
    
            )
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse selected result numbers
            numbers = re.findall(r'\d+', response_content)
            selected_indices = [int(n) - 1 for n in numbers if 1 <= int(n) <= len(filtered_results)]
            
            if selected_indices:
                return [filtered_results[i] for i in selected_indices[:self.MAX_SECONDARY_RESULTS]]
            
        except Exception as e:
            logger.warning(f"Secondary filtering failed: {e}")
        
        # If secondary filtering fails, return top results
        return filtered_results[:self.MAX_SECONDARY_RESULTS]
    
    def format_filtered_results(self, filtered_data: Dict[str, Any]) -> str:
        """Format filtered results as text"""
        if not filtered_data.get('filtered_results'):
            return filtered_data.get('summary', 'No relevant information found')
        
        formatted_parts = []
        
        # Add summary
        if filtered_data.get('summary'):
            formatted_parts.append(f"## Search Results Summary\n\n{filtered_data['summary']}\n")
        
        # Add filtered results
        formatted_parts.append("## Relevant Information\n")
        
        for i, result in enumerate(filtered_data['filtered_results'], 1):
            title = result.get('title', f'Result {i}')
            content = result.get('content', '')
            url = result.get('url', '')
            relevance = result.get('relevance_score', 'N/A')
            
            result_text = f"### {i}. {title}\n\n"
            if content:
                result_text += f"{content}\n\n"
            if url and url != 'No link':
                result_text += f"**Source**: {url}\n\n"
            if relevance != 'N/A':
                result_text += f"**Relevance**: {relevance}\n\n"
            
            formatted_parts.append(result_text)
        
        # Add statistics
        total_filtered = filtered_data.get('total_filtered', 0)
        total_original = filtered_data.get('total_original', 0)
        formatted_parts.append(f"---\n\n*Filtering Statistics: {total_filtered} relevant results filtered from {total_original} total results*")
        
        return "\n".join(formatted_parts)