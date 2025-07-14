# -*- coding: utf-8 -*-
"""
Token estimation utilities for content processing.
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TokenEstimate:
    """Token estimation result."""
    token_count: int
    character_count: int
    word_count: int
    estimated_cost: Optional[float] = None
    model_used: Optional[str] = None


class TokenEstimator:
    """Estimates token counts for various content types."""
    
    # Rough token-to-character ratios for different models
    TOKEN_RATIOS = {
        'gpt-4': 4.0,  # ~4 characters per token
        'gpt-3.5-turbo': 4.0,
        'claude': 3.5,
        'default': 4.0
    }
    
    def __init__(self, model: str = 'default'):
        """Initialize token estimator.
        
        Args:
            model: Model name for token estimation
        """
        self.model = model
        self.ratio = self.TOKEN_RATIOS.get(model, self.TOKEN_RATIOS['default'])
    
    def estimate_tokens(self, text: str) -> TokenEstimate:
        """Estimate token count for given text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            TokenEstimate with counts and estimates
        """
        if not text:
            return TokenEstimate(
                token_count=0,
                character_count=0,
                word_count=0,
                model_used=self.model
            )
        
        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        
        # Estimate tokens based on character count and model ratio
        estimated_tokens = int(char_count / self.ratio)
        
        # Adjust for special tokens and formatting
        estimated_tokens = self._adjust_for_formatting(text, estimated_tokens)
        
        return TokenEstimate(
            token_count=estimated_tokens,
            character_count=char_count,
            word_count=word_count,
            model_used=self.model
        )
    
    def estimate_tokens_batch(self, texts: list) -> Dict[int, TokenEstimate]:
        """Estimate tokens for multiple texts.
        
        Args:
            texts: List of texts to estimate
            
        Returns:
            Dictionary mapping index to TokenEstimate
        """
        return {i: self.estimate_tokens(text) for i, text in enumerate(texts)}
    
    def _adjust_for_formatting(self, text: str, base_estimate: int) -> int:
        """Adjust token estimate based on text formatting.
        
        Args:
            text: Original text
            base_estimate: Base token estimate
            
        Returns:
            Adjusted token estimate
        """
        # Count special formatting that might affect tokenization
        code_blocks = len(re.findall(r'```[\s\S]*?```', text))
        inline_code = len(re.findall(r'`[^`]+`', text))
        urls = len(re.findall(r'https?://[^\s]+', text))
        
        # Adjust estimate (these are rough heuristics)
        adjustment = 0
        adjustment += code_blocks * 5  # Code blocks tend to use more tokens
        adjustment += inline_code * 2  # Inline code adjustment
        adjustment += urls * 3  # URLs can be tokenized differently
        
        return max(1, base_estimate + adjustment)
    
    def estimate_cost(self, token_count: int, model: Optional[str] = None) -> float:
        """Estimate cost based on token count.
        
        Args:
            token_count: Number of tokens
            model: Model name (uses instance model if not provided)
            
        Returns:
            Estimated cost in USD
        """
        model = model or self.model
        
        # Rough cost estimates per 1K tokens (as of 2024)
        cost_per_1k = {
            'gpt-4': 0.03,
            'gpt-3.5-turbo': 0.002,
            'claude': 0.008,
            'default': 0.01
        }
        
        rate = cost_per_1k.get(model, cost_per_1k['default'])
        return (token_count / 1000) * rate