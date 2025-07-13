# SPDX-License-Identifier: MIT
"""
Accurate token counting module using tiktoken
Provides precise token counting for different models and content types
"""

import logging
import tiktoken
from typing import Dict, List
from dataclasses import dataclass
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


@dataclass
class TokenCountResult:
    """Token count result with detailed breakdown"""

    total_tokens: int
    input_tokens: int
    output_tokens: int = 0
    model_name: str = ""
    encoding_name: str = ""

    def __str__(self) -> str:
        return f"TokenCount(total={self.total_tokens}, input={self.input_tokens}, output={self.output_tokens}, model={self.model_name})"


class TokenCounter:
    """Accurate token counter using tiktoken"""

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        # OpenAI models
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        # DeepSeek models (use cl100k_base as default)
        "deepseek-chat": "cl100k_base",
        "deepseek-reasoner": "cl100k_base",
        "deepseek-coder": "cl100k_base",
        # Claude models (approximate with cl100k_base)
        "claude-3-opus": "cl100k_base",
        "claude-3-sonnet": "cl100k_base",
        "claude-3-haiku": "cl100k_base",
        "claude-3-5-sonnet": "cl100k_base",
        # Other models
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }

    def __init__(self):
        self._encoders: Dict[str, tiktoken.Encoding] = {}
        self._load_default_encoders()

    def _load_default_encoders(self):
        """Pre-load commonly used encoders"""
        try:
            # Load most common encodings
            common_encodings = ["cl100k_base", "o200k_base", "p50k_base"]
            for encoding_name in common_encodings:
                try:
                    self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
                    logger.debug(f"Loaded encoding: {encoding_name}")
                except Exception as e:
                    logger.warning(f"Failed to load encoding {encoding_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to load default encoders: {e}")

    def get_encoding_for_model(self, model_name: str) -> str:
        """Get the appropriate encoding for a model"""
        # Direct match
        if model_name in self.MODEL_ENCODINGS:
            return self.MODEL_ENCODINGS[model_name]

        # Fuzzy match for model variants
        model_lower = model_name.lower()
        for known_model, encoding in self.MODEL_ENCODINGS.items():
            if known_model.lower() in model_lower or model_lower in known_model.lower():
                logger.debug(
                    f"Using encoding {encoding} for model {model_name} (matched with {known_model})"
                )
                return encoding

        # Default encoding for unknown models
        logger.warning(
            f"Unknown model {model_name}, using default encoding cl100k_base"
        )
        return "cl100k_base"

    def get_encoder(self, model_name: str) -> tiktoken.Encoding:
        """Get encoder for a specific model"""
        encoding_name = self.get_encoding_for_model(model_name)

        if encoding_name not in self._encoders:
            try:
                self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
                logger.debug(f"Loaded new encoding: {encoding_name}")
            except Exception as e:
                logger.error(f"Failed to load encoding {encoding_name}: {e}")
                # Fallback to cl100k_base
                if "cl100k_base" not in self._encoders:
                    self._encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
                return self._encoders["cl100k_base"]

        return self._encoders[encoding_name]

    def count_tokens(self, text: str, model_name: str = "gpt-4") -> TokenCountResult:
        """Count tokens in text using appropriate encoder"""
        if not text:
            return TokenCountResult(
                total_tokens=0,
                input_tokens=0,
                model_name=model_name,
                encoding_name=self.get_encoding_for_model(model_name),
            )

        try:
            encoder = self.get_encoder(model_name)
            tokens = encoder.encode(text)
            token_count = len(tokens)

            return TokenCountResult(
                total_tokens=token_count,
                input_tokens=token_count,
                model_name=model_name,
                encoding_name=encoder.name,
            )
        except Exception as e:
            logger.error(f"Failed to count tokens for model {model_name}: {e}")
            # Fallback to character-based estimation
            estimated_tokens = self._fallback_estimate(text)
            return TokenCountResult(
                total_tokens=estimated_tokens,
                input_tokens=estimated_tokens,
                model_name=model_name,
                encoding_name="fallback_estimation",
            )

    def count_message_tokens(
        self, messages: List[BaseMessage], model_name: str = "gpt-4"
    ) -> TokenCountResult:
        """Count tokens in a list of messages"""
        total_tokens = 0

        try:
            encoder = self.get_encoder(model_name)

            for message in messages:
                # Count message content
                content_tokens = len(encoder.encode(message.content))

                # Add overhead for message formatting
                # Different models have different overhead
                if "gpt-4" in model_name.lower():
                    # GPT-4 message overhead: ~3 tokens per message
                    message_overhead = 3
                elif "gpt-3.5" in model_name.lower():
                    # GPT-3.5 message overhead: ~4 tokens per message
                    message_overhead = 4
                else:
                    # Default overhead for other models
                    message_overhead = 3

                total_tokens += content_tokens + message_overhead

            # Add conversation overhead (varies by model)
            if "gpt-4" in model_name.lower():
                conversation_overhead = 3
            elif "gpt-3.5" in model_name.lower():
                conversation_overhead = 3
            else:
                conversation_overhead = 2

            total_tokens += conversation_overhead

            return TokenCountResult(
                total_tokens=total_tokens,
                input_tokens=total_tokens,
                model_name=model_name,
                encoding_name=encoder.name,
            )

        except Exception as e:
            logger.error(f"Failed to count message tokens for model {model_name}: {e}")
            # Fallback: count all message content as plain text
            all_content = "\n".join([msg.content for msg in messages])
            return self.count_tokens(all_content, model_name)

    def _fallback_estimate(self, text: str) -> int:
        """Fallback token estimation when tiktoken fails"""
        # Conservative estimation similar to the original method
        chinese_chars = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
        english_chars = len(text) - chinese_chars

        # Conservative multipliers
        chinese_tokens = int(chinese_chars * 1.2)
        english_tokens = int(english_chars / 3)

        total_tokens = chinese_tokens + english_tokens
        safety_buffer = int(total_tokens * 0.15)  # 15% safety buffer for fallback

        return total_tokens + safety_buffer

    def check_token_limit(
        self, text: str, model_name: str, max_tokens: int, safety_margin: float = 0.9
    ) -> tuple[bool, TokenCountResult]:
        """Check if text exceeds token limit

        Args:
            text: Text to check
            model_name: Model name
            max_tokens: Maximum allowed tokens
            safety_margin: Safety margin (0.9 = use 90% of limit)

        Returns:
            Tuple of (is_within_limit, token_count_result)
        """
        token_result = self.count_tokens(text, model_name)
        safe_limit = int(max_tokens * safety_margin)
        is_within_limit = token_result.total_tokens <= safe_limit

        if not is_within_limit:
            logger.warning(
                f"Token limit exceeded: {token_result.total_tokens} > {safe_limit} "
                f"(limit: {max_tokens}, margin: {safety_margin:.1%})"
            )

        return is_within_limit, token_result

    def estimate_output_tokens(
        self, input_text: str, model_name: str, compression_ratio: float = 0.3
    ) -> int:
        """Estimate output tokens based on input

        Args:
            input_text: Input text
            model_name: Model name
            compression_ratio: Expected compression ratio for summaries

        Returns:
            Estimated output tokens
        """
        input_tokens = self.count_tokens(input_text, model_name).total_tokens
        return int(input_tokens * compression_ratio)

    def get_available_models(self) -> List[str]:
        """Get list of supported models"""
        return list(self.MODEL_ENCODINGS.keys())

    def get_available_encodings(self) -> List[str]:
        """Get list of available encodings"""
        return list(set(self.MODEL_ENCODINGS.values()))


# Global token counter instance
_token_counter = None


def get_token_counter() -> TokenCounter:
    """Get global token counter instance"""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


# Convenience functions
def count_tokens(text: str, model_name: str = "gpt-4") -> TokenCountResult:
    """Count tokens in text"""
    return get_token_counter().count_tokens(text, model_name)


def count_message_tokens(
    messages: List[BaseMessage], model_name: str = "gpt-4"
) -> TokenCountResult:
    """Count tokens in messages"""
    return get_token_counter().count_message_tokens(messages, model_name)


def check_token_limit(
    text: str, model_name: str, max_tokens: int, safety_margin: float = 0.9
) -> tuple[bool, TokenCountResult]:
    """Check if text exceeds token limit"""
    return get_token_counter().check_token_limit(
        text, model_name, max_tokens, safety_margin
    )
