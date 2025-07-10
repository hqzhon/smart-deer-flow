from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import AIMessage

class BaseLLMErrorHandler(ABC):
    """Abstract base class for LLM error handlers."""

    @abstractmethod
    def classify_error(self, error_message: str) -> str:
        """Classify error type based on error message."""
        pass

    @abstractmethod
    def should_skip_error(self, error_type: str) -> bool:
        """Determine if this error should be skipped."""
        pass

    @abstractmethod
    def should_retry_error(self, error_type: str) -> bool:
        """Determine if this error should be retried."""
        pass

    @abstractmethod
    def is_fatal_error(self, error_type: str) -> bool:
        """Determine if this is a fatal error."""
        pass

    @abstractmethod
    def get_fallback_response(self, error_type: str, context: str = "") -> AIMessage:
        """Get fallback response for error."""
        pass

    @abstractmethod
    def needs_smart_processing(self, error_type: str) -> bool:
        """Determine if smart processing is needed."""
        pass

    @abstractmethod
    def handle_error(
        self, error: Exception, context: str = "", operation_name: str = "LLM Operation"
    ) -> tuple[bool, Optional[AIMessage], bool]:
        """Handle LLM error.

        Args:
            error: Exception object
            context: Error context
            operation_name: Operation name

        Returns:
            tuple: (should skip error, fallback response, needs smart processing)
        """
        pass
