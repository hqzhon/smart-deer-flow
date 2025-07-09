#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Request Processor Module

Provides robust request processing infrastructure for batch operations
with proper context management, error handling, and resource management.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Union
from uuid import uuid4

from src.server.chat_request import ChatRequest
from src.server.rag_request import RAGResourceRequest
from src.server.mcp_request import MCPServerMetadataRequest

logger = logging.getLogger(__name__)

# Context variables for request tracking
request_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('request_context', default=None)
connection_context: ContextVar[Optional[str]] = ContextVar('connection_context', default=None)


class RequestType(Enum):
    """Supported request types for batch processing."""
    CHAT = "chat"
    GENERATE_PPT = "generate_ppt"
    GENERATE_PROSE = "generate_prose"
    ENHANCE_PROMPT = "enhance_prompt"
    RAG_QUERY = "rag_query"
    MCP_REQUEST = "mcp_request"
    UNKNOWN = "unknown"


@dataclass
class BatchRequestData:
    """Standardized batch request data structure."""
    id: str
    type: RequestType
    payload: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self, max_age: float = 300.0) -> bool:
        """Check if request has expired (default: 5 minutes)."""
        return (time.time() - self.created_at) > max_age
    
    def should_retry(self) -> bool:
        """Check if request should be retried."""
        return self.retry_count < self.max_retries
    
    def get_retry_delay(self) -> float:
        """Calculate exponential backoff delay for retry."""
        return min(2 ** self.retry_count, 60.0)  # Max 60 seconds


class RequestContext:
    """Request context manager for proper resource and state management."""
    
    def __init__(self, request_data: BatchRequestData):
        self.request_data = request_data
        self.connection_id: Optional[str] = None
        self.start_time = time.time()
        self.metrics = {
            'processing_time': 0.0,
            'memory_usage': 0,
            'cache_hits': 0,
            'errors': []
        }
    
    async def __aenter__(self):
        """Enter context: set up request context and acquire resources."""
        # Set request context
        request_context.set({
            'request_id': self.request_data.id,
            'request_type': self.request_data.type.value,
            'start_time': self.start_time,
            'context': self.request_data.context
        })
        
        # Acquire connection
        try:
            self.connection_id = await self._acquire_connection()
            connection_context.set(self.connection_id)
            logger.debug(f"Acquired connection {self.connection_id} for request {self.request_data.id}")
        except Exception as e:
            logger.error(f"Failed to acquire connection for request {self.request_data.id}: {e}")
            raise
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context: clean up resources and log metrics."""
        # Calculate processing time
        self.metrics['processing_time'] = time.time() - self.start_time
        
        # Release connection
        if self.connection_id:
            try:
                await self._release_connection(self.connection_id)
                logger.debug(f"Released connection {self.connection_id} for request {self.request_data.id}")
            except Exception as e:
                logger.error(f"Failed to release connection {self.connection_id}: {e}")
        
        # Log metrics
        await self._log_metrics(exc_type is not None)
        
        # Clear context
        request_context.set(None)
        connection_context.set(None)
    
    async def _acquire_connection(self) -> str:
        """Acquire connection and return connection ID."""
        # Import here to avoid circular imports
        from src.server.app import _acquire_connection
        
        await _acquire_connection()
        return f"conn_{self.request_data.id}_{int(time.time())}"
    
    async def _release_connection(self, connection_id: str):
        """Release connection."""
        # Import here to avoid circular imports
        from src.server.app import _release_connection
        
        await _release_connection(connection_id)
    
    async def _log_metrics(self, has_error: bool):
        """Log request processing metrics."""
        logger.info(
            f"Request {self.request_data.id} completed",
            extra={
                'request_id': self.request_data.id,
                'request_type': self.request_data.type.value,
                'processing_time': self.metrics['processing_time'],
                'has_error': has_error,
                'retry_count': self.request_data.retry_count,
                'priority': self.request_data.priority
            }
        )


class RequestProcessor(ABC):
    """Abstract base class for request processors."""
    
    @abstractmethod
    async def process(self, payload: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Process the request payload.
        
        Args:
            payload: Request payload data
            context: Request context for resource management
            
        Returns:
            Dict containing processing result
        """
        pass
    
    @abstractmethod
    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """Validate request payload.
        
        Args:
            payload: Request payload to validate
            
        Returns:
            True if payload is valid, False otherwise
        """
        pass
    
    def get_processor_name(self) -> str:
        """Get processor name for logging."""
        return self.__class__.__name__


class ChatRequestProcessor(RequestProcessor):
    """Processor for chat requests."""
    
    async def process(self, payload: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Process chat request."""
        try:
            # Parse and validate chat request
            chat_request = ChatRequest(**payload)
            
            # Log processing start
            logger.debug(
                f"Processing chat request for thread {chat_request.thread_id}",
                extra={'request_id': context.request_data.id}
            )
            
            # Simulate chat processing (replace with actual implementation)
            result = await self._process_chat_request(chat_request, context)
            
            return {
                'type': 'chat_response',
                'thread_id': chat_request.thread_id,
                'result': result,
                'processed_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}", exc_info=True)
            raise
    
    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """Validate chat request payload."""
        try:
            ChatRequest(**payload)
            return True
        except Exception as e:
            logger.warning(f"Chat payload validation failed: {e}")
            return False
    
    async def _process_chat_request(self, request: ChatRequest, context: RequestContext) -> Dict[str, Any]:
        """Actual chat processing logic (placeholder)."""
        # This would contain the actual chat processing logic
        # For now, return a mock response
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'message': 'Chat request processed successfully',
            'message_count': len(request.messages),
            'debug_enabled': request.debug
        }


class RAGQueryProcessor(RequestProcessor):
    """Processor for RAG query requests."""
    
    async def process(self, payload: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Process RAG query request."""
        try:
            # Parse and validate RAG request
            rag_request = RAGResourceRequest(**payload)
            
            logger.debug(
                f"Processing RAG query: {rag_request.query}",
                extra={'request_id': context.request_data.id}
            )
            
            # Simulate RAG processing
            result = await self._process_rag_query(rag_request, context)
            
            return {
                'type': 'rag_response',
                'query': rag_request.query,
                'result': result,
                'processed_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"RAG processing failed: {e}", exc_info=True)
            raise
    
    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """Validate RAG query payload."""
        try:
            RAGResourceRequest(**payload)
            return True
        except Exception as e:
            logger.warning(f"RAG payload validation failed: {e}")
            return False
    
    async def _process_rag_query(self, request: RAGResourceRequest, context: RequestContext) -> Dict[str, Any]:
        """Actual RAG processing logic (placeholder)."""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        return {
            'documents': [],
            'query_processed': request.query,
            'relevance_score': 0.85
        }


class UnknownRequestProcessor(RequestProcessor):
    """Fallback processor for unknown request types."""
    
    async def process(self, payload: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Process unknown request type."""
        logger.warning(
            f"Processing unknown request type",
            extra={'request_id': context.request_data.id, 'payload_keys': list(payload.keys())}
        )
        
        return {
            'type': 'unknown_response',
            'message': 'Request type not recognized, processed as generic request',
            'payload_summary': {k: type(v).__name__ for k, v in payload.items()},
            'processed_at': time.time()
        }
    
    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """Always validate unknown payloads as true (best effort processing)."""
        return isinstance(payload, dict)


class RequestProcessorFactory:
    """Factory for creating request processors."""
    
    _processors = {
        RequestType.CHAT: ChatRequestProcessor(),
        RequestType.RAG_QUERY: RAGQueryProcessor(),
        RequestType.UNKNOWN: UnknownRequestProcessor(),
    }
    
    @classmethod
    def get_processor(cls, request_type: RequestType) -> RequestProcessor:
        """Get processor for request type.
        
        Args:
            request_type: Type of request to process
            
        Returns:
            Appropriate processor instance
        """
        processor = cls._processors.get(request_type)
        if not processor:
            logger.warning(f"No specific processor found for {request_type}, using unknown processor")
            return cls._processors[RequestType.UNKNOWN]
        return processor
    
    @classmethod
    def register_processor(cls, request_type: RequestType, processor: RequestProcessor):
        """Register a new processor for a request type.
        
        Args:
            request_type: Request type to handle
            processor: Processor instance
        """
        cls._processors[request_type] = processor
        logger.info(f"Registered processor {processor.get_processor_name()} for type {request_type}")


def parse_request_data(request_data: Dict[str, Any]) -> BatchRequestData:
    """Parse raw request data into standardized format.
    
    Args:
        request_data: Raw request data dictionary
        
    Returns:
        Parsed BatchRequestData instance
        
    Raises:
        ValueError: If request data is invalid
    """
    if not isinstance(request_data, dict):
        raise ValueError("Request data must be a dictionary")
    
    # Check required fields
    required_fields = ['id', 'type', 'payload']
    missing_fields = [field for field in required_fields if field not in request_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Parse request type
    try:
        request_type = RequestType(request_data['type'])
    except ValueError:
        logger.warning(f"Unknown request type: {request_data['type']}, using UNKNOWN")
        request_type = RequestType.UNKNOWN
    
    # Generate ID if not provided or invalid
    request_id = request_data.get('id')
    if not request_id or not isinstance(request_id, str):
        request_id = str(uuid4())
        logger.warning(f"Generated new request ID: {request_id}")
    
    return BatchRequestData(
        id=request_id,
        type=request_type,
        payload=request_data['payload'],
        context=request_data.get('context', {}),
        priority=request_data.get('priority', 1),
        timeout=request_data.get('timeout', 30.0),
        retry_count=request_data.get('retry_count', 0),
        max_retries=request_data.get('max_retries', 3)
    )


async def record_success_metrics(request: BatchRequestData, context: RequestContext):
    """Record metrics for successful request processing.
    
    Args:
        request: Processed request data
        context: Request context with metrics
    """
    # This could integrate with monitoring systems like Prometheus
    logger.info(
        f"Request {request.id} processed successfully",
        extra={
            'request_id': request.id,
            'request_type': request.type.value,
            'processing_time': context.metrics['processing_time'],
            'priority': request.priority,
            'retry_count': request.retry_count
        }
    )


async def record_failure_metrics(request: BatchRequestData, context: RequestContext, error: str):
    """Record metrics for failed request processing.
    
    Args:
        request: Failed request data
        context: Request context with metrics
        error: Error message
    """
    logger.error(
        f"Request {request.id} processing failed",
        extra={
            'request_id': request.id,
            'request_type': request.type.value,
            'processing_time': context.metrics['processing_time'],
            'error': error,
            'retry_count': request.retry_count,
            'priority': request.priority
        }
    )


async def schedule_retry(request: BatchRequestData, error: str) -> Dict[str, Any]:
    """Schedule request for retry with exponential backoff.
    
    Args:
        request: Request to retry
        error: Error that caused the retry
        
    Returns:
        Dict with retry scheduling information
    """
    request.retry_count += 1
    delay = request.get_retry_delay()
    
    logger.info(
        f"Scheduling retry for request {request.id} in {delay}s (attempt {request.retry_count})",
        extra={
            'request_id': request.id,
            'retry_count': request.retry_count,
            'retry_delay': delay,
            'error': error
        }
    )
    
    # In a real implementation, this would re-queue the request with delay
    # For now, just return the retry information
    return {
        'status': 'retry_scheduled',
        'request_id': request.id,
        'retry_count': request.retry_count,
        'retry_delay': delay,
        'error': error,
        'next_attempt_at': time.time() + delay
    }