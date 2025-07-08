# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union
import re

from pydantic import BaseModel, Field, validator, HttpUrl

from src.rag.retriever import Resource
from src.config.report_style import ReportStyle


class ContentItem(BaseModel):
    type: str = Field(..., description="The type of content (text, image, etc.)", pattern=r"^(text|image)$")
    text: Optional[str] = Field(None, description="The text content if type is 'text'", max_length=50000)
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'", max_length=2048
    )
    
    @validator('image_url')
    def validate_image_url(cls, v):
        if v is not None:
            # Check for valid URL format
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
                r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(v):
                raise ValueError('Invalid URL format')
            # Check for suspicious patterns
            suspicious_patterns = [
                r'javascript:', r'data:', r'vbscript:', r'file:', r'ftp:'
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially unsafe URL scheme')
        return v
    
    @validator('text')
    def validate_text_content(cls, v):
        if v is not None:
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>'
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially unsafe content detected')
        return v


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)", pattern=r"^(user|assistant|system)$"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )
    
    @validator('content')
    def validate_content(cls, v):
        if isinstance(v, str):
            if len(v) > 100000:  # 100KB limit for string content
                raise ValueError('Content too long')
            # Check for suspicious patterns in string content
            suspicious_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>'
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially unsafe content detected')
        elif isinstance(v, list):
            if len(v) > 50:  # Limit number of content items
                raise ValueError('Too many content items')
        return v


class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(
        [], description="History of messages between the user and the assistant"
    )
    resources: Optional[List[Resource]] = Field(
        [], description="Resources to be used for the research"
    )
    debug: Optional[bool] = Field(False, description="Whether to enable debug logging")
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier", max_length=100
    )
    max_plan_iterations: Optional[int] = Field(
        1, description="The maximum number of plan iterations", ge=1, le=10
    )
    max_step_num: Optional[int] = Field(
        3, description="The maximum number of steps in a plan", ge=1, le=20
    )
    max_search_results: Optional[int] = Field(
        3, description="The maximum number of search results", ge=1, le=50
    )
    auto_accepted_plan: Optional[bool] = Field(
        False, description="Whether to automatically accept the plan"
    )
    interrupt_feedback: Optional[str] = Field(
        None, description="Interrupt feedback from the user on the plan", max_length=10000
    )
    mcp_settings: Optional[dict] = Field(
        None, description="MCP settings for the chat request"
    )
    enable_background_investigation: Optional[bool] = Field(
        True, description="Whether to get background investigation before plan"
    )
    report_style: Optional[ReportStyle] = Field(
        ReportStyle.ACADEMIC, description="The style of the report"
    )
    enable_deep_thinking: Optional[bool] = Field(
        False, description="Whether to enable deep thinking"
    )
    enable_collaboration: Optional[bool] = Field(
        True, description="Whether to enable collaboration features (role bidding, human loop, conflict resolution)"
    )
    enable_parallel_execution: Optional[bool] = Field(
        True, description="Whether to enable parallel execution of research steps"
    )
    max_parallel_tasks: Optional[int] = Field(
        2, description="Maximum number of parallel tasks to execute simultaneously", ge=1, le=10
    )
    
    @validator('messages')
    def validate_messages(cls, v):
        if v and len(v) > 1000:  # Limit number of messages
            raise ValueError('Too many messages in conversation history')
        return v
    
    @validator('resources')
    def validate_resources(cls, v):
        if v and len(v) > 100:  # Limit number of resources
            raise ValueError('Too many resources specified')
        return v
    
    @validator('thread_id')
    def validate_thread_id(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Thread ID contains invalid characters')
        return v
    
    @validator('interrupt_feedback')
    def validate_interrupt_feedback(cls, v):
        if v is not None:
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*='
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially unsafe content in feedback')
        return v


class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech", max_length=10000)
    voice_type: Optional[str] = Field(
        "BV700_V2_streaming", description="The voice type to use", max_length=50
    )
    encoding: Optional[str] = Field("mp3", description="The audio encoding format", pattern=r"^(mp3|wav|ogg)$")
    speed_ratio: Optional[float] = Field(1.0, description="Speech speed ratio", ge=0.1, le=3.0)
    volume_ratio: Optional[float] = Field(1.0, description="Speech volume ratio", ge=0.1, le=2.0)
    pitch_ratio: Optional[float] = Field(1.0, description="Speech pitch ratio", ge=0.1, le=2.0)
    text_type: Optional[str] = Field("plain", description="Text type (plain or ssml)", pattern=r"^(plain|ssml)$")
    with_frontend: Optional[int] = Field(
        1, description="Whether to use frontend processing", ge=0, le=1
    )
    frontend_type: Optional[str] = Field("unitTson", description="Frontend type", max_length=50)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*='
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially unsafe content in text')
        return v


class GeneratePodcastRequest(BaseModel):
    content: str = Field(..., description="The content of the podcast", max_length=100000)
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*='
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially unsafe content detected')
        return v


class GeneratePPTRequest(BaseModel):
    content: str = Field(..., description="The content of the ppt", max_length=100000)
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*='
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially unsafe content detected')
        return v


class GenerateProseRequest(BaseModel):
    prompt: str = Field(..., description="The content of the prose", max_length=50000)
    option: str = Field(..., description="The option of the prose writer", max_length=100)
    command: Optional[str] = Field(
        "", description="The user custom command of the prose writer", max_length=1000
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*='
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially unsafe content in prompt')
        return v
    
    @validator('option')
    def validate_option(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Option contains invalid characters')
        return v
    
    @validator('command')
    def validate_command(cls, v):
        if v:
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*='
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially unsafe content in command')
        return v


class EnhancePromptRequest(BaseModel):
    prompt: str = Field(..., description="The original prompt to enhance", max_length=50000)
    context: Optional[str] = Field(
        "", description="Additional context about the intended use", max_length=10000
    )
    report_style: Optional[str] = Field(
        "academic", description="The style of the report", pattern=r"^(academic|business|casual|technical)$"
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*='
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially unsafe content in prompt')
        return v
    
    @validator('context')
    def validate_context(cls, v):
        if v:
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*='
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially unsafe content in context')
        return v
