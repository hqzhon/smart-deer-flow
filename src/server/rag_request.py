# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import re
from pydantic import BaseModel, Field, validator

from src.rag.retriever import Resource


class RAGConfigResponse(BaseModel):
    """Response model for RAG config."""

    provider: str | None = Field(
        None, description="The provider of the RAG, default is ragflow"
    )


class RAGResourceRequest(BaseModel):
    """Request model for RAG resource."""

    query: str | None = Field(
        None, description="The query of the resource need to be searched", max_length=1000
    )
    
    @validator('query')
    def validate_query(cls, v):
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
                    raise ValueError('Potentially unsafe content in query')
            # Check for SQL injection patterns
            sql_patterns = [
                r'\bunion\b.*\bselect\b',
                r'\bselect\b.*\bfrom\b',
                r'\binsert\b.*\binto\b',
                r'\bupdate\b.*\bset\b',
                r'\bdelete\b.*\bfrom\b',
                r'\bdrop\b.*\btable\b',
                r'--',
                r'/\*.*\*/'
            ]
            for pattern in sql_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially unsafe SQL pattern in query')
        return v


class RAGResourcesResponse(BaseModel):
    """Response model for RAG resources."""

    resources: list[Resource] = Field(..., description="The resources of the RAG")
