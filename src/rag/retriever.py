# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import abc
import re
from pydantic import BaseModel, Field, validator


class Chunk:
    content: str
    similarity: float

    def __init__(self, content: str, similarity: float):
        self.content = content
        self.similarity = similarity


class Document:
    """
    Document is a class that represents a document.
    """

    id: str
    url: str | None = None
    title: str | None = None
    chunks: list[Chunk] = []

    def __init__(
        self,
        id: str,
        url: str | None = None,
        title: str | None = None,
        chunks: list[Chunk] = [],
    ):
        self.id = id
        self.url = url
        self.title = title
        self.chunks = chunks

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "content": "\n\n".join([chunk.content for chunk in self.chunks]),
        }
        if self.url:
            d["url"] = self.url
        if self.title:
            d["title"] = self.title
        return d


class Resource(BaseModel):
    """
    Resource is a class that represents a resource.
    """

    uri: str = Field(..., description="The URI of the resource", max_length=2048)
    title: str = Field(..., description="The title of the resource", max_length=500)
    description: str | None = Field(
        "", description="The description of the resource", max_length=2000
    )

    @validator("uri")
    def validate_uri(cls, v):
        if not v.strip():
            raise ValueError("URI cannot be empty")
        # Check for valid URI format (basic validation)
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", v):
            raise ValueError("Invalid URI format")
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe content in URI")
        return v

    @validator("title")
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError("Title cannot be empty")
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe content in title")
        return v

    @validator("description")
    def validate_description(cls, v):
        if v:
            # Check for suspicious patterns
            suspicious_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError("Potentially unsafe content in description")
        return v


class Retriever(abc.ABC):
    """
    Define a RAG provider, which can be used to query documents and resources.
    """

    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        """
        List resources from the rag provider.
        """
        pass

    @abc.abstractmethod
    def query_relevant_documents(
        self, query: str, resources: list[Resource] = []
    ) -> list[Document]:
        """
        Query relevant documents from the resources.
        """
        pass
