# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from enum import Enum
from typing import List, Optional
import re

from pydantic import BaseModel, Field, validator


class StepType(str, Enum):
    RESEARCH = "research"
    PROCESSING = "processing"


class Step(BaseModel):
    need_search: bool = Field(..., description="Must be explicitly set for each step")
    title: str = Field(..., max_length=200)
    description: str = Field(
        ..., description="Specify exactly what data to collect", max_length=1000
    )
    step_type: StepType = Field(..., description="Indicates the nature of the step")
    execution_res: Optional[str] = Field(
        default=None, description="The Step execution result", max_length=50000
    )

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
        if not v.strip():
            raise ValueError("Description cannot be empty")
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

    @validator("execution_res")
    def validate_execution_res(cls, v):
        if v is not None:
            # Check for suspicious patterns
            suspicious_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError("Potentially unsafe content in execution result")
        return v


class Plan(BaseModel):
    locale: str = Field(
        ...,
        description="e.g. 'en-US' or 'zh-CN', based on the user's language",
        pattern=r"^[a-z]{2}-[A-Z]{2}$",
        max_length=10,
    )
    has_enough_context: bool
    thought: str = Field(..., max_length=5000)
    title: str = Field(..., max_length=200)
    steps: List[Step] = Field(
        default_factory=list,
        description="Research & Processing steps to get more context",
    )

    @validator("thought")
    def validate_thought(cls, v):
        if not v.strip():
            raise ValueError("Thought cannot be empty")
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe content in thought")
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

    @validator("steps")
    def validate_steps(cls, v):
        if len(v) > 50:  # Limit number of steps
            raise ValueError("Too many steps in plan")
        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "has_enough_context": False,
                    "thought": (
                        "To understand the current market trends in AI, we need to gather comprehensive information."
                    ),
                    "title": "AI Market Research Plan",
                    "steps": [
                        {
                            "need_search": True,
                            "title": "Current AI Market Analysis",
                            "description": (
                                "Collect data on market size, growth rates, major players, and investment trends in AI sector."
                            ),
                            "step_type": "research",
                        }
                    ],
                }
            ]
        }
