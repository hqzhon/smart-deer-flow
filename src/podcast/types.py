# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Literal
import re

from pydantic import BaseModel, Field, validator


class ScriptLine(BaseModel):
    speaker: Literal["male", "female"] = Field(default="male")
    paragraph: str = Field(default="", max_length=5000)

    @validator("paragraph")
    def validate_paragraph(cls, v):
        if v:
            # Check for suspicious patterns
            suspicious_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError("Potentially unsafe content in paragraph")
        return v


class Script(BaseModel):
    locale: Literal["en", "zh"] = Field(default="en")
    lines: list[ScriptLine] = Field(default=[])

    @validator("lines")
    def validate_lines(cls, v):
        if len(v) > 1000:  # Limit number of script lines
            raise ValueError("Too many script lines")
        return v
