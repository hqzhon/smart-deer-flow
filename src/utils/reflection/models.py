# -*- coding: utf-8 -*-
"""
Unified Reflection Models - Shared data models for reflection system.

Provides base models for reflection results with common fields,
allowing different reflection implementations to share common structure.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class BaseReflectionResult(BaseModel):
    """Base model for reflection results with common fields."""

    is_sufficient: bool = Field(
        description="Whether the current research results are sufficient"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score for the sufficiency assessment (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    quality_assessment: Dict[str, Any] = Field(
        default_factory=dict, description="Quality assessment metrics"
    )


class ReflectionResult(BaseReflectionResult):
    """Standard reflection result for enhanced reflection analysis.

    Focuses on research sufficiency assessment and knowledge gap identification.
    """

    primary_knowledge_gap: Optional[str] = Field(
        default=None,
        description="The most critical knowledge gap or missing information area",
    )
    primary_follow_up_query: Optional[str] = Field(
        default=None,
        description="The most important follow-up query to address the primary knowledge gap",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improving research",
    )
    priority_areas: List[str] = Field(
        default_factory=list,
        description="Priority areas that need immediate attention",
    )

    @field_validator("recommendations", mode="before")
    @classmethod
    def validate_recommendations(cls, v):
        """Ensure recommendations is always a list of strings."""
        if not v:
            return []

        result = []
        for item in v:
            if isinstance(item, slice):
                logger.warning(f"Skipping slice object in recommendations: {item}")
                continue

            if isinstance(item, str):
                if item.strip() and "slice(" not in item and not item.startswith("<"):
                    result.append(item)
                else:
                    logger.warning(
                        f"Skipping invalid string in recommendations: '{item}'"
                    )
            elif isinstance(item, dict):
                if "recommendation" in item:
                    rec_str = str(item["recommendation"])
                    if rec_str.strip() and "slice(" not in rec_str:
                        result.append(rec_str)
                elif "description" in item:
                    desc_str = str(item["description"])
                    if desc_str.strip() and "slice(" not in desc_str:
                        result.append(desc_str)
                else:
                    item_str = str(item)
                    if (
                        item_str.strip()
                        and "slice(" not in item_str
                        and not item_str.startswith("<")
                    ):
                        result.append(item_str)
            else:
                item_str = str(item)
                if (
                    item_str.strip()
                    and "slice(" not in item_str
                    and not item_str.startswith("<")
                ):
                    result.append(item_str)
                else:
                    logger.warning(
                        f"Skipping invalid item in recommendations: '{item_str}'"
                    )
        return result
