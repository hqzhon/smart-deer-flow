# -*- coding: utf-8 -*-
"""
Reflexion Models - Enhanced data models for Reflexion pattern.

Implements the Reflexion pattern with:
- Multiple knowledge gaps identification
- External knowledge retrieval
- Knowledge integration
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

from src.utils.reflection.models import BaseReflectionResult


class KnowledgeGapCategory(str, Enum):
    """Categories of knowledge gaps."""

    FACTUAL = "factual"
    CONTEXTUAL = "contextual"
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"
    METHODOLOGICAL = "methodological"


class ExternalKnowledgeSource(str, Enum):
    """Sources of external knowledge."""

    WEB_SEARCH = "web_search"
    LOCAL_RAG = "local_rag"
    API = "api"
    DATABASE = "database"


class KnowledgeGap(BaseModel):
    """Structured knowledge gap with priority and category."""

    description: str = Field(description="Description of the knowledge gap")
    priority: int = Field(
        default=3, ge=1, le=5, description="Priority level (1=lowest, 5=highest)"
    )
    category: KnowledgeGapCategory = Field(
        default=KnowledgeGapCategory.FACTUAL,
        description="Category of the knowledge gap",
    )
    suggested_query: str = Field(
        description="Suggested search query to address this gap"
    )
    impact_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Impact on overall research quality if not addressed",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "priority": self.priority,
            "category": self.category.value,
            "suggested_query": self.suggested_query,
            "impact_score": self.impact_score,
        }


class ExternalKnowledge(BaseModel):
    """External knowledge retrieved to address knowledge gaps."""

    source: ExternalKnowledgeSource = Field(
        description="Source of the external knowledge"
    )
    query: str = Field(description="Query used to retrieve this knowledge")
    content: str = Field(description="Retrieved knowledge content")
    relevance_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Relevance to the original query"
    )
    url: Optional[str] = Field(default=None, description="Source URL if applicable")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_summary(self, max_length: int = 200) -> str:
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


class ReflexionResult(BaseReflectionResult):
    """Result from Reflexion analysis with external knowledge integration.

    Extends BaseReflectionResult with Reflexion-specific fields for
    knowledge gaps and external knowledge integration.
    """

    knowledge_gaps: List[KnowledgeGap] = Field(
        default_factory=list,
        description="Identified knowledge gaps, sorted by priority",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Generated search queries for external knowledge",
    )
    external_knowledge: List[ExternalKnowledge] = Field(
        default_factory=list, description="Retrieved external knowledge"
    )
    integrated_response: Optional[str] = Field(
        default=None,
        description="Integrated response combining initial and external knowledge",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for further research"
    )

    def get_top_gaps(self, n: int = 3) -> List[KnowledgeGap]:
        return sorted(self.knowledge_gaps, key=lambda x: -x.priority)[:n]

    def get_primary_gap(self) -> Optional[KnowledgeGap]:
        if not self.knowledge_gaps:
            return None
        return max(self.knowledge_gaps, key=lambda x: x.priority)

    def to_reflection_result_dict(self) -> Dict[str, Any]:
        primary_gap = self.get_primary_gap()
        return {
            "is_sufficient": self.is_sufficient,
            "primary_knowledge_gap": primary_gap.description if primary_gap else None,
            "primary_follow_up_query": (
                primary_gap.suggested_query if primary_gap else None
            ),
            "confidence_score": self.confidence_score,
            "quality_assessment": self.quality_assessment,
            "recommendations": self.recommendations,
            "priority_areas": [gap.description for gap in self.get_top_gaps()],
        }


class SmartReflectionControllerConfig(BaseModel):
    """Configuration for smart reflection loop controller."""

    min_loops: int = Field(default=1, ge=1, description="Minimum reflection loops")
    max_loops: int = Field(default=5, ge=1, description="Maximum reflection loops")
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for early termination",
    )
    improvement_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum improvement to continue reflection",
    )
    enable_external_search: bool = Field(
        default=True, description="Enable external knowledge search in Reflexion mode"
    )
    max_knowledge_gaps: int = Field(
        default=5, ge=1, le=10, description="Maximum number of knowledge gaps to track"
    )
