# -*- coding: utf-8 -*-
"""
Reflexion Agent - Enhanced reflection with external knowledge retrieval.

Implements the Reflexion pattern:
1. Self-reflection to identify knowledge gaps
2. External knowledge retrieval
3. Knowledge integration
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from .reflexion_models import (
    ReflexionResult,
    KnowledgeGap,
    KnowledgeGapCategory,
    ExternalKnowledge,
    ExternalKnowledgeSource,
    SmartReflectionControllerConfig,
)
from .smart_reflection_controller import (
    SmartReflectionController,
    KnowledgeGapPrioritizer,
)
from .enhanced_reflection import ReflectionContext, ReflectionResult

if TYPE_CHECKING:
    from .enhanced_reflection import EnhancedReflectionAgent

logger = logging.getLogger(__name__)


class KnowledgeGapList(BaseModel):
    """Structured output for knowledge gap identification."""

    gaps: List[KnowledgeGap] = Field(
        default_factory=list, description="List of identified knowledge gaps"
    )
    is_sufficient: bool = Field(
        default=False, description="Whether current research is sufficient"
    )
    confidence_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    quality_assessment: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )


class ReflexionAgent:
    """Reflexion agent with external knowledge retrieval capability.

    The Reflexion pattern extends basic reflection with:
    1. Multi-gap identification (not just primary gap)
    2. External knowledge search
    3. Knowledge integration
    """

    def __init__(
        self,
        config: Optional[SmartReflectionControllerConfig] = None,
        search_tool=None,
    ):
        self.config = config or SmartReflectionControllerConfig()
        self.search_tool = search_tool
        self.controller = SmartReflectionController(self.config)
        self.prioritizer = KnowledgeGapPrioritizer(self.config.max_knowledge_gaps)

        self._reflection_history: List[ReflexionResult] = []
        self._completed_queries: List[str] = []

    async def analyze_with_reflexion(
        self,
        context: ReflectionContext,
        reflection_agent: "EnhancedReflectionAgent",
        runnable_config: Optional[RunnableConfig] = None,
    ) -> ReflexionResult:
        """Perform Reflexion analysis with external knowledge retrieval.

        Args:
            context: Reflection context with research information
            reflection_agent: The base reflection agent for gap analysis
            runnable_config: Configuration for LLM execution

        Returns:
            ReflexionResult with gaps, external knowledge, and integrated response
        """
        logger.info("Starting Reflexion analysis")

        base_result = await reflection_agent.analyze_knowledge_gaps(
            context, runnable_config
        )

        knowledge_gaps = await self._identify_multiple_gaps(
            context, base_result, runnable_config
        )

        if not knowledge_gaps and base_result.is_sufficient:
            return ReflexionResult(
                is_sufficient=True,
                knowledge_gaps=[],
                search_queries=[],
                external_knowledge=[],
                confidence_score=base_result.confidence_score or 0.8,
                quality_assessment=base_result.quality_assessment,
            )

        prioritized_gaps = self.prioritizer.prioritize(
            knowledge_gaps,
            context.research_topic,
            self._completed_queries,
        )

        search_queries = [gap.suggested_query for gap in prioritized_gaps]

        external_knowledge = []
        if self.config.enable_external_search and self.search_tool:
            external_knowledge = await self._search_external_knowledge(
                search_queries, context.research_topic
            )

        integrated_response = None
        if external_knowledge:
            integrated_response = await self._integrate_knowledge(
                context, external_knowledge, prioritized_gaps, runnable_config
            )

        result = ReflexionResult(
            is_sufficient=base_result.is_sufficient and len(knowledge_gaps) == 0,
            knowledge_gaps=prioritized_gaps,
            search_queries=search_queries,
            external_knowledge=external_knowledge,
            integrated_response=integrated_response,
            confidence_score=base_result.confidence_score or 0.5,
            quality_assessment=base_result.quality_assessment,
            recommendations=base_result.recommendations,
        )

        self._reflection_history.append(result)
        self._completed_queries.extend(search_queries)

        logger.info(
            f"Reflexion analysis complete: sufficient={result.is_sufficient}, "
            f"gaps={len(result.knowledge_gaps)}, external_knowledge={len(result.external_knowledge)}"
        )

        return result

    async def _identify_multiple_gaps(
        self,
        context: ReflectionContext,
        base_result: ReflectionResult,
        runnable_config: Optional[RunnableConfig] = None,
    ) -> List[KnowledgeGap]:
        """Identify multiple knowledge gaps from the research context.

        Args:
            context: Reflection context
            base_result: Base reflection result
            runnable_config: LLM configuration

        Returns:
            List of identified knowledge gaps
        """
        if base_result.is_sufficient:
            return []

        gaps = []

        if base_result.primary_knowledge_gap:
            gaps.append(
                KnowledgeGap(
                    description=base_result.primary_knowledge_gap,
                    priority=5,
                    category=KnowledgeGapCategory.FACTUAL,
                    suggested_query=base_result.primary_follow_up_query
                    or base_result.primary_knowledge_gap,
                    impact_score=0.8,
                )
            )

        for area in base_result.priority_areas[:3]:
            if area and area != base_result.primary_knowledge_gap:
                gaps.append(
                    KnowledgeGap(
                        description=area,
                        priority=3,
                        category=KnowledgeGapCategory.CONTEXTUAL,
                        suggested_query=f"Research about {area}",
                        impact_score=0.5,
                    )
                )

        return gaps

    async def _search_external_knowledge(
        self,
        queries: List[str],
        research_topic: str,
    ) -> List[ExternalKnowledge]:
        """Search for external knowledge using the search tool.

        Args:
            queries: Search queries
            research_topic: Research topic for context

        Returns:
            List of external knowledge items
        """
        if not self.search_tool:
            logger.warning("No search tool available for external knowledge retrieval")
            return []

        external_knowledge = []

        for query in queries[:3]:
            try:
                logger.info(f"Searching for external knowledge: {query}")

                if hasattr(self.search_tool, "ainvoke"):
                    result = await self.search_tool.ainvoke(query)
                elif hasattr(self.search_tool, "invoke"):
                    result = self.search_tool.invoke(query)
                else:
                    logger.warning("Search tool does not have invoke method")
                    continue

                if result:
                    content = self._extract_search_content(result)
                    if content:
                        external_knowledge.append(
                            ExternalKnowledge(
                                source=ExternalKnowledgeSource.WEB_SEARCH,
                                query=query,
                                content=content,
                                relevance_score=0.7,
                                url=self._extract_url(result),
                            )
                        )

            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")

        logger.info(f"Retrieved {len(external_knowledge)} external knowledge items")
        return external_knowledge

    def _extract_search_content(self, result: Any) -> str:
        """Extract content from search result."""
        if isinstance(result, str):
            return result[:1000] if len(result) > 1000 else result
        elif isinstance(result, dict):
            content = (
                result.get("content", "")
                or result.get("snippet", "")
                or result.get("text", "")
            )
            return content[:1000] if len(content) > 1000 else content
        elif hasattr(result, "content"):
            return str(result.content)[:1000]
        else:
            return str(result)[:1000]

    def _extract_url(self, result: Any) -> Optional[str]:
        """Extract URL from search result."""
        if isinstance(result, dict):
            return result.get("url") or result.get("link")
        elif hasattr(result, "url"):
            return result.url
        return None

    async def _integrate_knowledge(
        self,
        context: ReflectionContext,
        external_knowledge: List[ExternalKnowledge],
        gaps: List[KnowledgeGap],
        runnable_config: Optional[RunnableConfig] = None,
    ) -> str:
        """Integrate external knowledge with existing research.

        Args:
            context: Reflection context
            external_knowledge: Retrieved external knowledge
            gaps: Knowledge gaps being addressed
            runnable_config: LLM configuration

        Returns:
            Integrated response
        """
        if not external_knowledge:
            return ""

        knowledge_summary = "\n\n".join(
            [
                f"Source: {k.source.value}\nQuery: {k.query}\nContent: {k.to_summary(300)}"
                for k in external_knowledge
            ]
        )

        gaps_summary = "\n".join([f"- {g.description}" for g in gaps])

        integration_prompt = f"""Based on the following external knowledge, provide a brief summary that addresses the identified knowledge gaps.

Research Topic: {context.research_topic}

Knowledge Gaps:
{gaps_summary}

External Knowledge Retrieved:
{knowledge_summary}

Please provide a concise summary (max 300 words) that integrates this new knowledge with the research topic."""

        return integration_prompt

    def should_continue(self, result: ReflexionResult) -> bool:
        """Check if reflection should continue."""
        should_continue, reason = self.controller.should_continue_reflection(result)
        logger.info(
            f"Reflection continue decision: {should_continue}, reason: {reason}"
        )
        return should_continue

    def get_stats(self) -> Dict[str, Any]:
        """Get reflection statistics."""
        return {
            **self.controller.get_loop_stats(),
            "total_queries": len(self._completed_queries),
            "total_results": len(self._reflection_history),
        }

    def reset(self) -> None:
        """Reset the agent for a new reflection session."""
        self.controller.reset()
        self._reflection_history.clear()
        self._completed_queries.clear()
