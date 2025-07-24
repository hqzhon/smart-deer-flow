# -*- coding: utf-8 -*-
"""
Enhanced Reflection Agent - Phase 1 Implementation

Provides intelligent reflection capabilities for DeerFlow researcher nodes,
inspired by GFLQ's reflection mechanism with knowledge gap identification,
follow-up query generation, and research sufficiency assessment.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from src.config.config_loader import get_settings
from src.utils.reflection.reflection_prompt_manager import ReflectionPromptManager
from src.report_quality.i18n import Language

from langchain_core.messages import HumanMessage


class KnowledgeGap(BaseModel):
    """Represents an identified knowledge gap in research."""

    gap_type: str = Field(description="Type of knowledge gap")
    description: str = Field(description="Description of the gap")
    importance: float = Field(description="Importance score (0.0-1.0)", ge=0.0, le=1.0)
    suggested_queries: List[str] = Field(
        description="Suggested queries to fill the gap", default_factory=list
    )


class ReflectionConfig(BaseModel):
    """Configuration for reflection system."""

    enable_enhanced_reflection: bool = Field(
        default=True, description="Enable enhanced reflection"
    )
    max_reflection_loops: int = Field(default=1, description="Maximum reflection loops")
    reflection_model: Optional[str] = Field(
        default=None, description="Model for reflection"
    )
    knowledge_gap_threshold: float = Field(
        default=0.6, description="Threshold for knowledge gaps"
    )
    sufficiency_threshold: float = Field(
        default=0.7, description="Threshold for sufficiency"
    )


from langchain_core.runnables import RunnableConfig
from src.llms.error_handler import safe_llm_call_async
from src.config import get_settings

logger = logging.getLogger(__name__)


class ReflectionResult(BaseModel):
    """Structured output for reflection analysis.

    Based on GFLQ's Reflection model with DeerFlow-specific enhancements.
    Enhanced to generate comprehensive reports that merge all knowledge.
    """

    is_sufficient: bool = Field(
        description="Whether the current research results are sufficient to complete the task"
    )
    comprehensive_report: str = Field(
        description="Comprehensive research report that synthesizes all findings, analysis, and insights",
        default="",
    )
    knowledge_gaps: List[str] = Field(
        description="List of knowledge gaps or missing information areas",
        default_factory=list,
    )
    follow_up_queries: List[str] = Field(
        description="List of specific follow-up queries to address the knowledge gaps",
        default_factory=list,
    )
    confidence_score: Optional[float] = Field(
        description="Confidence score for the sufficiency assessment (0.0-1.0)",
        ge=0.0,
        le=1.0,
        default=None,
    )
    quality_assessment: Dict[str, Any] = Field(
        description="Quality assessment metrics", default_factory=dict
    )
    recommendations: List[str] = Field(
        description="Actionable recommendations for improving research",
        default_factory=list,
    )
    priority_areas: List[str] = Field(
        description="Priority areas that need immediate attention", default_factory=list
    )

    @field_validator("knowledge_gaps", mode="before")
    @classmethod
    def validate_knowledge_gaps(cls, v):
        """Ensure knowledge_gaps is always a list of strings."""
        if not v:
            return []

        result = []
        for item in v:
            # Skip slice objects and other invalid types
            if isinstance(item, slice):
                logger.warning(f"Skipping slice object in knowledge_gaps: {item}")
                continue
            
            if isinstance(item, str):
                if item.strip() and "slice(" not in item and not item.startswith("<"):
                    result.append(item)
                else:
                    logger.warning(f"Skipping invalid string in knowledge_gaps: '{item}'")
            elif isinstance(item, dict):
                # Extract description or convert dict to string
                if "description" in item:
                    desc_str = str(item["description"])
                    if desc_str.strip() and "slice(" not in desc_str:
                        result.append(desc_str)
                elif "gap_type" in item:
                    gap_str = str(item["gap_type"])
                    if gap_str.strip() and "slice(" not in gap_str:
                        result.append(gap_str)
                else:
                    item_str = str(item)
                    if item_str.strip() and "slice(" not in item_str and not item_str.startswith("<"):
                        result.append(item_str)
            else:
                item_str = str(item)
                if item_str.strip() and "slice(" not in item_str and not item_str.startswith("<"):
                    result.append(item_str)
                else:
                    logger.warning(f"Skipping invalid item in knowledge_gaps: '{item_str}'")
        return result

    @field_validator("follow_up_queries", mode="before")
    @classmethod
    def validate_follow_up_queries(cls, v):
        """Ensure follow_up_queries is always a list of strings."""
        if not v:
            return []

        result = []
        for item in v:
            # Skip slice objects and other invalid types
            if isinstance(item, slice):
                logger.warning(f"Skipping slice object in follow_up_queries: {item}")
                continue
            
            if isinstance(item, str):
                # Validate string content
                if item.strip() and "slice(" not in item and not item.startswith("<"):
                    result.append(item)
                else:
                    logger.warning(f"Skipping invalid string in follow_up_queries: '{item}'")
            elif isinstance(item, dict):
                # Extract meaningful string representation from dictionary
                if "query" in item:
                    query_str = str(item["query"])
                    if query_str.strip() and "slice(" not in query_str:
                        result.append(query_str)
                elif "description" in item:
                    desc_str = str(item["description"])
                    if desc_str.strip() and "slice(" not in desc_str:
                        result.append(desc_str)
                elif "gap_type" in item:
                    gap_str = str(item["gap_type"])
                    if gap_str.strip() and "slice(" not in gap_str:
                        result.append(gap_str)
                else:
                    item_str = str(item)
                    if item_str.strip() and "slice(" not in item_str and not item_str.startswith("<"):
                        result.append(item_str)
            else:
                # Convert other types to string with validation
                item_str = str(item)
                if item_str.strip() and "slice(" not in item_str and not item_str.startswith("<"):
                    result.append(item_str)
                else:
                    logger.warning(f"Skipping invalid item in follow_up_queries: '{item_str}'")
        return result

    @field_validator("recommendations", mode="before")
    @classmethod
    def validate_recommendations(cls, v):
        """Ensure recommendations is always a list of strings."""
        if not v:
            return []

        result = []
        for item in v:
            # Skip slice objects and other invalid types
            if isinstance(item, slice):
                logger.warning(f"Skipping slice object in recommendations: {item}")
                continue
            
            if isinstance(item, str):
                if item.strip() and "slice(" not in item and not item.startswith("<"):
                    result.append(item)
                else:
                    logger.warning(f"Skipping invalid string in recommendations: '{item}'")
            elif isinstance(item, dict):
                # Extract relevant field or convert to string
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
                    if item_str.strip() and "slice(" not in item_str and not item_str.startswith("<"):
                        result.append(item_str)
            else:
                item_str = str(item)
                if item_str.strip() and "slice(" not in item_str and not item_str.startswith("<"):
                    result.append(item_str)
                else:
                    logger.warning(f"Skipping invalid item in recommendations: '{item_str}'")
        return result


@dataclass
class ReflectionContext:
    """Context information for reflection analysis"""

    research_topic: str
    completed_steps: List[Dict[str, Any]]
    current_step: Optional[Dict[str, Any]] = None
    execution_results: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    resources_found: int = 0
    total_steps: int = 0
    current_step_index: int = 0
    locale: str = "en-US"
    max_reflection_loops: int = 1
    current_reflection_loop: int = 0


class EnhancedReflectionAgent:
    """Enhanced reflection agent for intelligent research analysis.

    Provides sophisticated reflection capabilities including:
    - Knowledge gap identification
    - Follow-up query generation
    - Research sufficiency assessment
    - Dynamic research strategy adjustment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced reflection agent.

        Args:
            config: Configuration instance for reflection settings
        """
        try:
            self.config = config or get_settings()
        except Exception:
            # Fallback to empty config if settings unavailable
            self.config = {}
        self.reflection_history: List[Tuple[ReflectionContext, ReflectionResult]] = []

        # Initialize prompt manager for multilingual support
        self.prompt_manager = ReflectionPromptManager()

        # Session tracking
        self.session_count = 0
        self.total_reflections = 0

        # Reflection cache for performance optimization
        self.reflection_cache: Dict[str, Any] = {}

        # Metrics tracking
        self.metrics = {}

        # Reflection configuration
        self.max_reflection_loops = getattr(self.config, "max_reflection_loops", 1)
        self.reflection_model_name = getattr(self.config, "reflection_model", None)
        self.reflection_model = (
            None  # Will be initialized lazily via _get_reflection_model
        )
        self.reflection_temperature = getattr(
            self.config, "reflection_temperature", 0.7
        )
        self.enable_enhanced_reflection = getattr(
            self.config, "enable_enhanced_reflection", True
        )

        logger.info(
            f"Initialized EnhancedReflectionAgent with enhanced_reflection={self.enable_enhanced_reflection}"
        )

    async def analyze_knowledge_gaps(
        self,
        context: ReflectionContext,
        runnable_config: Optional[RunnableConfig] = None,
    ) -> ReflectionResult:
        """Analyze current research state and identify knowledge gaps.

        Args:
            context: Reflection context with research information
            runnable_config: Configuration for LLM execution

        Returns:
            ReflectionResult with gap analysis and recommendations
        """
        if not self.enable_enhanced_reflection:
            logger.debug("Enhanced reflection disabled, returning basic result")
            return self._create_basic_reflection_result(context)

        try:
            # Determine language from context
            language = (
                Language.ZH_CN if context.locale.startswith("zh") else Language.EN_US
            )

            # Prepare reflection prompt using multilingual prompt manager
            reflection_prompt = self._build_reflection_prompt(context, language)

            # Get appropriate model for reflection
            model = self._get_reflection_model(runnable_config)
            if not model:
                logger.warning(
                    "No reflection model available, falling back to basic reflection"
                )
                return self._create_basic_reflection_result(context)

            # Use project's standard approach: structured_output with safe_llm_call_async
            # Use json_mode method for better compatibility with DeepSeek and other models
            structured_model = model.with_structured_output(
                ReflectionResult, method="json_mode"
            )

            # Create messages for the model
            messages = [HumanMessage(content=reflection_prompt)]

            # Log analysis start
            logger.info(
                f"Starting knowledge gap analysis with reflection prompt length: {len(reflection_prompt)}"
            )
            logger.debug(
                f"Knowledge gap analysis input prompt: {reflection_prompt[:500]}..."
                if len(reflection_prompt) > 500
                else f"Knowledge gap analysis input prompt: {reflection_prompt}"
            )

            # Use safe_llm_call_async with structured model
            try:
                result = await safe_llm_call_async(
                    structured_model.ainvoke,
                    messages,
                    operation_name="Enhanced Reflection Analysis",
                    context="Analyzing research knowledge gaps and sufficiency",
                )

                # Log AI analysis results
                logger.info("Knowledge gap analysis AI response received successfully")
                logger.debug(f"Knowledge gap analysis AI result: {result}")

                # If result is None or invalid, use fallback
                if not result or not isinstance(result, ReflectionResult):
                    logger.warning("Invalid reflection result, using fallback")
                    result = self._create_basic_reflection_result(context)
                else:
                    # Log detailed analysis results
                    logger.info(
                        f"Knowledge gap analysis completed: sufficient={result.is_sufficient}, confidence={result.confidence_score}, gaps_found={len(result.knowledge_gaps)}"
                    )
                    logger.debug(f"Knowledge gaps identified: {result.knowledge_gaps}")
                    logger.debug(
                        f"Follow-up queries generated: {result.follow_up_queries}"
                    )
                    logger.debug(
                        f"Comprehensive report length: {len(result.comprehensive_report)}"
                    )

            except Exception as parse_error:
                logger.warning(
                    f"Failed to parse structured output (likely truncated response): {parse_error}"
                )

                # Try to get raw response and handle truncated JSON
                try:
                    # Get raw model without structured output
                    raw_model = self._get_reflection_model(runnable_config)
                    raw_response = await safe_llm_call_async(
                        raw_model.ainvoke,
                        messages,
                        operation_name="Enhanced Reflection Analysis (Raw)",
                        context="Analyzing research knowledge gaps and sufficiency",
                    )

                    # Try to parse and fix truncated JSON
                    result = self._parse_truncated_reflection_response(
                        raw_response.content, context
                    )

                except Exception as raw_error:
                    logger.error(f"Raw response parsing also failed: {raw_error}")
                    result = self._create_basic_reflection_result(
                        context, error=str(parse_error)
                    )

            # Store reflection history
            self.reflection_history.append((context, result))

            logger.info(
                f"Reflection analysis completed: sufficient={result.is_sufficient}, "
                f"confidence={result.confidence_score:.2f}, gaps_found={len(result.follow_up_queries)}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in knowledge gap analysis: {e}")
            return self._create_basic_reflection_result(context, error=str(e))

    async def generate_follow_up_queries(
        self,
        research_topic: str,
        knowledge_gaps: List[str],
        priority_areas: Optional[List[str]] = None,
        language: Language = Language.EN_US,
    ) -> List[str]:
        """Generate specific follow-up queries based on identified knowledge gaps.

        Args:
            research_topic: The research topic
            knowledge_gaps: List of identified knowledge gaps
            priority_areas: Priority areas that need attention
            language: Target language for query generation

        Returns:
            List of refined follow-up queries
        """
        if not knowledge_gaps:
            return []

        try:
            # Use prompt manager for multilingual query generation
            prompt = self.prompt_manager.get_follow_up_queries_prompt(
                research_topic=research_topic,
                knowledge_gaps=knowledge_gaps,
                priority_areas=priority_areas or [],
                language=language,
            )

            # Get model and generate queries
            model = self._get_reflection_model()
            if not model:
                # Fallback to simple query generation
                if language == Language.ZH_CN:
                    return [
                        f"关于{gap}的更多信息需要什么？" for gap in knowledge_gaps[:3]
                    ]
                else:
                    return [
                        f"What additional information is needed about {gap}?"
                        for gap in knowledge_gaps[:3]
                    ]

            response = await self._call_reflection_model(model, prompt)

            # Log AI response details
            logger.info("Follow-up queries generation - AI response received")
            logger.debug(f"Follow-up queries AI raw response: {response}")

            # Parse response
            import json

            try:
                result = json.loads(response.content)
                raw_queries = result.get("follow_up_queries", [])

                # Ensure all queries are strings
                follow_up_queries = []
                for query in raw_queries:
                    if isinstance(query, str):
                        follow_up_queries.append(query)
                    elif isinstance(query, dict):
                        # Extract meaningful string from dict
                        if "query" in query:
                            follow_up_queries.append(str(query["query"]))
                        elif "question" in query:
                            follow_up_queries.append(str(query["question"]))
                        elif "description" in query:
                            follow_up_queries.append(str(query["description"]))
                        else:
                            follow_up_queries.append(str(query))
                    else:
                        follow_up_queries.append(str(query))

                # Log parsed results
                logger.info(f"Generated {len(follow_up_queries)} follow-up queries")
                logger.debug(f"Follow-up queries result: {follow_up_queries}")

                return follow_up_queries
            except (json.JSONDecodeError, AttributeError):
                # Fallback
                logger.warning(
                    "Failed to parse follow-up queries response, using fallback"
                )
                if language == Language.ZH_CN:
                    return [
                        f"关于{gap}的更多信息需要什么？" for gap in knowledge_gaps[:3]
                    ]
                else:
                    return [
                        f"What additional information is needed about {gap}?"
                        for gap in knowledge_gaps[:3]
                    ]

        except Exception as e:
            logger.error(f"Error generating follow-up queries: {e}")
            if language == Language.ZH_CN:
                return [f"关于{gap}的更多信息需要什么？" for gap in knowledge_gaps[:3]]
            else:
                return [
                    f"What additional information is needed about {gap}?"
                    for gap in knowledge_gaps[:3]
                ]

    def assess_sufficiency(
        self, context: ReflectionContext, reflection_result: ReflectionResult
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Assess whether current research is sufficient for task completion.

        Args:
            context: Current reflection context
            reflection_result: Result from reflection analysis

        Returns:
            Tuple of (is_sufficient, reasoning, assessment_details)
        """
        # Basic sufficiency checks
        basic_sufficient = (
            len(context.execution_results) > 0
            and context.current_step_index >= context.total_steps - 1
        )

        # Enhanced sufficiency assessment
        confidence_threshold = getattr(
            self.config, "reflection_confidence_threshold", 0.7
        )
        enhanced_sufficient = (
            reflection_result.is_sufficient
            and reflection_result.confidence_score is not None
            and reflection_result.confidence_score >= confidence_threshold
        )

        # Combine assessments
        is_sufficient = basic_sufficient or enhanced_sufficient

        # Generate reasoning
        reasoning_parts = []
        if basic_sufficient:
            reasoning_parts.append("Basic completion criteria met")
        if enhanced_sufficient:
            reasoning_parts.append(
                f"Enhanced reflection confirms sufficiency (confidence: {reflection_result.confidence_score:.2f})"
            )
        if not is_sufficient:
            # Ensure knowledge_gaps are all strings before joining
            safe_gaps = []
            for gap in reflection_result.knowledge_gaps or []:
                if isinstance(gap, str):
                    safe_gaps.append(gap)
                elif isinstance(gap, dict):
                    # Extract meaningful string from dict
                    if "description" in gap:
                        safe_gaps.append(str(gap["description"]))
                    elif "gap_type" in gap:
                        safe_gaps.append(str(gap["gap_type"]))
                    else:
                        safe_gaps.append(str(gap))
                else:
                    safe_gaps.append(str(gap))

            gaps_text = ", ".join(safe_gaps) if safe_gaps else "general research gaps"
            reasoning_parts.append(f"Knowledge gaps identified: {gaps_text}")

        reasoning = "; ".join(reasoning_parts)

        # Assessment details
        assessment_details = {
            "basic_sufficient": basic_sufficient,
            "enhanced_sufficient": enhanced_sufficient,
            "confidence_score": reflection_result.confidence_score,
            "knowledge_gaps_count": len(reflection_result.knowledge_gaps),
            "reflection_loop": context.current_reflection_loop,
            "max_loops": self.max_reflection_loops,
        }

        logger.info(f"Sufficiency assessment: {is_sufficient} - {reasoning}")
        return is_sufficient, reasoning, assessment_details

    def _build_reflection_prompt(
        self, context: ReflectionContext, language: Language = Language.EN_US
    ) -> str:
        """Build the reflection prompt based on context using multilingual prompt manager.

        Args:
            context: Reflection context with research information
            language: Target language for the prompt

        Returns:
            Formatted reflection prompt
        """
        # Prepare execution results summary with length limits to prevent truncation
        max_results_length = 2000  # Limit to prevent overly long prompts
        if context.execution_results:
            combined_results = "\n\n---\n\n".join(str(res) for res in context.execution_results)
            if len(combined_results) > max_results_length:
                # Truncate and add note
                truncated_results = combined_results[:max_results_length]
                truncation_note = (
                    "\n\n[更多结果已省略以控制长度]"
                    if language == Language.ZH_CN
                    else "\n\n[Additional results truncated for length control]"
                )
                results_summary = truncated_results + truncation_note
            else:
                results_summary = combined_results
        else:
            results_summary = (
                "暂无执行结果。"
                if language == Language.ZH_CN
                else "No execution results available yet."
            )

        # Prepare completed steps summary with length limits
        max_steps_length = 1500
        steps_summary = ""
        for i, step in enumerate(context.completed_steps):
            step_text = ""
            if language == Language.ZH_CN:
                step_text += f"步骤 {i+1}: {step.get('step', '未知')}\n"
                step_text += f"描述: {step.get('description', '无描述')}\n"
                if step.get("execution_res"):
                    # Limit individual step result length
                    exec_res = (
                        step["execution_res"][:150] + "..."
                        if len(step["execution_res"]) > 150
                        else step["execution_res"]
                    )
                    step_text += f"结果: {exec_res}\n"
            else:
                step_text += f"Step {i+1}: {step.get('step', 'Unknown')}\n"
                step_text += (
                    f"Description: {step.get('description', 'No description')}\n"
                )
                if step.get("execution_res"):
                    # Limit individual step result length
                    exec_res = (
                        step["execution_res"][:150] + "..."
                        if len(step["execution_res"]) > 150
                        else step["execution_res"]
                    )
                    step_text += f"Result: {exec_res}\n"
            step_text += "\n"

            # Check if adding this step would exceed length limit
            if len(steps_summary + step_text) > max_steps_length:
                truncation_note = (
                    "\n[更多步骤已省略]"
                    if language == Language.ZH_CN
                    else "\n[Additional steps truncated]"
                )
                steps_summary += truncation_note
                break
            steps_summary += step_text

        # Prepare observations summary with length limits
        max_obs_length = 1000
        if context.observations:
            combined_obs = "\n".join(str(obs) for obs in context.observations)
            if len(combined_obs) > max_obs_length:
                truncated_obs = combined_obs[:max_obs_length]
                truncation_note = (
                    "\n[更多观察已省略]"
                    if language == Language.ZH_CN
                    else "\n[Additional observations truncated]"
                )
                observations_summary = truncated_obs + truncation_note
            else:
                observations_summary = combined_obs
        else:
            observations_summary = (
                "无观察记录。"
                if language == Language.ZH_CN
                else "No observations recorded."
            )

        # Use prompt manager to generate multilingual prompt
        return self.prompt_manager.get_reflection_analysis_prompt(
            research_topic=context.research_topic,
            current_step_index=context.current_step_index + 1,
            total_steps=context.total_steps,
            current_reflection_loop=context.current_reflection_loop + 1,
            max_reflection_loops=self.max_reflection_loops,
            steps_summary=steps_summary,
            results_summary=results_summary,
            observations_summary=observations_summary,
            language=language,
        )

    def _parse_truncated_reflection_response(
        self, raw_content: str, context: ReflectionContext
    ) -> ReflectionResult:
        """Parse potentially truncated JSON response and extract what we can.

        Args:
            raw_content: Raw response content from LLM
            context: Reflection context for fallback values

        Returns:
            ReflectionResult with extracted or fallback values
        """
        import json
        import re

        try:
            # First try direct JSON parsing
            parsed = json.loads(raw_content)

            # Fix comprehensive_report if it's a dict instead of string
            if isinstance(parsed.get("comprehensive_report"), dict):
                report_dict = parsed["comprehensive_report"]
                # Convert dict to formatted string
                report_parts = []

                if "executive_summary" in report_dict:
                    report_parts.append(
                        f"## Executive Summary\n{report_dict['executive_summary']}"
                    )

                if "detailed_analysis" in report_dict:
                    analysis = report_dict["detailed_analysis"]
                    if isinstance(analysis, dict):
                        report_parts.append("## Detailed Analysis")
                        for key, value in analysis.items():
                            report_parts.append(
                                f"### {key.replace('_', ' ').title()}\n{value}"
                            )
                    else:
                        report_parts.append(f"## Detailed Analysis\n{analysis}")

                if "conclusions" in report_dict:
                    report_parts.append(f"## Conclusions\n{report_dict['conclusions']}")

                # Join all parts into a single string
                parsed["comprehensive_report"] = "\n\n".join(report_parts)
                logger.info(
                    f"Converted comprehensive_report from dict to string (length: {len(parsed['comprehensive_report'])})"
                )

            return ReflectionResult(**parsed)
        except json.JSONDecodeError:
            logger.info(
                "Direct JSON parsing failed, attempting to extract partial data"
            )
        except Exception as parse_error:
            logger.warning(
                f"Failed to parse structured output: {parse_error}, attempting to extract partial data"
            )

        # Try to extract partial data from truncated JSON
        extracted_data = {
            "is_sufficient": False,
            "comprehensive_report": "",
            "knowledge_gaps": [],
            "follow_up_queries": [],
            "confidence_score": 0.5,
            "quality_assessment": {},
            "recommendations": [],
            "priority_areas": [],
        }

        try:
            # Extract is_sufficient
            is_sufficient_match = re.search(
                r'"is_sufficient"\s*:\s*(true|false)', raw_content, re.IGNORECASE
            )
            if is_sufficient_match:
                extracted_data["is_sufficient"] = (
                    is_sufficient_match.group(1).lower() == "true"
                )

            # Extract comprehensive_report (even if truncated)
            report_match = re.search(
                r'"comprehensive_report"\s*:\s*"([^"]*(?:\\.[^"]*)*)',
                raw_content,
                re.DOTALL,
            )
            if report_match:
                # Decode unicode escapes and clean up
                report_content = report_match.group(1)
                try:
                    # Handle unicode escapes
                    if isinstance(report_content, str):
                        report_content = report_content.encode().decode(
                            "unicode_escape"
                        )
                except:
                    pass
                extracted_data["comprehensive_report"] = report_content

                # If report seems truncated, add a note
                if len(report_content) > 1000 and not report_content.endswith(
                    ("。", ".", "!", "?")
                ):
                    extracted_data["comprehensive_report"] += (
                        "\n\n[注：由于响应长度限制，报告可能被截断]"
                        if context.locale.startswith("zh")
                        else "\n\n[Note: Report may be truncated due to response length limits]"
                    )

            # Extract confidence_score
            confidence_match = re.search(
                r'"confidence_score"\s*:\s*([0-9.]+)', raw_content
            )
            if confidence_match:
                extracted_data["confidence_score"] = float(confidence_match.group(1))

            # Extract arrays (knowledge_gaps, follow_up_queries, etc.)
            for field in [
                "knowledge_gaps",
                "follow_up_queries",
                "recommendations",
                "priority_areas",
            ]:
                array_pattern = rf'"({field})"\s*:\s*\[([^\]]*(?:\][^\]]*)*?)\]'
                array_match = re.search(array_pattern, raw_content, re.DOTALL)
                if array_match:
                    array_content = array_match.group(2)
                    # Try to extract both simple strings and complex objects
                    items = []

                    # First try to extract simple quoted strings
                    string_items = re.findall(r'"([^"]*(?:\\.[^"]*)*?)"', array_content)
                    if string_items:
                        items.extend(string_items)
                    else:
                        # If no simple strings found, try to extract objects and convert them
                        object_pattern = r"\{[^{}]*\}"
                        object_matches = re.findall(object_pattern, array_content)
                        for obj_str in object_matches:
                            try:
                                import json

                                obj = json.loads(obj_str)
                                if isinstance(obj, dict):
                                    # Extract meaningful string from object
                                    if "description" in obj:
                                        items.append(str(obj["description"]))
                                    elif "gap_type" in obj:
                                        items.append(str(obj["gap_type"]))
                                    else:
                                        items.append(str(obj))
                                else:
                                    items.append(str(obj))
                            except:
                                # If JSON parsing fails, just use the string representation
                                items.append(str(obj_str))

                    if items:
                        # Decode unicode escapes and ensure all items are strings
                        decoded_items = []
                        for item in items:
                            try:
                                # Skip slice objects and other invalid types
                                if isinstance(item, slice):
                                    logger.warning(f"Skipping slice object in {field}: {item}")
                                    continue
                                
                                if isinstance(item, str):
                                    decoded_item = item.encode().decode("unicode_escape")
                                    # Additional validation for decoded string
                                    if decoded_item and decoded_item.strip() and "slice(" not in decoded_item:
                                        decoded_items.append(decoded_item)
                                    else:
                                        logger.warning(f"Skipping invalid decoded item in {field}: '{decoded_item}'")
                                else:
                                    str_item = str(item)
                                    # Validate string representation
                                    if str_item and str_item.strip() and "slice(" not in str_item and not str_item.startswith("<"):
                                        decoded_items.append(str_item)
                                    else:
                                        logger.warning(f"Skipping invalid item in {field}: '{str_item}'")
                            except Exception as e:
                                logger.warning(f"Error processing item in {field}: {e}")
                                # Only add if it's a valid string
                                if isinstance(item, str) and item.strip() and "slice(" not in item:
                                    decoded_items.append(item)
                        
                        # Only set the field if we have valid items
                        if decoded_items:
                            extracted_data[field] = decoded_items
                        else:
                            logger.warning(f"No valid items found for {field}, using empty list")
                            extracted_data[field] = []

            logger.info(
                f"Extracted partial data from truncated response: is_sufficient={extracted_data['is_sufficient']}, report_length={len(extracted_data['comprehensive_report'])}, gaps={len(extracted_data['knowledge_gaps'])}"
            )

        except Exception as extract_error:
            logger.error(f"Failed to extract partial data: {extract_error}")
            # Use basic fallback
            extracted_data["comprehensive_report"] = (
                "研究分析报告\n\n由于技术限制，无法生成完整的综合报告。"
                if context.locale.startswith("zh")
                else "Research Analysis Report\n\nUnable to generate complete comprehensive report due to technical limitations."
            )

        return ReflectionResult(**extracted_data)

    def _get_reflection_model(self, runnable_config: Optional[RunnableConfig] = None):
        """Get the appropriate model for reflection analysis."""
        if self.reflection_model:
            return self.reflection_model

        # Import here to avoid circular imports
        try:
            from src.llms.llm import get_llm_by_type

            # Default to basic model
            model_type = "basic"

            # Check enable_deep_thinking from runnable_config
            if runnable_config:
                try:
                    from src.graph.nodes import get_configuration_from_config

                    config = get_configuration_from_config(runnable_config)
                    if (
                        hasattr(config, "agents")
                        and hasattr(config.agents, "enable_deep_thinking")
                        and config.agents.enable_deep_thinking
                    ):
                        model_type = "reasoning"
                except Exception as config_error:
                    logger.warning(
                        f"Failed to get configuration from runnable_config: {config_error}"
                    )

            # Get the model and cache it
            model = get_llm_by_type(model_type)
            if model:
                self.reflection_model = model
            return model

        except Exception as e:
            logger.warning(f"Failed to get LLM instance: {e}")

        return None

    async def _call_reflection_model(self, model, prompt: str):
        """Call the reflection model with the given prompt.

        Args:
            model: The language model to use
            prompt: The prompt to send to the model

        Returns:
            Model response
        """
        try:
            messages = [HumanMessage(content=prompt)]

            logger.info(f"Calling reflection model with prompt length: {len(prompt)}")
            logger.debug(
                f"Reflection model input prompt: {prompt[:500]}..."
                if len(prompt) > 500
                else f"Reflection model input prompt: {prompt}"
            )

            response = await model.ainvoke(messages)

            # Log AI return results
            logger.info("Reflection model response received successfully")
            logger.debug(f"Reflection model response: {response}")

            # Log response content if available
            if hasattr(response, "content"):
                logger.info(
                    f"Reflection model content length: {len(str(response.content))}"
                )
                logger.debug(f"Reflection model content: {response.content}")

            return response
        except Exception as e:
            logger.error(f"Error calling reflection model: {e}")

            # Return a mock response for fallback
            class MockResponse:
                def __init__(self, content):
                    self.content = content

            return MockResponse('{"follow_up_queries": []}')

    def _enhance_query_with_context(
        self, query: str, context: ReflectionContext
    ) -> str:
        """Enhance a follow-up query with contextual information."""
        # Add research topic context if not already present
        if context.research_topic.lower() not in query.lower():
            query = f"{query} related to {context.research_topic}"

        # Add temporal context for current information
        current_year = datetime.now().year
        if str(current_year) not in query and "recent" not in query.lower():
            query = f"{query} {current_year}"

        return query

    def _create_basic_reflection_result(
        self, context: ReflectionContext, error: Optional[str] = None
    ) -> ReflectionResult:
        """Create a basic reflection result when enhanced reflection is unavailable."""
        # Simple heuristic-based assessment
        has_results = len(context.execution_results) > 0
        is_near_completion = context.current_step_index >= context.total_steps - 1

        is_sufficient = has_results and is_near_completion
        confidence = 0.6 if is_sufficient else 0.4

        knowledge_gaps = []
        follow_up_queries = []

        if not is_sufficient:
            if not has_results:
                knowledge_gaps = [
                    "No execution results available yet, need to complete current research steps"
                ]
            else:
                knowledge_gaps = [
                    "Research in progress, may need additional information to ensure completeness"
                ]
                follow_up_queries = [
                    f"Additional details about {context.research_topic}"
                ]

        recommendations = (
            ["Continue with planned research steps"]
            if not is_sufficient
            else ["Proceed to report generation"]
        )
        if error:
            recommendations.append(f"Enhanced reflection failed: {error}")

        return ReflectionResult(
            is_sufficient=is_sufficient,
            knowledge_gaps=knowledge_gaps,
            follow_up_queries=follow_up_queries,
            confidence_score=confidence,
            recommendations=recommendations,
        )

    def get_reflection_metrics(self) -> Dict[str, Any]:
        """Get metrics about reflection performance."""
        if not self.reflection_history:
            return {"total_reflections": 0}

        total_reflections = len(self.reflection_history)
        sufficient_count = sum(
            1 for _, result in self.reflection_history if result.is_sufficient
        )
        confidence_scores = [
            result.confidence_score
            for _, result in self.reflection_history
            if result.confidence_score is not None
        ]
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )
        avg_follow_ups = (
            sum(len(result.follow_up_queries) for _, result in self.reflection_history)
            / total_reflections
        )

        return {
            "total_reflections": total_reflections,
            "sufficient_rate": sufficient_count / total_reflections,
            "average_confidence": avg_confidence,
            "average_follow_up_queries": avg_follow_ups,
            "enhanced_reflection_enabled": self.enable_enhanced_reflection,
        }

    async def analyze_research_quality(
        self, context: ReflectionContext
    ) -> Dict[str, Any]:
        """Analyze the quality of current research.

        Args:
            context: Current reflection context

        Returns:
            Dictionary with quality metrics and assessment
        """
        quality_metrics = {
            "completeness_score": 0.0,
            "depth_score": 0.0,
            "relevance_score": 0.0,
            "overall_quality": 0.0,
            "areas_for_improvement": [],
        }

        # Calculate completeness based on execution results
        if context.execution_results:
            quality_metrics["completeness_score"] = min(
                1.0, len(context.execution_results) / max(1, context.total_steps)
            )

        # Calculate depth based on result length and detail
        if context.execution_results:
            avg_result_length = sum(
                len(result) for result in context.execution_results
            ) / len(context.execution_results)
            quality_metrics["depth_score"] = min(
                1.0, avg_result_length / 500
            )  # Normalize to 500 chars

        # Calculate relevance (simplified heuristic)
        if context.research_topic and context.execution_results:
            topic_mentions = sum(
                1
                for result in context.execution_results
                if context.research_topic.lower() in result.lower()
            )
            quality_metrics["relevance_score"] = topic_mentions / len(
                context.execution_results
            )

        # Calculate overall quality
        quality_metrics["overall_quality"] = (
            quality_metrics["completeness_score"] * 0.4
            + quality_metrics["depth_score"] * 0.3
            + quality_metrics["relevance_score"] * 0.3
        )

        # Identify areas for improvement
        if quality_metrics["completeness_score"] < 0.7:
            quality_metrics["areas_for_improvement"].append("Research completeness")
        if quality_metrics["depth_score"] < 0.6:
            quality_metrics["areas_for_improvement"].append("Research depth and detail")
        if quality_metrics["relevance_score"] < 0.8:
            quality_metrics["areas_for_improvement"].append("Topic relevance")

        return quality_metrics

    def cleanup(self):
        """Clean up reflection system resources."""
        self.reflection_cache.clear()
        self.reflection_history.clear()
        self.metrics.clear()
        logger.info("Reflection system resources cleaned up")

    async def assess_research_sufficiency(
        self, context: ReflectionContext
    ) -> ReflectionResult:
        """Assess if current research is sufficient.

        Args:
            context: Current reflection context

        Returns:
            ReflectionResult with sufficiency assessment and details
        """
        # Perform knowledge gap analysis
        reflection_result = await self.analyze_knowledge_gaps(context)

        return reflection_result
