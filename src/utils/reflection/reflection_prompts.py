# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Reflection prompts for enhanced research quality analysis.
Based on GFLQ (Gemini Full-Stack LangGraph Quickstart) reflection mechanism.
"""

from typing import List, Dict, Any
from src.prompts.prompt_manager import get_prompt_with_variables


def get_reflection_instructions(
    research_topic: str = "",
    current_findings: List[str] = None,
    step_count: int = 0,
    previous_gaps: List[str] = None,
    locale: str = "en-US",
) -> str:
    """
    Get the core reflection analysis instructions.

    Args:
        research_topic: The main research topic
        current_findings: List of current research findings
        step_count: Number of research steps completed
        previous_gaps: Previously identified knowledge gaps
        locale: Language locale (e.g., "en-US", "zh-CN")

    Returns:
        str: Reflection analysis prompt template
    """
    return get_prompt_with_variables(
        "reflection_instructions",
        {
            "research_topic": research_topic,
            "current_findings": current_findings or [],
            "step_count": step_count,
            "previous_gaps": previous_gaps or [],
            "locale": locale,
        },
    )


def get_context_analysis_prompt(
    research_topic: str,
    current_findings: List[str],
    step_count: int,
    previous_gaps: List[str] = None,
    locale: str = "en-US",
) -> str:
    """
    Generate a context-specific reflection prompt.

    Args:
        research_topic: The main research topic
        current_findings: List of current research findings
        step_count: Number of research steps completed
        previous_gaps: Previously identified knowledge gaps
        locale: Language locale (e.g., "en-US", "zh-CN")

    Returns:
        str: Formatted reflection prompt
    """
    return get_reflection_instructions(
        research_topic=research_topic,
        current_findings=current_findings,
        step_count=step_count,
        previous_gaps=previous_gaps,
        locale=locale,
    )


def get_progressive_reflection_prompt(
    complexity_score: float, isolation_active: bool, context_size: int
) -> str:
    """
    Generate a prompt for progressive reflection analysis.

    Args:
        complexity_score: Research complexity score (0.0 to 1.0)
        isolation_active: Whether context isolation is active
        context_size: Current context size

    Returns:
        str: Progressive reflection prompt
    """
    return f"""
Progressive Reflection Analysis

Research Context:
- Complexity Score: {complexity_score:.2f}
- Context Isolation Active: {isolation_active}
- Context Size: {context_size} tokens

Adjust your reflection analysis based on the research complexity:
- High complexity (>0.7): Focus on comprehensive coverage and deep analysis
- Medium complexity (0.3-0.7): Balance breadth and depth appropriately
- Low complexity (<0.3): Ensure basic requirements are met efficiently

If context isolation is active, pay special attention to potential information fragmentation and ensure continuity across isolated contexts.
"""


def get_integration_reflection_prompt(
    plan_update_needed: bool, researcher_context: Dict[str, Any]
) -> str:
    """
    Generate a prompt for reflection integration with existing components.

    Args:
        plan_update_needed: Whether plan updates are needed
        researcher_context: Current researcher context information

    Returns:
        str: Integration reflection prompt
    """
    return f"""
Reflection Integration Analysis

Current Context:
- Plan Update Needed: {plan_update_needed}
- Researcher Context: {researcher_context}

Provide reflection analysis that considers:
1. How findings integrate with the current research plan
2. Whether plan modifications are necessary based on new insights
3. How to optimize researcher context for better outcomes
4. Recommendations for workflow adjustments

Ensure your recommendations are compatible with the existing DeerFlow architecture and can be implemented without disrupting ongoing research processes.
"""


def get_metrics_reflection_prompt(session_metrics: Dict[str, Any]) -> str:
    """
    Generate a prompt for metrics-based reflection analysis.

    Args:
        session_metrics: Current session performance metrics

    Returns:
        str: Metrics reflection prompt
    """
    return f"""
Metrics-Based Reflection Analysis

Session Performance Metrics:
{session_metrics}

Analyze the research performance and provide insights on:
1. Efficiency of current research approach
2. Quality trends in research outputs
3. Resource utilization optimization
4. Potential bottlenecks or improvement areas

Provide specific, measurable recommendations for enhancing research performance based on the metrics data.
"""
