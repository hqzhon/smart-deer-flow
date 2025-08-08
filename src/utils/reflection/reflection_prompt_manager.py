from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from ..template import apply_prompt_template
from ...report_quality.i18n import Language, I18nManager


class ReflectionPromptManager:
    """Manages multilingual prompt templates for reflection mechanism"""

    def __init__(self, i18n_manager: Optional[I18nManager] = None):
        self.i18n_manager = i18n_manager or I18nManager()
        self.template_dir = Path(__file__).parent

    def get_reflection_analysis_prompt(
        self,
        research_topic: str,
        current_step_index: int,
        total_steps: int,
        current_reflection_loop: int,
        max_reflection_loops: int,
        steps_summary: str,
        results_summary: str,
        observations_summary: str,
        language: Language = Language.EN_US,
    ) -> str:
        """Get reflection analysis prompt

        Args:
            research_topic: Research topic
            current_step_index: Current step index
            total_steps: Total number of steps
            current_reflection_loop: Current reflection loop
            max_reflection_loops: Maximum reflection loops
            steps_summary: Summary of completed steps
            results_summary: Summary of execution results
            observations_summary: Summary of current observations
            language: Target language

        Returns:
            Formatted reflection analysis prompt
        """
        # Set language environment
        locale = "zh-CN" if language == Language.ZH_CN else "en-US"

        # Prepare template variables
        template_vars = {
            "CURRENT_TIME": datetime.now().isoformat(),
            "locale": locale,
            "research_topic": research_topic,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_step_index": current_step_index,
            "total_steps": total_steps,
            "current_reflection_loop": current_reflection_loop,
            "max_reflection_loops": max_reflection_loops,
            "steps_summary": steps_summary,
            "results_summary": results_summary,
            "observations_summary": observations_summary,
        }

        # Apply template using the project's template system
        # Convert template_vars to AgentState format
        agent_state = {**template_vars, "messages": []}
        # Always use the same template file, language is controlled by locale variable
        template_name = "reflection_analysis"
        result = apply_prompt_template(template_name, agent_state)
        # Extract the system prompt content
        if result and len(result) > 0 and "content" in result[0]:
            return result[0]["content"]
        return "Template rendering failed"

    def get_follow_up_queries_prompt(
        self,
        research_topic: str,
        primary_knowledge_gap: Optional[str],
        priority_areas: list,
        language: Language = Language.EN_US,
    ) -> str:
        """Get follow-up queries generation prompt

        Args:
            research_topic: Research topic
            primary_knowledge_gap: Primary knowledge gap to address
            priority_areas: Priority focus areas
            language: Target language

        Returns:
            Formatted follow-up queries generation prompt
        """
        # Set language environment
        locale = "zh-CN" if language == Language.ZH_CN else "en-US"

        # Prepare template variables
        template_vars = {
            "CURRENT_TIME": datetime.now().isoformat(),
            "locale": locale,
            "research_topic": research_topic,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "primary_knowledge_gap": primary_knowledge_gap,
            "priority_areas": priority_areas,
        }

        # Apply template using the project's template system
        # Convert template_vars to AgentState format
        agent_state = {**template_vars, "messages": []}
        result = apply_prompt_template("reflection_follow_up_queries", agent_state)
        # Extract the system prompt content
        if result and len(result) > 0 and "content" in result[0]:
            return result[0]["content"]
        return "Template rendering failed"

    def get_quality_assessment_labels(
        self, language: Language = Language.EN_US
    ) -> Dict[str, str]:
        """Get quality assessment label text

        Args:
            language: Target language

        Returns:
            Quality assessment labels dictionary
        """
        if language == Language.ZH_CN:
            return {
                "completeness": "Completeness",
                "accuracy": "Accuracy",
                "depth": "Depth",
                "relevance": "Relevance",
                "currency": "Currency",
                "overall_score": "Overall Score",
            }
        else:
            return {
                "completeness": "Completeness",
                "accuracy": "Accuracy",
                "depth": "Depth",
                "relevance": "Relevance",
                "currency": "Currency",
                "overall_score": "Overall Score",
            }

    def get_recommendation_categories(
        self, language: Language = Language.EN_US
    ) -> Dict[str, str]:
        """Get recommendation category label text

        Args:
            language: Target language

        Returns:
            Recommendation category labels dictionary
        """
        if language == Language.ZH_CN:
            return {
                "methodology": "Methodology Improvement",
                "sources": "Source Expansion",
                "analysis": "Analysis Deepening",
                "validation": "Result Validation",
                "presentation": "Presentation Optimization",
            }
        else:
            return {
                "methodology": "Methodology Improvement",
                "sources": "Source Expansion",
                "analysis": "Analysis Deepening",
                "validation": "Result Validation",
                "presentation": "Presentation Optimization",
            }
